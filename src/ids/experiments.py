"""End-to-end experiments. Each step reads what earlier steps wrote to disk, so steps can be
re-run independently (and from notebooks) without repeating expensive work.

    prepare -> benchmark -> stats / deploy / explain / cross_dataset / loao -> report
"""
from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import datasets, deploy, plots, stats
from .config import Config, results_dir
from .evaluate import binary_metrics, evaluate
from .explain import explain
from .features import select_features
from .models import STRATEGIES, build_pipeline, fit_params, param_distributions, predict_scores, supports
from .preprocess import cap_benign, clean, drop_rare_classes
from .split import cv_splitter, stratified_sample, time_groups, train_test_indices

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- data

@dataclass
class Prepared:
    train: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    classes: list[str]
    meta: dict

    benign = 0  # classes are ordered with Benign first

    def xy(self, part: str) -> tuple[np.ndarray, np.ndarray]:
        df = self.train if part == "train" else self.test
        return df[self.features].to_numpy(np.float32), df["y"].to_numpy()


def _cache_dir(cfg: Config) -> Path:
    p = Path(cfg.paths.processed, f"{cfg.dataset.name}__{cfg.task}__{cfg.split.strategy}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def prepare(cfg: Config, force: bool = False) -> Prepared:
    """Load, clean, split and select features once; cache the result as parquet."""
    cache = _cache_dir(cfg)
    if not force and (cache / "meta.json").exists():
        meta = json.loads((cache / "meta.json").read_text())
        return Prepared(pd.read_parquet(cache / "train.parquet"), pd.read_parquet(cache / "test.parquet"),
                        meta["features"]["selected"], meta["classes"], meta)

    df = datasets.load(cfg.dataset.name, cfg.dataset.dir, cfg.dataset.attempted_as_benign)
    df, clean_report = clean(df, cfg.preprocess.drop_duplicates)
    df = cap_benign(df, cfg.preprocess.max_benign_rows, cfg.seed)

    rare = {}
    if cfg.task == "multiclass":
        df, rare = drop_rare_classes(df, cfg.preprocess.min_class_rows)
        classes = ["Benign"] + sorted(set(df["family"]) - {"Benign"})
        y = df["family"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    else:
        classes = ["Benign", "Attack"]
        y = df["is_attack"].to_numpy()
    df["y"] = y

    # stratify on family even for the binary task, so every attack family appears in both splits
    strat = df["family"].to_numpy()
    tr, te = train_test_indices(df, strat, cfg.split.strategy, cfg.split.test_size, cfg.seed,
                                cfg.split.time_block_minutes)
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)

    feats = datasets.feature_columns(df.drop(columns="y"))
    fs = select_features(train[feats], train["y"].to_numpy(), cfg)
    keep = fs["selected"] + datasets.META_COLS + ["y"]
    train, test = train[keep], test[keep]

    meta = {"dataset": cfg.dataset.name, "task": cfg.task, "split": cfg.split.strategy,
            "classes": classes, "features": fs, "clean": clean_report,
            "rare_classes_excluded": rare,
            "n_train": len(train), "n_test": len(test),
            "train_counts": train["family"].value_counts().to_dict(),
            "test_counts": test["family"].value_counts().to_dict()}
    train.to_parquet(cache / "train.parquet")
    test.to_parquet(cache / "test.parquet")
    _json(cache / "meta.json", meta)
    log.info("prepared %s: %d train / %d test, %d features", cache.name, len(train), len(test),
             len(fs["selected"]))
    return Prepared(train, test, fs["selected"], classes, meta)


# --------------------------------------------------------------------------- benchmark

def _search_jobs(name: str) -> int:
    return 1 if name in {"random_forest", "xgboost"} else 4  # tree models already use all cores


def tune(name: str, P: Prepared, cfg: Config) -> dict:
    """RandomizedSearchCV with a fixed budget, on a stratified subsample, imbalance strategy 'none'.
    The chosen hyper-parameters are then reused for every imbalance strategy so that the
    strategy comparison is not confounded by different tuning."""
    X, y = P.xy("train")
    cap = cfg.models.max_train_rows.get(name)
    n = min(x for x in (cfg.models.search.sample_rows, cap, len(y)) if x)
    idx = stratified_sample(y, n, cfg.seed)
    search = RandomizedSearchCV(
        build_pipeline(name, "none", cfg, len(P.classes)), param_distributions(name),
        n_iter=cfg.models.search.n_iter, scoring="f1_macro",
        cv=StratifiedKFold(cfg.models.search.cv_folds, shuffle=True, random_state=cfg.seed),
        n_jobs=_search_jobs(name), random_state=cfg.seed, error_score=np.nan)
    t0 = time.perf_counter()
    search.fit(X[idx], y[idx])
    log.info("tuned %s in %.0fs: %s (cv f1_macro=%.4f)", name, time.perf_counter() - t0,
             search.best_params_, search.best_score_)
    return {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()},
            "cv_f1_macro": float(search.best_score_), "rows": int(len(idx)),
            "seconds": time.perf_counter() - t0}


def _set_params(pipe, params: dict):
    fixed = {k: (tuple(v) if k.endswith("hidden_layer_sizes") else v) for k, v in params.items()}
    return pipe.set_params(**fixed)


def cross_validate(name: str, strategy: str, params: dict, X, y, groups, cfg: Config, n_classes: int) -> dict:
    """Per-fold scores on the training split, for the corrected resampled t-test."""
    idx = stratified_sample(y, cfg.split.cv_sample_rows, cfg.seed)
    Xc, yc = X[idx], y[idx]
    g = groups[idx] if groups is not None else None
    cv = cv_splitter(cfg.split.cv_folds, cfg.seed, g)
    folds = {"f1_macro": [], "bin_recall": [], "bin_fpr": [], "n_train": [], "n_test": []}
    for a, b in cv.split(Xc, yc):
        pipe = _set_params(build_pipeline(name, strategy, cfg, n_classes), params)
        pipe.fit(Xc[a], yc[a], **fit_params(pipe, name, strategy, yc[a]))
        pred = pipe.predict(Xc[b])
        bm = binary_metrics((yc[b] != 0).astype(int), (pred != 0).astype(int), None)
        folds["f1_macro"].append(float(f1_score(yc[b], pred, average="macro", zero_division=0)))
        folds["bin_recall"].append(bm["recall"])
        folds["bin_fpr"].append(bm["fpr"])
        folds["n_train"].append(int(len(a)))
        folds["n_test"].append(int(len(b)))
    return folds


def benchmark(cfg: Config, models: list[str] | None = None, strategies: list[str] | None = None,
              skip_cv: bool = False) -> pd.DataFrame:
    P = prepare(cfg)
    out = results_dir(cfg)
    (out / "models").mkdir(exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    Xtr, ytr = P.xy("train")
    Xte, yte = P.xy("test")
    groups = time_groups(P.train, cfg.split.time_block_minutes) if cfg.split.strategy == "time_block" else None

    tuning_path = out / "tuning.json"
    tuned = json.loads(tuning_path.read_text()) if tuning_path.exists() else {}
    rows = []
    for name in models or cfg.models.list:
        if cfg.models.search.enabled and name not in tuned:
            tuned[name] = tune(name, P, cfg)
            _json(tuning_path, tuned)
        params = tuned.get(name, {}).get("params", {})
        for strategy in strategies or cfg.imbalance.strategies:
            run = f"{name}__{strategy}"
            if not supports(name, strategy):
                log.warning("skip %s: %s has no class_weight or sample_weight support", run, name)
                continue
            log.info("=== %s", run)
            cap = stratified_sample(ytr, cfg.models.max_train_rows.get(name), cfg.seed)
            folds = {} if skip_cv else cross_validate(name, strategy, params, Xtr[cap], ytr[cap],
                                                      None if groups is None else groups[cap], cfg, len(P.classes))
            pipe = _set_params(build_pipeline(name, strategy, cfg, len(P.classes)), params)
            t0 = time.perf_counter()
            pipe.fit(Xtr[cap], ytr[cap], **fit_params(pipe, name, strategy, ytr[cap]))
            train_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred = pipe.predict(Xte)
            scores = predict_scores(pipe, Xte)
            predict_s = time.perf_counter() - t0
            summary, per_class, cm = evaluate(yte, pred, scores, P.classes, P.benign, cfg.evaluation.fpr_targets)

            record = {"model": name, "strategy": strategy, **summary, "train_seconds": train_s,
                      "predict_seconds_test": predict_s, "n_train_rows": int(len(cap)),
                      "params": params, "cv_folds": folds}
            _json(out / "runs" / f"{run}.json", record)
            per_class.to_csv(out / "runs" / f"{run}__per_class.csv", index=False)
            np.save(out / "runs" / f"{run}__cm.npy", cm)
            pd.DataFrame({"y_true": yte, "y_pred": pred, "is_attack": (yte != P.benign).astype(int),
                          "attack_score": 1 - scores[:, P.benign]}).to_parquet(out / "runs" / f"{run}__preds.parquet")
            joblib.dump(pipe, out / "models" / f"{run}.joblib", compress=3)
            plots.confusion(cm, P.classes, run, out / "runs" / f"{run}__cm.png")
            rows.append(record)
            log.info("%s: f1_macro=%.4f bin_fpr=%.5f bin_recall=%.4f train=%.0fs", run,
                     summary["f1_macro"], summary["bin_fpr"], summary["bin_recall"], train_s)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- analysis

def _runs(cfg: Config) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir(cfg, "runs").glob("*.json"))]


def run_stats(cfg: Config) -> pd.DataFrame:
    """Pairwise significance tests between models, within each imbalance strategy."""
    P = prepare(cfg)
    out = results_dir(cfg)
    runs = {(r["model"], r["strategy"]): r for r in _runs(cfg)}
    preds = {k: pd.read_parquet(out / "runs" / f"{k[0]}__{k[1]}__preds.parquet") for k in runs}
    yte = P.test["y"].to_numpy()
    rows = []
    for strategy in STRATEGIES:
        names = sorted(m for m, s in runs if s == strategy)
        for a, b in itertools.combinations(names, 2):
            ra, rb = runs[(a, strategy)], runs[(b, strategy)]
            row = {"strategy": strategy, "model_a": a, "model_b": b,
                   "f1_macro_a": ra["f1_macro"], "f1_macro_b": rb["f1_macro"]}
            fa, fb = ra.get("cv_folds", {}).get("f1_macro"), rb.get("cv_folds", {}).get("f1_macro")
            if fa and fb and len(fa) == len(fb):
                n_tr, n_te = np.mean(ra["cv_folds"]["n_train"]), np.mean(ra["cv_folds"]["n_test"])
                t = stats.corrected_resampled_ttest(np.array(fa), np.array(fb), n_tr, n_te)
                row.update({f"cv_{k}": v for k, v in t.items()})
            mc = stats.mcnemar(yte, preds[(a, strategy)]["y_pred"].to_numpy(),
                               preds[(b, strategy)]["y_pred"].to_numpy())
            row.update({f"mcnemar_{k}": v for k, v in mc.items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "significance.csv", index=False)
    return df


def run_deploy(cfg: Config) -> pd.DataFrame:
    P = prepare(cfg)
    out = results_dir(cfg)
    Xte, _ = P.xy("test")
    rows = []
    for r in _runs(cfg):
        run = f"{r['model']}__{r['strategy']}"
        pipe = joblib.load(out / "models" / f"{run}.joblib")
        prof = deploy.profile(pipe, Xte, cfg)
        rows.append({"model": r["model"], "strategy": r["strategy"], "f1_macro": r["f1_macro"],
                     "bin_fpr": r["bin_fpr"], "train_seconds": r["train_seconds"],
                     "n_train_rows": r["n_train_rows"], "size_mb": prof["size_bytes"] / 1e6, **prof})
        log.info("deploy %s: %.3f ms/flow, %.1f MB", run, prof["latency_median_ms"], prof["size_bytes"] / 1e6)
    df = pd.DataFrame(rows)
    df.to_csv(out / "deployability.csv", index=False)
    if len(df):
        plots.latency_vs_f1(df, out / "latency_vs_f1.png")
    return df


def best_run(cfg: Config, model: str | None = None) -> dict:
    runs = [r for r in _runs(cfg) if model in (None, "best", r["model"])]
    if not runs:
        raise FileNotFoundError("no benchmark runs found; run `ids benchmark` first")
    return max(runs, key=lambda r: r["f1_macro"])


def run_explain(cfg: Config, model: str | None = None) -> dict:
    P = prepare(cfg)
    out = results_dir(cfg, "shap")
    r = best_run(cfg, model or cfg.explain.model)
    run = f"{r['model']}__{r['strategy']}"
    pipe = joblib.load(results_dir(cfg) / "models" / f"{run}.joblib")
    bg = P.train.iloc[stratified_sample(P.train["y"].to_numpy(), cfg.explain.background_rows, cfg.seed)]
    ex = P.test.iloc[stratified_sample(P.test["y"].to_numpy(), cfg.explain.explain_rows, cfg.seed)]
    res = explain(pipe, ex[P.features], bg[P.features], P.classes, P.benign, out,
                  cfg.explain.local_examples, cfg.seed)
    res["run"] = run
    _json(out / "summary.json", res)
    return res


def run_cross_dataset(cfg: Config) -> pd.DataFrame:
    """Evaluate every trained model, without retraining, on an independent dataset (binary)."""
    P = prepare(cfg)
    out = results_dir(cfg)
    cd = cfg.cross_dataset
    target = datasets.load(cd.target, cd.target_dir, cfg.dataset.attempted_as_benign)
    target, report = clean(target, cfg.preprocess.drop_duplicates)
    missing = [f for f in P.features if f not in target.columns]
    if missing:
        raise ValueError(f"target dataset lacks {len(missing)} selected features, e.g. {missing[:5]}. "
                         "Cross-dataset testing needs the same feature extractor (see README).")
    Xt = target[P.features].to_numpy(np.float32)
    is_attack = target["is_attack"].to_numpy()
    fams = target["family"].to_numpy()
    rows, per_family = [], []
    for r in _runs(cfg):
        run = f"{r['model']}__{r['strategy']}"
        pipe = joblib.load(out / "models" / f"{run}.joblib")
        scores = predict_scores(pipe, Xt)
        pred_attack = (pipe.predict(Xt) != P.benign).astype(int)
        m = binary_metrics(is_attack, pred_attack, 1 - scores[:, P.benign], cfg.evaluation.fpr_targets)
        rows.append({"model": r["model"], "strategy": r["strategy"],
                     "in_dist_bin_f1": r["bin_f1"], "in_dist_bin_recall": r["bin_recall"],
                     "in_dist_bin_fpr": r["bin_fpr"],
                     **{f"target_{k}": v for k, v in m.items()},
                     "f1_gap": r["bin_f1"] - m["f1"]})
        for fam in np.unique(fams):
            sel = fams == fam
            rate = pred_attack[sel].mean()
            per_family.append({"model": r["model"], "strategy": r["strategy"], "family": fam,
                               "n": int(sel.sum()),
                               "flagged_as_attack": float(rate)})
    df = pd.DataFrame(rows)
    df.to_csv(out / f"cross_dataset__{cd.target}.csv", index=False)
    pd.DataFrame(per_family).to_csv(out / f"cross_dataset__{cd.target}__per_family.csv", index=False)
    _json(out / f"cross_dataset__{cd.target}__clean.json", report)
    return df


def run_loao(cfg: Config) -> pd.DataFrame:
    """Leave-one-attack-out: can a binary detector flag an attack family it never saw?

    This directly tests the 'ML generalises to unseen/zero-day attacks' claim.
    """
    bcfg = Config({**cfg, "task": "binary"})
    P = prepare(bcfg)
    out = results_dir(cfg)
    lc = cfg.loao
    params = {}
    tuning = results_dir(cfg).parent / f"{cfg.dataset.name}__binary__{cfg.split.strategy}" / "tuning.json"
    if tuning.exists():
        params = json.loads(tuning.read_text()).get(lc.model, {}).get("params", {})
    X_all = pd.concat([P.train, P.test], ignore_index=True)
    rows = []
    for fam in sorted(set(P.train["family"]) - {"Benign"}):
        tr = P.train[P.train["family"] != fam]
        idx = stratified_sample(tr["family"].to_numpy(), lc.max_train_rows, cfg.seed)
        tr = tr.iloc[idx]
        pipe = _set_params(build_pipeline(lc.model, lc.strategy, bcfg, 2), params)
        y = tr["y"].to_numpy()
        pipe.fit(tr[P.features].to_numpy(np.float32), y, **fit_params(pipe, lc.model, lc.strategy, y))

        def score(df):
            X = df[P.features].to_numpy(np.float32)
            return pipe.predict(X), 1 - predict_scores(pipe, X)[:, 0]

        unseen = X_all[X_all["family"] == fam]          # never in training, from either split
        benign = P.test[P.test["family"] == "Benign"]
        seen = P.test[(P.test["family"] != "Benign") & (P.test["family"] != fam)]
        pu, su = score(unseen)
        pb, sb = score(benign)
        ps, _ = score(seen)
        auc_df = binary_metrics(np.r_[np.zeros(len(pb)), np.ones(len(pu))].astype(int),
                                np.r_[pb, pu].astype(int), np.r_[sb, su])
        rows.append({"held_out": fam, "n_unseen": len(unseen),
                     "unseen_detection_rate": float(pu.mean()),
                     "seen_detection_rate": float(ps.mean()) if len(ps) else np.nan,
                     "benign_fpr": float(pb.mean()),
                     "unseen_vs_benign_roc_auc": auc_df.get("roc_auc", np.nan),
                     **{f"unseen_{k}": v for k, v in auc_df.items() if k.startswith("recall_at_fpr")}})
        log.info("LOAO %s: unseen detection %.3f", fam, pu.mean())
    df = pd.DataFrame(rows)
    df.to_csv(out / "leave_one_attack_out.csv", index=False)
    return df


def report(cfg: Config) -> pd.DataFrame:
    """Collect every run into summary.csv / summary.md and draw the comparison figures."""
    P = prepare(cfg)
    out = results_dir(cfg)
    runs = _runs(cfg)
    if not runs:
        raise FileNotFoundError("no benchmark runs found; run `ids benchmark` first")
    cols = ["model", "strategy", "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
            "roc_auc_ovr_macro", "bin_precision", "bin_recall", "bin_fpr", "bin_roc_auc", "bin_pr_auc",
            *[f"bin_recall_at_fpr_{t:g}" for t in cfg.evaluation.fpr_targets],
            "train_seconds", "n_train_rows", "classes_absent_from_test"]
    summary = pd.DataFrame(runs).reindex(columns=cols).sort_values("f1_macro", ascending=False)
    for r in runs:
        f = r.get("cv_folds", {}).get("f1_macro")
        summary.loc[(summary.model == r["model"]) & (summary.strategy == r["strategy"]),
                    "cv_f1_macro_mean±sd"] = f"{np.mean(f):.4f} ± {np.std(f, ddof=1):.4f}" if f else ""
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(summary.to_markdown(index=False, floatfmt=".4f"))

    per_class = {}
    for r in runs:
        run = f"{r['model']}__{r['strategy']}"
        pc = pd.read_csv(out / "runs" / f"{run}__per_class.csv").set_index("class")
        per_class[run] = pc
    for metric in ("recall", "f1"):
        table = pd.DataFrame({k: v[metric] for k, v in per_class.items()}).T
        plots.per_class_heatmap(table, metric, out / f"per_class_{metric}.png")
    plots.metric_bars(summary, ["f1_macro", "bin_fpr", f"bin_recall_at_fpr_{cfg.evaluation.fpr_targets[0]:g}"],
                      out / "model_comparison.png")
    preds = {f"{r['model']}__{r['strategy']}": pd.read_parquet(out / "runs" / f"{r['model']}__{r['strategy']}__preds.parquet")
             for r in runs}
    plots.roc_pr(preds, out / "roc_pr.png")
    _json(out / "dataset_summary.json", {k: P.meta[k] for k in
                                         ("classes", "n_train", "n_test", "train_counts", "test_counts",
                                          "rare_classes_excluded", "clean")})
    return summary
