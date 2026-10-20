"""Phase 6: SHAP explanations for the best model (global ranking + per-flow waterfalls)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .models import transform_inputs


def _as_3d(values: np.ndarray, n_classes: int) -> np.ndarray:
    """Normalise SHAP output to (rows, features, classes)."""
    v = np.asarray(values)
    if v.ndim == 2:  # binary models give one column of log-odds for the positive class
        return np.stack([-v, v], axis=2) if n_classes == 2 else v[:, :, None]
    return v


def explain(pipe, X: pd.DataFrame, background: pd.DataFrame, classes: list[str], benign: int,
            out_dir: Path, n_local: int = 5, seed: int = 0) -> dict:
    """Write SHAP plots to out_dir and return the global importance table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clf = pipe[-1]
    # Explain the classifier on the inputs it actually sees (after log/scale for SVM/MLP/LR);
    # plots still show the raw feature values.
    Xt = pd.DataFrame(transform_inputs(pipe, X.to_numpy()), columns=X.columns, index=X.index)
    Bt = pd.DataFrame(transform_inputs(pipe, background.to_numpy()), columns=X.columns)

    if type(clf).__name__ in {"XGBClassifier", "RandomForestClassifier"}:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt, check_additivity=False)
        base = np.atleast_1d(explainer.expected_value)
    else:
        f = clf.predict_proba
        explainer = shap.KernelExplainer(f, shap.sample(Bt, 100, random_state=seed))
        sv = explainer.shap_values(Xt.iloc[:200], nsamples=200)
        Xt, X = Xt.iloc[:200], X.iloc[:200]
        base = np.atleast_1d(explainer.expected_value)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=2)
    sv = _as_3d(sv, len(classes))
    if len(base) == 1 and sv.shape[2] == 2:
        base = np.array([-base[0], base[0]])

    # Global: mean |SHAP| per feature, per class, and overall.
    imp = pd.DataFrame(np.abs(sv).mean(axis=0), index=X.columns, columns=classes[: sv.shape[2]])
    imp["overall"] = imp.mean(axis=1)
    imp = imp.sort_values("overall", ascending=False)
    imp.to_csv(out_dir / "shap_global_importance.csv")

    top = imp.head(20)
    ax = top.drop(columns="overall").iloc[::-1].plot.barh(stacked=True, figsize=(9, 7), width=0.8)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance (SHAP), stacked by class")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_bar.png", dpi=150)
    plt.close()

    # Beeswarm for the "attack" direction: SHAP of P(benign), negated, so red = pushes to attack.
    attack_sv = -sv[:, :, benign]
    shap.summary_plot(attack_sv, X, show=False, max_display=20)
    plt.title("Features pushing a flow towards 'attack' (negated benign SHAP)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm_attack.png", dpi=150)
    plt.close()

    # Local: why were some specific flows flagged?
    pred = clf.predict(Xt)
    flagged = np.where(pred != benign)[0]
    rng = np.random.default_rng(seed)
    picks = rng.choice(flagged, size=min(n_local, len(flagged)), replace=False) if len(flagged) else []
    local = []
    for k, i in enumerate(picks):
        c = int(pred[i])
        exp = shap.Explanation(values=sv[i, :, c], base_values=float(base[c] if len(base) > c else base[0]),
                               data=X.iloc[i].to_numpy(), feature_names=list(X.columns))
        shap.plots.waterfall(exp, max_display=12, show=False)
        plt.title(f"Flow #{X.index[i]} predicted {classes[c]}")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_local_{k}_{classes[c]}.png", dpi=150)
        plt.close()
        top_f = pd.Series(sv[i, :, c], index=X.columns).abs().sort_values(ascending=False).head(5)
        local.append({"row": int(X.index[i]), "predicted": classes[c], "top_features": top_f.index.tolist()})
    return {"global_top10": imp.head(10)["overall"].to_dict(), "local": local}
