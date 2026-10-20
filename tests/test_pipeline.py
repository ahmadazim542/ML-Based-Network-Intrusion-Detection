"""End-to-end smoke test on synthetic data shaped like the corrected CICIDS2017 CSVs."""
import json

import numpy as np
import pandas as pd
import pytest

from ids import experiments as ex
from ids.config import load_config, results_dir
from ids.datasets import label_to_family
from ids.sampling import SafeSMOTE
from ids.stats import corrected_resampled_ttest, mcnemar
from ids.synthetic import write_days

FAST = {
    "preprocess": {"max_benign_rows": 4000, "min_class_rows": 30},
    "split": {"cv_folds": 3, "cv_sample_rows": 3000},
    "features": {"top_k": 12, "rank_sample_rows": 3000},
    "imbalance": {"smote_min_rows": 400},
    "models": {"max_train_rows": {"svm": 1500, "mlp": 3000},
               "search": {"enabled": True, "n_iter": 2, "cv_folds": 2, "sample_rows": 1500}},
    "deploy": {"latency_samples": 50, "batch_size": 500},
    "explain": {"background_rows": 100, "explain_rows": 200, "local_examples": 2},
    "loao": {"max_train_rows": 3000},
}


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("ids")
    write_days(root / "raw17", rows_per_day=1500, seed=1)
    write_days(root / "raw18", rows_per_day=800, seed=2, shift=0.3)
    return load_config(None, **FAST,
                       paths={"raw": str(root), "processed": str(root / "proc"), "results": str(root / "res")},
                       dataset={"dir": str(root / "raw17")},
                       cross_dataset={"target_dir": str(root / "raw18")})


@pytest.mark.parametrize("label,family", [
    ("BENIGN", "Benign"), ("FTP-Patator", "BruteForce"), ("SSH-Patator - Attempted", "Benign"),
    ("DoS Hulk", "DoS"), ("DDoS", "DDoS"), ("Web Attack - Brute Force", "WebAttack"),
    ("Web Attack - SQL Injection", "WebAttack"), ("Infiltration - Portscan", "PortScan"),
    ("Portscan", "PortScan"), ("Botnet", "Botnet"), ("Heartbleed", "Heartbleed"),
    ("DDOS attack-HOIC", "DDoS"), ("DoS attacks-GoldenEye", "DoS"), ("Bot", "Botnet"),
    ("Brute Force -XSS", "WebAttack"), ("FTP-BruteForce", "BruteForce"), ("Infilteration", "Infiltration"),
])
def test_label_families(label, family):
    assert label_to_family(label) == family


def test_safe_smote_only_raises_minorities():
    rng = np.random.default_rng(0)
    X = np.exp(rng.normal(size=(1000, 4)))
    y = np.r_[np.zeros(950), np.ones(45), np.full(5, 2)]
    Xr, yr = SafeSMOTE(min_rows=200, random_state=0).fit_resample(X, y)
    counts = pd.Series(yr).value_counts()
    assert counts[0] == 950 and counts[1] == 200 and counts[2] == 200
    np.testing.assert_array_equal(Xr[:1000], X)
    assert (Xr > 0).all()  # synthetic rows stay in the original (positive) domain


def test_stats():
    t = corrected_resampled_ttest(np.array([.9, .91, .92]), np.array([.8, .81, .79]), 800, 200)
    assert t["p_value"] < 0.05
    y = np.array([0, 1, 1, 0])
    assert mcnemar(y, y, y)["p_value"] == 1.0


def test_end_to_end(cfg):
    P = ex.prepare(cfg)
    assert "Heartbleed" in P.meta["rare_classes_excluded"]           # 6 flows < min_class_rows
    assert P.classes[0] == "Benign"
    assert not {"src_ip", "dst_port", "flow_id", "timestamp"} & set(P.features)
    assert P.meta["clean"]["duplicate_rows_dropped"] > 0
    assert "subflow_fwd_packets" not in P.features or "total_fwd_packet" not in P.features

    df = ex.benchmark(cfg)
    assert len(df) == 14  # 5 models x 3 strategies, minus mlp+class_weight (no sample_weight)
    assert df["f1_macro"].min() > 0.3

    sig = ex.run_stats(cfg)
    assert {"cv_p_value", "mcnemar_p_value"} <= set(sig.columns)
    dep = ex.run_deploy(cfg)
    assert (dep["latency_median_ms"] > 0).all()
    shap_res = ex.run_explain(cfg)
    assert shap_res["global_top10"]
    cross = ex.run_cross_dataset(cfg)
    assert len(cross) == len(df) and "target_recall" in cross
    loao = ex.run_loao(cfg)
    assert "Heartbleed" in set(loao["held_out"])                     # rare family still tested here
    summary = ex.report(cfg)
    assert (results_dir(cfg) / "summary.md").exists()
    assert json.loads((results_dir(cfg) / "dataset_summary.json").read_text())["n_test"] > 0
    assert summary.iloc[0]["f1_macro"] >= summary.iloc[-1]["f1_macro"]


def test_time_block_split(cfg):
    from ids.config import Config
    c = Config({**cfg, "split": Config({**cfg.split, "strategy": "time_block"})})
    P = ex.prepare(c)
    tr_blocks = set(zip(P.train["day"], pd.to_datetime(P.train["timestamp"]).dt.floor("5min")))
    te_blocks = set(zip(P.test["day"], pd.to_datetime(P.test["timestamp"]).dt.floor("5min")))
    assert not tr_blocks & te_blocks
    assert set(P.test["family"]) == set(P.train["family"])          # no class stranded in train


def test_macro_metrics_ignore_classes_absent_from_test():
    from ids.evaluate import evaluate
    y = np.array([0, 0, 1, 1])
    scores = np.eye(3)[y]
    summary, per_class, _ = evaluate(y, y, scores, ["Benign", "A", "B"], 0)
    assert summary["f1_macro"] == 1.0 and summary["classes_absent_from_test"] == ["B"]
