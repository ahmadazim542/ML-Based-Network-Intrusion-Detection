"""Phase 7: deployability - model size, single-flow latency and batch throughput.

Note for the report: these numbers are *classification* cost only. A flow-based IDS can only
classify a flow after CICFlowMeter closes it (FIN/RST or the 120 s flow timeout), so end-to-end
detection delay is dominated by flow aggregation, not by the model.
"""
from __future__ import annotations

import io
import logging
import time

import joblib
import numpy as np

log = logging.getLogger(__name__)


def model_size_bytes(model) -> int:
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return buf.getbuffer().nbytes


def single_flow_latency(predict, X: np.ndarray, n: int, warmup: int = 20) -> dict:
    """Latency of classifying one flow at a time, as an inline IDS would."""
    rows = X[: max(n, 1)]
    for i in range(min(warmup, len(rows))):
        predict(rows[i : i + 1])
    t = np.empty(len(rows))
    for i in range(len(rows)):
        t0 = time.perf_counter()
        predict(rows[i : i + 1])
        t[i] = time.perf_counter() - t0
    return {"latency_median_ms": float(np.median(t) * 1e3),
            "latency_p95_ms": float(np.percentile(t, 95) * 1e3),
            "latency_p99_ms": float(np.percentile(t, 99) * 1e3)}


def batch_throughput(predict, X: np.ndarray, batch_size: int) -> dict:
    rows = X[:batch_size]
    predict(rows[:100])
    t0 = time.perf_counter()
    predict(rows)
    dt = time.perf_counter() - t0
    return {"batch_flows_per_s": float(len(rows) / dt)}


def onnx_latency(pipe, X: np.ndarray, n: int) -> dict:
    """Best-effort ONNX export of tree models (the likely edge-deployment candidates).
    Returns {} when the model or its preprocessing cannot be converted."""
    try:
        import onnxruntime as ort
        clf = pipe[-1]
        if len(pipe) > 1 and not all(name == "smote" for name, _ in pipe.steps[:-1]):
            return {}  # preprocessing steps would need converting too; skip rather than mis-measure
        n_feat = X.shape[1]
        name = type(clf).__name__
        if name == "XGBClassifier":
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
            onx = convert_xgboost(clf, initial_types=[("input", FloatTensorType([None, n_feat]))])
        elif name == "RandomForestClassifier":
            from skl2onnx import to_onnx
            onx = to_onnx(clf, X[:1].astype(np.float32), options={id(clf): {"zipmap": False}})
        else:
            return {}
        data = onx.SerializeToString()
        sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        Xf = X.astype(np.float32)
        lat = single_flow_latency(lambda r: sess.run(None, {inp: r}), Xf, n)
        return {"onnx_size_bytes": len(data), **{f"onnx_{k}": v for k, v in lat.items()}}
    except Exception as e:  # conversion libraries are optional and version-sensitive
        log.warning("ONNX export skipped: %s", e)
        return {}


def profile(pipe, X: np.ndarray, cfg) -> dict:
    X = np.ascontiguousarray(X)
    out = {"size_bytes": model_size_bytes(pipe)}
    clf = pipe[-1]
    n_jobs = getattr(clf, "n_jobs", None)
    # One flow at a time is measured single-threaded: with n_jobs=-1, RF spends ~10 ms per call
    # just dispatching a thread pool, which would measure joblib rather than the model.
    if n_jobs is not None:
        clf.set_params(n_jobs=1)
    out.update(single_flow_latency(pipe.predict, X, cfg.deploy.latency_samples))
    if n_jobs is not None:
        clf.set_params(n_jobs=n_jobs)
    out.update(batch_throughput(pipe.predict, X, cfg.deploy.batch_size))
    out.update(onnx_latency(pipe, X, cfg.deploy.latency_samples))
    return out
