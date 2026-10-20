"""Phase 4: the five classifiers, their search spaces and the shared pipeline wrapper."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import has_fit_parameter
from xgboost import XGBClassifier

from .sampling import SafeSMOTE, signed_log1p

STRATEGIES = ("none", "class_weight", "smote")


@dataclass
class ModelSpec:
    name: str
    scaled: bool                 # needs signed-log + standardisation (distance/gradient models)
    native_class_weight: bool    # accepts class_weight=; otherwise balanced sample_weight is used
    params: dict = field(default_factory=dict)


def _make(name: str, seed: int, n_classes: int, class_weight: bool):
    cw = "balanced" if class_weight else None
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=cw, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed,
                                      class_weight="balanced_subsample" if class_weight else None)
    if name == "xgboost":
        return XGBClassifier(n_estimators=300, tree_method="hist", n_jobs=-1, random_state=seed,
                             objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                             eval_metric="mlogloss" if n_classes > 2 else "logloss")
    if name == "svm":
        # decision_function is used for scores; probability=True would cost 5x via internal CV
        return SVC(kernel="rbf", class_weight=cw, cache_size=2000, random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=200,
                             random_state=seed)
    raise ValueError(f"unknown model {name!r}")


SPECS = {
    "logreg": ModelSpec("logreg", scaled=True, native_class_weight=True,
                        params={"C": loguniform(1e-3, 1e2)}),
    "random_forest": ModelSpec("random_forest", scaled=False, native_class_weight=True, params={
        "n_estimators": randint(100, 400),
        "max_depth": [None, 20, 30, 40],
        "min_samples_split": randint(2, 11),
        "max_features": ["sqrt", 0.3, 0.5],
    }),
    "xgboost": ModelSpec("xgboost", scaled=False, native_class_weight=False, params={
        "learning_rate": loguniform(0.03, 0.3),
        "max_depth": randint(4, 11),
        "n_estimators": randint(150, 500),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.6, 0.4),
    }),
    "svm": ModelSpec("svm", scaled=True, native_class_weight=True, params={
        "C": loguniform(1e-1, 1e3),
        "gamma": loguniform(1e-4, 1e0),
    }),
    "mlp": ModelSpec("mlp", scaled=True, native_class_weight=False, params={
        "hidden_layer_sizes": [(64,), (128, 64), (256, 128), (128, 64, 32)],
        "learning_rate_init": loguniform(1e-4, 1e-2),
        "alpha": loguniform(1e-6, 1e-2),
        "activation": ["relu", "tanh"],
    }),
}


def supports(name: str, strategy: str) -> bool:
    """class_weight needs either a class_weight= argument or sample_weight in fit()."""
    if strategy != "class_weight":
        return True
    spec = SPECS[name]
    return spec.native_class_weight or has_fit_parameter(_make(name, 0, 2, False), "sample_weight")


def build_pipeline(name: str, strategy: str, cfg, n_classes: int) -> Pipeline:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown imbalance strategy {strategy!r}")
    spec = SPECS[name]
    steps = []
    if strategy == "smote":
        steps.append(("smote", SafeSMOTE(min_rows=cfg.imbalance.smote_min_rows, random_state=cfg.seed)))
    if spec.scaled:
        # Flow features are extremely heavy-tailed; log first so the scaler is not dominated by outliers.
        steps.append(("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")))
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", _make(name, cfg.seed, n_classes, class_weight=strategy == "class_weight")))
    return Pipeline(steps)


def param_distributions(name: str) -> dict:
    return {f"clf__{k}": v for k, v in SPECS[name].params.items()}


def fit_params(pipe: Pipeline, name: str, strategy: str, y: np.ndarray) -> dict:
    """Balanced sample weights for models without a class_weight argument (XGBoost)."""
    if strategy != "class_weight" or SPECS[name].native_class_weight:
        return {}
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return {"clf__sample_weight": np.array([w[v] for v in y], dtype=np.float32)}


def transform_inputs(pipe: Pipeline, X):
    """Apply the pipeline's predict-time preprocessing (samplers are training-only)."""
    for _, step in pipe.steps[:-1]:
        if not hasattr(step, "fit_resample"):
            X = step.transform(X)
    return X


def predict_scores(pipe: Pipeline, X) -> np.ndarray:
    """(n, n_classes) class scores. SVC has no probabilities, so its decision values are
    softmax-normalised: monotone per class, enough for ROC/PR curves, not calibrated."""
    if hasattr(pipe, "predict_proba") and not isinstance(pipe[-1], SVC):
        return pipe.predict_proba(X)
    d = pipe.decision_function(X)
    if d.ndim == 1:
        d = np.column_stack([-d, d])
    d = d - d.max(axis=1, keepdims=True)
    e = np.exp(d)
    return e / e.sum(axis=1, keepdims=True)
