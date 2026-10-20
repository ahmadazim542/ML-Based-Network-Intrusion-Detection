"""Phase 3: feature selection, fitted on the training split only.

Selection runs once, and the chosen subset is reused by every model, so that all classifiers
see the same inputs. It never sees the test split.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

from .split import stratified_sample

log = logging.getLogger(__name__)


def drop_near_constant(X: pd.DataFrame, tol: float) -> list[str]:
    var = X.astype(np.float64).var()
    return var[var <= tol].index.tolist()


def correlation_prune(X: pd.DataFrame, threshold: float) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Greedy pruning: walk features in order, drop any whose |Pearson r| with an already-kept
    feature exceeds `threshold`. Returns (kept, [(dropped, kept_partner, r), ...])."""
    corr = X.astype(np.float64).corr().abs().fillna(0.0).to_numpy()
    cols = list(X.columns)
    kept: list[int] = []
    dropped = []
    for i in range(len(cols)):
        hit = [j for j in kept if corr[i, j] > threshold]
        if hit:
            j = max(hit, key=lambda j: corr[i, j])
            dropped.append((cols[i], cols[j], float(corr[i, j])))
        else:
            kept.append(i)
    return [cols[i] for i in kept], dropped


def rank_features(X: pd.DataFrame, y: np.ndarray, top_k: int | None, seed: int) -> tuple[list[str], pd.Series]:
    """Recursive Feature Elimination with a Random Forest base estimator (proposal Phase 3)."""
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1,
                                class_weight="balanced_subsample", random_state=seed)
    n_select = min(top_k or X.shape[1], X.shape[1])
    rfe = RFE(rf, n_features_to_select=n_select, step=max(1, X.shape[1] // 10))
    rfe.fit(X.to_numpy(np.float32), y)
    ranking = pd.Series(rfe.ranking_, index=X.columns).sort_values()
    final_imp = pd.Series(rfe.estimator_.feature_importances_,
                          index=X.columns[rfe.support_]).sort_values(ascending=False)
    return final_imp.index.tolist(), ranking


def select_features(X_train: pd.DataFrame, y_train: np.ndarray, cfg) -> dict:
    fc = cfg.features
    const = drop_near_constant(X_train, cfg.preprocess.near_zero_variance)
    X = X_train.drop(columns=const)
    kept, corr_dropped = correlation_prune(X, fc.corr_threshold)
    idx = stratified_sample(y_train, fc.rank_sample_rows, cfg.seed)
    selected, ranking = rank_features(X[kept].iloc[idx], y_train[idx], fc.top_k, cfg.seed)
    log.info("features: %d raw -> %d non-constant -> %d after corr -> %d selected",
             X_train.shape[1], X.shape[1], len(kept), len(selected))
    return {
        "selected": selected,
        "constant_dropped": const,
        "correlation_dropped": corr_dropped,
        "rfe_ranking": ranking.to_dict(),
        "n_raw": X_train.shape[1],
    }
