"""Train/test splitting and CV fold generation."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

log = logging.getLogger(__name__)


def time_groups(df: pd.DataFrame, minutes: int) -> np.ndarray:
    """Group id = (day, timestamp bucket). Flows in the same bucket stay on the same side."""
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    bucket = ts.dt.floor(f"{minutes}min").astype(str)
    return pd.factorize(df["day"].astype(str) + "_" + bucket)[0]


def train_test_indices(df: pd.DataFrame, y: np.ndarray, strategy: str, test_size: float,
                       seed: int, block_minutes: int = 5) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(df))
    if strategy == "random":
        return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    if strategy == "time_block":
        return _time_block_split(idx, y, time_groups(df, block_minutes), test_size, seed)
    raise ValueError(f"unknown split strategy {strategy!r}")


def _time_block_split(idx, y, groups, test_size: float, seed: int, rounds: int = 10):
    """Group-aware split that keeps every class on both sides whenever possible.

    Short attacks occupy only a handful of time blocks (DDoS in CICIDS2017 spans ~4 five-minute
    blocks), so a single StratifiedGroupKFold fold can put all of a class in train. Try several
    shuffles x folds and keep the one that (1) misses the fewest classes that *could* be on both
    sides (>= 2 blocks), then (2) has a test fraction closest to `test_size`.
    """
    y = np.asarray(y)
    blocks = pd.Series(groups).groupby(y).nunique()
    splittable = set(blocks[blocks >= 2].index)
    n_splits = max(2, round(1 / test_size))
    best = None
    for r in range(rounds):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for tr, te in sgkf.split(idx, y, groups):
            missing = len(splittable - set(y[te])) + len(splittable - set(y[tr]))
            key = (missing, abs(len(te) / len(y) - test_size))
            if best is None or key < best[0]:
                best = (key, tr, te)
        if best[0][0] == 0 and best[0][1] < 0.03:
            break
    if best[0][0]:
        log.warning("time_block split: %d splittable classes still missing from one side", best[0][0])
    unsplittable = sorted(set(np.unique(y)) - splittable)
    if unsplittable:
        log.warning("time_block split: classes in a single time block cannot be on both sides: %s", unsplittable)
    return best[1], best[2]


def cv_splitter(n_folds: int, seed: int, groups: np.ndarray | None = None):
    """Folds for X/y; grouped when time blocks are given so CV matches the test-split rule."""
    if groups is None:
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    class _Grouped(StratifiedGroupKFold):
        def split(self, X, y=None, groups_=None):
            return super().split(X, y, groups)

    return _Grouped(n_splits=n_folds, shuffle=True, random_state=seed)


def stratified_sample(y: np.ndarray, n: int | None, seed: int, min_per_class: int = 200) -> np.ndarray:
    """Indices of a proportional subsample of size ~n that keeps at least `min_per_class`
    rows of every class (or all of them if the class is smaller).

    Used to cap expensive models (RBF-SVM, hyper-parameter search) without wiping out the
    rare classes, while still preserving the natural class imbalance.
    """
    idx = np.arange(len(y))
    if n is None or n >= len(y):
        return idx
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    floor = np.minimum(counts, min_per_class)
    remaining = max(n - floor.sum(), 0)
    extra = np.floor((counts - floor) / max((counts - floor).sum(), 1) * remaining).astype(int)
    take = np.minimum(counts, floor + extra)
    out = []
    for c, k in zip(classes, take):
        members = idx[y == c]
        out.append(rng.choice(members, size=k, replace=False))
    return np.sort(np.concatenate(out))
