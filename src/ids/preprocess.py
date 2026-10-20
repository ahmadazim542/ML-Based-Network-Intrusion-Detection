"""Row-level cleaning that is safe to do before the train/test split.

Nothing here looks at feature *distributions* (scaling, variance, correlation, SMOTE): those
are fitted on the training split only, inside the model pipeline or in features.py.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .datasets import feature_columns

log = logging.getLogger(__name__)


def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """Drop rows with NaN/inf features and duplicate flows. Returns (df, report)."""
    report: dict = {"rows_in": len(df)}
    feats = feature_columns(df)

    # Columns that are entirely missing (e.g. a feature absent from one extractor version)
    empty = [c for c in feats if df[c].isna().all()]
    if empty:
        df = df.drop(columns=empty)
        feats = [c for c in feats if c not in empty]
    report["empty_columns_dropped"] = empty

    bad = ~np.isfinite(df[feats].to_numpy(dtype=np.float32)).all(axis=1)
    report["nan_inf_rows_dropped"] = int(bad.sum())
    df = df.loc[~bad]

    if drop_duplicates:
        # One 64-bit hash per feature vector is far cheaper than grouping on ~80 columns.
        h = pd.util.hash_pandas_object(df[feats], index=False)
        # Identical feature vectors with *different* labels cannot be classified correctly by
        # any model and indicate labelling noise: drop all copies.
        conflict = df.groupby(h.values)["family"].transform("nunique").to_numpy() > 1
        report["conflicting_label_rows_dropped"] = int(conflict.sum())
        df, h = df.loc[~conflict], h[~conflict]
        # Exact duplicates inflate scores when copies land on both sides of the split.
        dups = h.duplicated(keep="first").to_numpy()
        report["duplicate_rows_dropped"] = int(dups.sum())
        df = df.loc[~dups]

    report["rows_out"] = len(df)
    report["family_counts"] = df["family"].value_counts().to_dict()
    log.info("clean: %s", report)
    return df.reset_index(drop=True), report


def cap_benign(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    """Randomly keep at most `max_rows` benign flows; all attack flows are kept."""
    if not max_rows:
        return df
    benign = df.index[df["family"] == "Benign"]
    if len(benign) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    drop = rng.choice(benign, size=len(benign) - max_rows, replace=False)
    return df.drop(index=drop).reset_index(drop=True)


def drop_rare_classes(df: pd.DataFrame, min_rows: int) -> tuple[pd.DataFrame, dict]:
    """For the multi-class task, remove families too small to evaluate (e.g. Heartbleed ~11 flows).

    They are reported, not silently lost: the binary task and the leave-one-attack-out
    experiment still use them.
    """
    counts = df["family"].value_counts()
    rare = counts[counts < min_rows]
    return df[~df["family"].isin(rare.index)].reset_index(drop=True), rare.to_dict()
