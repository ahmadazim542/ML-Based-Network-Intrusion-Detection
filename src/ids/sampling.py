"""SMOTE that is safe for CICIDS-style data, plus the signed-log transform it relies on."""
from __future__ import annotations

from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator


def signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def signed_expm1(Z):
    return np.sign(Z) * np.expm1(np.abs(Z))


class SafeSMOTE(BaseEstimator):
    """Oversample minority classes up to `min_rows` each.

    Differences from plain SMOTE, and why:
    * Only classes below `min_rows` are oversampled, rather than balancing everything to the
      majority size (which would mean millions of synthetic rows on CICIDS2017).
    * Neighbours are found in signed-log, standardised space. Raw flow features span ~10 orders
      of magnitude (Flow Bytes/s vs flag counts), so raw Euclidean neighbours would be decided
      by one or two features. Synthetic rows are mapped back to raw units, so tree models
      still see raw features and SHAP plots stay readable.
    * k_neighbors shrinks automatically when a class (inside a CV fold) has very few rows.
    * Used inside an imblearn Pipeline, so it only ever touches the training fold.
    """

    def __init__(self, min_rows: int = 20000, k_neighbors: int = 5, random_state=None):
        self.min_rows = min_rows
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        counts = Counter(y.tolist())
        target = {c: self.min_rows for c, n in counts.items() if 2 <= n < self.min_rows}
        if not target:
            return X, y
        k = min(self.k_neighbors, min(counts[c] for c in target) - 1)
        Z = signed_log1p(X)
        mu, sd = Z.mean(axis=0), Z.std(axis=0)
        sd[sd == 0] = 1.0
        sm = SMOTE(sampling_strategy=target, k_neighbors=k, random_state=self.random_state)
        Zr, yr = sm.fit_resample((Z - mu) / sd, y)
        Xr = signed_expm1(Zr * sd + mu)
        Xr[: len(X)] = X  # imblearn returns originals first; keep them bit-exact
        return Xr, yr
