"""Significance tests for "is model A really better than model B?".

A plain paired t-test over CV folds is too optimistic because the training sets of different
folds overlap, so fold scores are not independent (Dietterich 1998). We use:
  * the Nadeau & Bengio (2003) corrected resampled t-test on per-fold CV scores, and
  * McNemar's test on the two models' predictions over the shared held-out test set.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def corrected_resampled_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"mean_diff": float(d.mean()), "t": float("nan"), "p_value": 1.0 if d.mean() == 0 else 0.0}
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"mean_diff": float(d.mean()), "t": float(t), "p_value": float(p)}


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    ca, cb = pred_a == y_true, pred_b == y_true
    b = int(np.sum(ca & ~cb))   # A right, B wrong
    c = int(np.sum(~ca & cb))   # A wrong, B right
    if b + c == 0:
        return {"a_only_correct": b, "b_only_correct": c, "p_value": 1.0}
    if b + c < 25:
        p = stats.binomtest(b, b + c, 0.5).pvalue
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = stats.chi2.sf(chi2, df=1)
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}
