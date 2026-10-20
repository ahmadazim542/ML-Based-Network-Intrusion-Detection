"""Phase 5 metrics.

`attack_score` = 1 - P(benign). It turns every model (binary or multi-class) into a
benign-vs-attack detector, which is what FPR, ROC/PR and "detection rate at a fixed FPR"
are defined on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def recall_at_fpr(is_attack: np.ndarray, attack_score: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Highest detection rate whose FPR does not exceed target_fpr. Returns (recall, threshold)."""
    fpr, tpr, thr = roc_curve(is_attack, attack_score)
    ok = np.where(fpr <= target_fpr)[0]
    i = ok[np.argmax(tpr[ok])]
    return float(tpr[i]), float(thr[i])


def binary_metrics(is_attack: np.ndarray, pred_attack: np.ndarray, attack_score: np.ndarray | None,
                   fpr_targets=(0.001, 0.01)) -> dict:
    tn, fp, fn, tp = confusion_matrix(is_attack, pred_attack, labels=[0, 1]).ravel()
    m = {
        "accuracy": accuracy_score(is_attack, pred_attack),
        "precision": precision_score(is_attack, pred_attack, zero_division=0),
        "recall": recall_score(is_attack, pred_attack, zero_division=0),
        "f1": f1_score(is_attack, pred_attack, zero_division=0),
        "fpr": fp / max(fp + tn, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if attack_score is not None and len(np.unique(is_attack)) == 2:
        m["roc_auc"] = roc_auc_score(is_attack, attack_score)
        m["pr_auc"] = average_precision_score(is_attack, attack_score)
        for t in fpr_targets:
            m[f"recall_at_fpr_{t:g}"] = recall_at_fpr(is_attack, attack_score, t)[0]
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, classes: list[str],
             benign: int, fpr_targets=(0.001, 0.01)) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Returns (summary metrics, per-class table, confusion matrix)."""
    labels = np.arange(len(classes))
    is_attack = (y_true != benign).astype(int)
    pred_attack = (y_pred != benign).astype(int)
    attack_score = 1.0 - scores[:, benign]

    # Macro averages run over the classes that actually occur in this test set. A class with no
    # test rows has undefined recall; counting it as 0 would drag every model down by 1/K.
    present = np.unique(y_true)
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=present, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", labels=present, zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", labels=present, zero_division=0)),
        "classes_absent_from_test": [classes[i] for i in labels if i not in set(present.tolist())],
    }
    if len(classes) > 2 and len(present) == len(classes):
        summary["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, scores, multi_class="ovr",
                                                           average="macro", labels=labels))
    # benign-vs-attack view of the same predictions (alert quality / alert fatigue)
    summary.update({f"bin_{k}": v for k, v in
                    binary_metrics(is_attack, pred_attack, attack_score, fpr_targets).items()})

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f, "support": s})
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return summary, per_class, cm
