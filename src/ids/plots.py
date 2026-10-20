"""Figures for the report. Every function writes a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

sns.set_theme(style="whitegrid", context="paper")


def confusion(cm: np.ndarray, classes: list[str], title: str, path: Path) -> Path:
    # Row-normalised: each row shows where that true class ended up (recall on the diagonal).
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(classes), 0.8 + 0.7 * len(classes)))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def per_class_heatmap(table: pd.DataFrame, metric: str, path: Path) -> Path:
    """table: rows = run (model/strategy), columns = class, values = metric."""
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * table.shape[1], 1.0 + 0.4 * table.shape[0]))
    sns.heatmap(table, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, cbar=False, ax=ax)
    ax.set_title(f"Per-class {metric}")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def metric_bars(summary: pd.DataFrame, metrics: list[str], path: Path) -> Path:
    long = summary.melt(id_vars=["model", "strategy"], value_vars=metrics, var_name="metric")
    g = sns.catplot(data=long, x="model", y="value", hue="strategy", col="metric", kind="bar",
                    col_wrap=min(3, len(metrics)), sharey=False, height=3, aspect=1.3)
    g.set_xticklabels(rotation=30)
    g.savefig(path, dpi=150)
    plt.close(g.figure)
    return path


def roc_pr(preds: dict[str, pd.DataFrame], path: Path) -> Path:
    """preds: run name -> DataFrame with is_attack, attack_score."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    # colour = model, line style = imbalance strategy (keeps 15 curves distinguishable)
    models = sorted({n.split("__")[0] for n in preds})
    colours = dict(zip(models, sns.color_palette("deep", len(models))))
    styles = {"none": "-", "class_weight": ":", "smote": "--"}
    for name, p in preds.items():
        model, _, strategy = name.partition("__")
        kw = {"color": colours[model], "linestyle": styles.get(strategy, "-"), "label": name}
        fpr, tpr, _ = roc_curve(p["is_attack"], p["attack_score"])
        a.plot(fpr, tpr, **kw)
        pr, rc, _ = precision_recall_curve(p["is_attack"], p["attack_score"])
        b.plot(rc, pr, **kw)
    a.set_xscale("log")
    a.set_xlim(1e-5, 1)
    a.set_xlabel("False-positive rate (log scale)")
    a.set_ylabel("Detection rate")
    a.set_title("ROC (benign vs attack)")
    b.set_xlabel("Recall")
    b.set_ylabel("Precision")
    b.set_title("Precision-Recall")
    a.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def latency_vs_f1(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=df, x="latency_median_ms", y="f1_macro", hue="model", style="strategy",
                    size="size_mb", sizes=(40, 400), ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Single-flow latency, median (ms, log scale)")
    ax.set_ylabel("Macro F1 (test)")
    ax.set_title("Accuracy vs deployability (bubble = model size)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
