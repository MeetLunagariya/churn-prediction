"""Evaluation metrics for binary churn classifiers.

Headlines: PR-AUC (honest under class imbalance), ROC-AUC (ranking),
Brier (calibration + sharpness combined), and F1 at a fixed 0.5
threshold. Cost-weighted threshold optimization arrives in Week 3.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


class BinaryMetrics(TypedDict):
    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float
    f1_at_0_5: float
    positive_rate: float
    n: int


def compute_metrics(y_true: ArrayLike, y_proba: ArrayLike) -> BinaryMetrics:
    """Compute headline metrics from labels and predicted positive-class probabilities."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    y_pred = (y_proba_arr >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_proba_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_proba_arr)),
        "brier": float(brier_score_loss(y_true_arr, y_proba_arr)),
        "log_loss": float(log_loss(y_true_arr, y_proba_arr)),
        "f1_at_0_5": float(f1_score(y_true_arr, y_pred)),
        "positive_rate": float(y_true_arr.mean()),
        "n": len(y_true_arr),
    }


def format_metrics(metrics: BinaryMetrics, prefix: str = "") -> str:
    """Human-readable single-line summary for logs."""
    p = f"{prefix} " if prefix else ""
    return (
        f"{p}n={metrics['n']:>5d} pos={metrics['positive_rate']:.3f} "
        f"roc_auc={metrics['roc_auc']:.3f} pr_auc={metrics['pr_auc']:.3f} "
        f"brier={metrics['brier']:.4f} f1@0.5={metrics['f1_at_0_5']:.3f}"
    )
