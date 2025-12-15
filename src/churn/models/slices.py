"""Slice-level metric computation.

Used to surface subgroups where the model underperforms. Slices are
specified by a column in the *original* dataframe (not the preprocessed
feature matrix) plus a function that maps row -> bucket label.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class SliceMetric:
    slice_name: str
    slice_value: str
    n: int
    positive_rate: float
    roc_auc: float | None
    pr_auc: float | None


def _safe_metric(
    fn: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray, p: np.ndarray
) -> float | None:
    if len(np.unique(y)) < 2:
        return None  # degenerate slice
    return float(fn(y, p))


def compute_slice_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    slice_column: str,
    *,
    min_samples: int = 30,
) -> list[SliceMetric]:
    """Compute ROC-AUC and PR-AUC per bucket of ``slice_column``.

    Skips buckets with fewer than ``min_samples`` rows (CIs would be
    misleading) and buckets with only one class present.
    """
    if slice_column not in df.columns:
        raise KeyError(f"slice column {slice_column!r} not found in dataframe")

    out: list[SliceMetric] = []
    for value in sorted(df[slice_column].unique()):
        mask = (df[slice_column] == value).to_numpy()
        if int(mask.sum()) < min_samples:
            continue
        y_sub = y_true[mask]
        p_sub = y_proba[mask]
        out.append(
            SliceMetric(
                slice_name=slice_column,
                slice_value=str(value),
                n=int(mask.sum()),
                positive_rate=float(y_sub.mean()),
                roc_auc=_safe_metric(roc_auc_score, y_sub, p_sub),
                pr_auc=_safe_metric(average_precision_score, y_sub, p_sub),
            )
        )
    return out


def slices_to_dataframe(slices: list[SliceMetric]) -> pd.DataFrame:
    """Pivot slice metrics to a sortable dataframe for reporting."""
    return pd.DataFrame(
        [
            {
                "slice": s.slice_name,
                "value": s.slice_value,
                "n": s.n,
                "positive_rate": s.positive_rate,
                "roc_auc": s.roc_auc,
                "pr_auc": s.pr_auc,
            }
            for s in slices
        ]
    )
