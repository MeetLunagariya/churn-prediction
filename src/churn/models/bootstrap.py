"""Bootstrap confidence intervals for binary-classification metrics.

Reporting metrics without intervals is a portfolio red flag: a single
point estimate hides whether a 1-point AUC gap is meaningful or noise.
This module produces percentile bootstrap CIs over arbitrary metric
functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

MetricFn = Callable[[NDArray[np.int64], NDArray[np.float64]], float]


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    point: float
    mean: float
    lo: float
    hi: float
    ci: float
    n_bootstrap: int

    def format(self) -> str:
        return (
            f"{self.metric}: {self.point:.4f} "
            f"({int(self.ci * 100)}% CI [{self.lo:.4f}, {self.hi:.4f}])"
        )


def bootstrap_metric(
    metric_fn: MetricFn,
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    metric_name: str = "metric",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI for a (y_true, y_proba) -> float metric.

    Resamples rows with replacement; degenerate folds (all-positive or
    all-negative) are skipped silently — they make ROC-AUC undefined
    but happen rarely on real datasets.
    """
    y_true_arr = np.asarray(y_true).astype(np.int64)
    y_proba_arr = np.asarray(y_proba).astype(np.float64)
    n = len(y_true_arr)

    point = float(metric_fn(y_true_arr, y_proba_arr))

    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_t = y_true_arr[idx]
        if y_t.min() == y_t.max():
            continue  # skip degenerate fold
        try:
            scores.append(float(metric_fn(y_t, y_proba_arr[idx])))
        except ValueError:
            continue

    scores_arr = np.array(scores, dtype=np.float64)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(scores_arr, alpha))
    hi = float(np.quantile(scores_arr, 1.0 - alpha))
    return BootstrapResult(
        metric=metric_name,
        point=point,
        mean=float(scores_arr.mean()),
        lo=lo,
        hi=hi,
        ci=ci,
        n_bootstrap=len(scores_arr),
    )
