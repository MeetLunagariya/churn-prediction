"""Cost-weighted decision-threshold optimization.

The default 0.5 cutoff is arbitrary: it maximizes accuracy under a
symmetric loss but real retention decisions are asymmetric. Two costs
are typically very different:

  - ``savings_per_save`` (S): expected revenue retained when a true
    churner is correctly flagged and retained.
  - ``retention_cost`` (R): cost of a retention action (offer/discount).

We use a simplified "utility vs. doing nothing" formulation:

  utility(threshold) = TP * (S - R) - FP * R

where TP and FP are computed from ``(y_proba >= threshold).astype(int)``.
The FN and TN contributions are zero relative to baseline because they
match what would happen without the model.

This module pairs with the cost matrix documented in
``docs/decisions/0003-cost-matrix.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class ThresholdSearch:
    """Result of a threshold sweep."""

    thresholds: NDArray[np.float64]
    utilities: NDArray[np.float64]
    best_threshold: float
    best_utility: float
    naive_utility: float  # utility at threshold = 0.5
    sensitivity_range: tuple[float, float]  # thresholds within 5% of best


def expected_utility(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    threshold: float,
    savings_per_save: float = 500.0,
    retention_cost: float = 50.0,
) -> float:
    """Expected utility (dollars retained) under the cost matrix.

    Assumes 100% retention success when an offer is accepted by a true
    churner — a deliberate simplification documented in the ADR.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    y_pred = (y_proba_arr >= threshold).astype(int)
    _, fp, _, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    return float(tp * (savings_per_save - retention_cost) - fp * retention_cost)


def optimize_threshold(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    savings_per_save: float = 500.0,
    retention_cost: float = 50.0,
    n_thresholds: int = 99,
    sensitivity_tolerance: float = 0.05,
) -> ThresholdSearch:
    """Sweep thresholds in (0, 1) and return the utility-maximizing one.

    Also reports:
      - utility at the naive 0.5 threshold for comparison
      - the range of thresholds whose utility is within
        ``sensitivity_tolerance`` (default 5%) of the optimum, which
        indicates how robust the decision is to small probability errors
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    thresholds = np.linspace(0.01, 0.99, n_thresholds, dtype=np.float64)
    utilities = np.array(
        [
            expected_utility(
                y_true_arr,
                y_proba_arr,
                t,
                savings_per_save=savings_per_save,
                retention_cost=retention_cost,
            )
            for t in thresholds
        ],
        dtype=np.float64,
    )

    best_idx = int(np.argmax(utilities))
    best_threshold = float(thresholds[best_idx])
    best_utility = float(utilities[best_idx])

    naive_utility = expected_utility(
        y_true_arr,
        y_proba_arr,
        0.5,
        savings_per_save=savings_per_save,
        retention_cost=retention_cost,
    )

    utility_floor = best_utility - sensitivity_tolerance * abs(best_utility)
    within = thresholds[utilities >= utility_floor]
    sensitivity_range = (float(within.min()), float(within.max()))

    return ThresholdSearch(
        thresholds=thresholds,
        utilities=utilities,
        best_threshold=best_threshold,
        best_utility=best_utility,
        naive_utility=naive_utility,
        sensitivity_range=sensitivity_range,
    )
