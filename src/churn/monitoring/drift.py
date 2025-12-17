"""Feature- and prediction-drift detection.

PSI (Population Stability Index) for numeric features, chi-squared for
categoricals, and KS for prediction-score distributions. The
implementation is dependency-light by design — Evidently is mature but
its API churns frequently; PSI and KS are stable and well-understood.

PSI interpretation (industry convention):
  < 0.10  : no significant change
  0.10-0.25 : minor drift; investigate
  > 0.25  : major drift; retrain
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import chi2_contingency, ks_2samp

DriftSeverity = Literal["stable", "minor", "major"]


@dataclass(frozen=True)
class DriftReport:
    feature: str
    kind: Literal["numeric", "categorical", "score"]
    statistic: float  # PSI for numeric, chi2 for categorical, KS for score
    p_value: float | None  # chi-squared / KS p-value; None for raw PSI
    severity: DriftSeverity
    n_reference: int
    n_current: int


def population_stability_index(
    reference: ArrayLike,
    current: ArrayLike,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """PSI between two distributions on the same numeric variable.

    Bins are quantile-based on ``reference`` so empty reference bins
    cannot dominate the score.
    """
    reference_arr = np.asarray(reference, dtype=float)
    current_arr = np.asarray(current, dtype=float)
    bin_edges = np.quantile(reference_arr, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    ref_counts, _ = np.histogram(reference_arr, bins=bin_edges)
    cur_counts, _ = np.histogram(current_arr, bins=bin_edges)
    ref_prop = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_prop = cur_counts / max(cur_counts.sum(), 1) + eps
    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def _classify_psi(psi: float) -> DriftSeverity:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "minor"
    return "major"


def numeric_drift(reference: pd.Series, current: pd.Series) -> DriftReport:
    psi = population_stability_index(reference.to_numpy(), current.to_numpy())
    return DriftReport(
        feature=str(reference.name),
        kind="numeric",
        statistic=psi,
        p_value=None,
        severity=_classify_psi(psi),
        n_reference=len(reference),
        n_current=len(current),
    )


def categorical_drift(reference: pd.Series, current: pd.Series, alpha: float = 0.01) -> DriftReport:
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()
    levels = sorted(set(ref_counts.index) | set(cur_counts.index))
    table = np.array(
        [[int(ref_counts.get(level, 0)), int(cur_counts.get(level, 0))] for level in levels]
    )
    # Drop zero-only rows that confuse chi2_contingency
    keep = table.sum(axis=1) > 0
    table = table[keep]
    chi2, p_value, _, _ = chi2_contingency(table)
    severity: DriftSeverity = "stable" if p_value > alpha else "major"
    return DriftReport(
        feature=str(reference.name),
        kind="categorical",
        statistic=float(chi2),
        p_value=float(p_value),
        severity=severity,
        n_reference=len(reference),
        n_current=len(current),
    )


def score_drift(reference: ArrayLike, current: ArrayLike, alpha: float = 0.01) -> DriftReport:
    """KS-based drift on prediction-probability distributions."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    stat, p_value = ks_2samp(ref, cur)
    severity: DriftSeverity = "stable" if p_value > alpha else "major"
    return DriftReport(
        feature="prediction_score",
        kind="score",
        statistic=float(stat),
        p_value=float(p_value),
        severity=severity,
        n_reference=len(ref),
        n_current=len(cur),
    )


def reports_to_dataframe(reports: list[DriftReport]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": r.feature,
                "kind": r.kind,
                "statistic": r.statistic,
                "p_value": r.p_value,
                "severity": r.severity,
                "n_reference": r.n_reference,
                "n_current": r.n_current,
            }
            for r in reports
        ]
    )
