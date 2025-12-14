"""Tests for cost-weighted threshold optimization."""

from __future__ import annotations

import numpy as np

from churn.models.threshold import expected_utility, optimize_threshold


def test_utility_at_zero_threshold_treats_everyone_as_positive() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.4, 0.6, 0.8])
    # Threshold 0 -> predict all positive -> 2 TP, 2 FP
    # utility = 2 * (S - R) - 2 * R = 2*(500-50) - 2*50 = 800
    u = expected_utility(y_true, y_proba, threshold=0.0)
    assert u == 800.0


def test_utility_at_one_threshold_treats_everyone_as_negative() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.4, 0.6, 0.8])
    # Threshold 1 -> predict all negative -> 0 TP, 0 FP -> utility 0
    u = expected_utility(y_true, y_proba, threshold=1.0)
    assert u == 0.0


def test_optimizer_finds_better_than_naive_when_costs_are_asymmetric() -> None:
    """When S >> R, the optimal threshold is below 0.5 (act on borderline)."""
    rng = np.random.default_rng(0)
    n = 500
    # Construct a clean dataset where probabilities are well-calibrated
    y_proba = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < y_proba).astype(int)

    result = optimize_threshold(y_true, y_proba, savings_per_save=500, retention_cost=50)
    assert result.best_utility >= result.naive_utility
    # With S=500 R=50 the optimum sits below 0.5
    assert result.best_threshold < 0.5


def test_optimizer_reports_sensitivity_range() -> None:
    rng = np.random.default_rng(0)
    n = 500
    y_proba = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < y_proba).astype(int)

    result = optimize_threshold(y_true, y_proba)
    lo, hi = result.sensitivity_range
    assert 0.0 < lo <= result.best_threshold <= hi < 1.0


def test_optimizer_threshold_grid_is_monotone() -> None:
    rng = np.random.default_rng(1)
    n = 200
    y_proba = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < y_proba).astype(int)
    result = optimize_threshold(y_true, y_proba, n_thresholds=99)
    assert result.thresholds[0] < result.thresholds[-1]
    assert len(result.thresholds) == 99
    assert len(result.utilities) == 99
