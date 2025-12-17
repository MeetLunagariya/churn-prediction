"""Tests for drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.monitoring.drift import (
    categorical_drift,
    numeric_drift,
    population_stability_index,
    score_drift,
)


def test_psi_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    psi = population_stability_index(x, x.copy())
    assert psi < 1e-3


def test_psi_grows_with_distribution_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    near = rng.normal(0.1, 1, 2000)
    far = rng.normal(2.0, 1, 2000)
    assert population_stability_index(ref, far) > population_stability_index(ref, near)


def test_numeric_drift_classifies_severity() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 2000), name="tenure")
    cur_stable = pd.Series(rng.normal(0, 1, 2000), name="tenure")
    cur_major = pd.Series(rng.normal(3, 1, 2000), name="tenure")
    assert numeric_drift(ref, cur_stable).severity == "stable"
    assert numeric_drift(ref, cur_major).severity == "major"


def test_categorical_drift_detects_shift() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.choice(["A", "B", "C"], p=[0.5, 0.3, 0.2], size=2000), name="Contract")
    cur_same = pd.Series(rng.choice(["A", "B", "C"], p=[0.5, 0.3, 0.2], size=2000), name="Contract")
    cur_diff = pd.Series(rng.choice(["A", "B", "C"], p=[0.1, 0.1, 0.8], size=2000), name="Contract")
    assert categorical_drift(ref, cur_same).severity == "stable"
    assert categorical_drift(ref, cur_diff).severity == "major"


def test_score_drift_handles_identical_scores() -> None:
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 2000)
    report = score_drift(scores, scores.copy())
    assert report.severity == "stable"
    assert report.feature == "prediction_score"
