"""Tests for the calibration module."""

from __future__ import annotations

import numpy as np

from churn.models.calibrate import expected_calibration_error, reliability_curve


def test_ece_zero_for_perfect_calibration() -> None:
    """If predicted probability equals empirical rate per bin, ECE = 0."""
    # Construct a dataset where each predicted-prob bin's empirical rate
    # exactly matches the bin center.
    rng = np.random.default_rng(0)
    n_per_bin = 1000
    probas, labels = [], []
    for bin_center in np.linspace(0.05, 0.95, 10):
        probas.extend([bin_center] * n_per_bin)
        labels.extend(rng.binomial(1, bin_center, size=n_per_bin).tolist())
    ece = expected_calibration_error(labels, probas, n_bins=10)
    assert ece < 0.02  # close to zero with this sample size


def test_ece_large_for_systematically_overconfident() -> None:
    """A model that always predicts 0.9 on a 50/50 dataset has ECE near 0.4."""
    n = 1000
    y_true = [0, 1] * (n // 2)
    y_proba = [0.9] * n
    ece = expected_calibration_error(y_true, y_proba, n_bins=10)
    assert ece > 0.3  # actual is 0.4 (|0.5 - 0.9|)


def test_reliability_curve_shapes() -> None:
    rng = np.random.default_rng(1)
    n = 500
    proba = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < proba).astype(int)
    confs, accs, counts = reliability_curve(y, proba, n_bins=10)
    assert confs.shape == (10,)
    assert accs.shape == (10,)
    assert counts.shape == (10,)
    assert counts.sum() == n
