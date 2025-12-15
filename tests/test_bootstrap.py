"""Tests for the bootstrap CI module."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from churn.models.bootstrap import bootstrap_metric


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    n = 500
    y_proba = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < y_proba).astype(int)

    result = bootstrap_metric(
        roc_auc_score, y_true, y_proba, n_bootstrap=300, metric_name="roc_auc"
    )
    assert result.lo <= result.point <= result.hi
    assert 0.5 < result.point < 1.0
    assert (result.hi - result.lo) > 0  # non-degenerate interval


def test_bootstrap_ci_narrows_with_more_data() -> None:
    rng = np.random.default_rng(0)

    def width(n: int) -> float:
        y_proba = rng.uniform(0, 1, n)
        y_true = (rng.uniform(0, 1, n) < y_proba).astype(int)
        r = bootstrap_metric(roc_auc_score, y_true, y_proba, n_bootstrap=300)
        return r.hi - r.lo

    w_small = width(100)
    w_large = width(2000)
    assert w_large < w_small  # bigger n -> tighter interval


def test_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(7)
    n = 300
    y_proba = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < y_proba).astype(int)
    a = bootstrap_metric(roc_auc_score, y_true, y_proba, n_bootstrap=200, seed=42)
    b = bootstrap_metric(roc_auc_score, y_true, y_proba, n_bootstrap=200, seed=42)
    assert a.lo == b.lo
    assert a.hi == b.hi
