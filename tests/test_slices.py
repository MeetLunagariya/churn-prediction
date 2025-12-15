"""Tests for slice-level metric computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.models.slices import compute_slice_metrics, slices_to_dataframe


def test_slice_metrics_split_by_category() -> None:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "contract": rng.choice(["A", "B", "C"], size=n),
        }
    )
    y_true = rng.integers(0, 2, size=n).astype(int)
    y_proba = rng.uniform(0, 1, size=n)
    slices = compute_slice_metrics(df, y_true, y_proba, "contract", min_samples=20)
    assert len(slices) == 3
    assert all(s.slice_name == "contract" for s in slices)
    assert all(s.n >= 20 for s in slices)


def test_slice_metrics_skip_too_small_buckets() -> None:
    df = pd.DataFrame({"x": ["a"] * 100 + ["b"] * 5})
    y_true = np.r_[np.zeros(50), np.ones(50), np.zeros(3), np.ones(2)].astype(int)
    y_proba = np.r_[np.linspace(0, 1, 100), np.linspace(0, 1, 5)]
    slices = compute_slice_metrics(df, y_true, y_proba, "x", min_samples=30)
    assert {s.slice_value for s in slices} == {"a"}


def test_slices_to_dataframe_has_expected_columns() -> None:
    df = pd.DataFrame({"x": ["a"] * 40 + ["b"] * 40})
    y_true = np.r_[np.zeros(20), np.ones(20), np.zeros(20), np.ones(20)].astype(int)
    y_proba = np.linspace(0, 1, 80)
    slices = compute_slice_metrics(df, y_true, y_proba, "x")
    table = slices_to_dataframe(slices)
    assert set(table.columns) == {
        "slice",
        "value",
        "n",
        "positive_rate",
        "roc_auc",
        "pr_auc",
    }
