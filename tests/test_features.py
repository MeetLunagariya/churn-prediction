"""Tests for the feature engineering layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.features.build import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)


def test_preprocessor_runs_on_synthetic_frame(synthetic_raw_df: pd.DataFrame) -> None:
    preprocessor = build_preprocessor()
    x = synthetic_raw_df.drop(columns=["Churn"])
    transformed = preprocessor.fit_transform(x)
    assert transformed.shape[0] == len(x)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)  # categoricals expanded


def test_preprocessor_standardizes_numerics(synthetic_raw_df: pd.DataFrame) -> None:
    preprocessor = build_preprocessor()
    x = synthetic_raw_df.drop(columns=["Churn"])
    transformed = preprocessor.fit_transform(x)
    feature_names = preprocessor.get_feature_names_out()
    for col in NUMERIC_FEATURES:
        idx = list(feature_names).index(col)
        column = transformed[:, idx]
        # StandardScaler: mean ~ 0, std ~ 1 (within fp tolerance)
        assert abs(column.mean()) < 1e-9
        assert abs(column.std() - 1.0) < 1e-3


def test_preprocessor_handles_unknown_category(
    synthetic_raw_df: pd.DataFrame,
) -> None:
    preprocessor = build_preprocessor()
    x = synthetic_raw_df.drop(columns=["Churn"])
    preprocessor.fit(x)

    unseen = x.iloc[[0]].copy()
    unseen["PaymentMethod"] = "Cryptocurrency"  # not in training categories
    out = preprocessor.transform(unseen)
    # PaymentMethod OHE columns should all be 0 for the unknown category
    feature_names = preprocessor.get_feature_names_out()
    payment_cols = [i for i, name in enumerate(feature_names) if name.startswith("PaymentMethod_")]
    assert payment_cols, "expected one-hot columns for PaymentMethod"
    assert np.allclose(out[0, payment_cols], 0.0)


def test_preprocessor_drops_unused_columns(synthetic_raw_df: pd.DataFrame) -> None:
    """customerID and TotalCharges must not appear in the output."""
    preprocessor = build_preprocessor()
    x = synthetic_raw_df.drop(columns=["Churn"])
    preprocessor.fit(x)
    out_names = list(preprocessor.get_feature_names_out())
    assert not any(name.startswith("customerID") for name in out_names)
    assert "TotalCharges" not in out_names


def test_feature_lists_are_consistent_with_schema() -> None:
    """Guard: feature lists shouldn't accidentally drift from the schema."""
    overlap = set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert not overlap, f"feature appears in both lists: {overlap}"
