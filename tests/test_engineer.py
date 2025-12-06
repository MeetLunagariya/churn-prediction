"""Tests for stateless engineered features."""

from __future__ import annotations

import pandas as pd

from churn.features.engineer import (
    SERVICE_COLUMNS,
    TENURE_LABELS,
    engineer_features,
)


def test_engineer_adds_expected_columns(synthetic_raw_df: pd.DataFrame) -> None:
    out = engineer_features(synthetic_raw_df)
    assert "tenure_bucket" in out.columns
    assert "charge_ratio" in out.columns
    assert "total_services" in out.columns
    # Original columns preserved
    for col in synthetic_raw_df.columns:
        assert col in out.columns


def test_engineer_does_not_mutate_input(synthetic_raw_df: pd.DataFrame) -> None:
    original_cols = list(synthetic_raw_df.columns)
    engineer_features(synthetic_raw_df)
    assert list(synthetic_raw_df.columns) == original_cols


def test_tenure_bucket_values_are_in_known_set(synthetic_raw_df: pd.DataFrame) -> None:
    out = engineer_features(synthetic_raw_df)
    assert set(out["tenure_bucket"].unique()).issubset(set(TENURE_LABELS))


def test_charge_ratio_avoids_division_by_zero() -> None:
    df = pd.DataFrame(
        {
            "tenure": [0, 1, 12],
            "MonthlyCharges": [50.0, 60.0, 70.0],
            "InternetService": ["No", "DSL", "Fiber optic"],
            **{col: ["No"] * 3 for col in SERVICE_COLUMNS},
        }
    )
    out = engineer_features(df)
    # tenure=0 -> charge_ratio = 50/(0+1) = 50; never inf
    assert out["charge_ratio"].iloc[0] == 50.0
    assert all(out["charge_ratio"].notna())


def test_total_services_counts_correctly() -> None:
    df = pd.DataFrame(
        {
            "tenure": [10],
            "MonthlyCharges": [50.0],
            "InternetService": ["Fiber optic"],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "OnlineSecurity": ["Yes"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["Yes"],
            "StreamingMovies": ["No"],
        }
    )
    out = engineer_features(df)
    # Yes: PhoneService, OnlineSecurity, StreamingTV (3) + InternetService not "No" (1) = 4
    assert out["total_services"].iloc[0] == 4
