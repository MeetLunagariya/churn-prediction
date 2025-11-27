"""Tests for the data layer."""

from __future__ import annotations

import pandas as pd
import pandera.errors as pa_errors
import pytest

from churn.data.schema import RawChurnSchema
from churn.data.splits import stratified_split


def test_schema_accepts_valid_frame(synthetic_raw_df: pd.DataFrame) -> None:
    RawChurnSchema.validate(synthetic_raw_df, lazy=True)


def test_schema_rejects_invalid_contract(synthetic_raw_df: pd.DataFrame) -> None:
    bad = synthetic_raw_df.copy()
    bad.loc[0, "Contract"] = "Lifetime"
    with pytest.raises(pa_errors.SchemaErrors):
        RawChurnSchema.validate(bad, lazy=True)


def test_schema_rejects_duplicate_customer_ids(synthetic_raw_df: pd.DataFrame) -> None:
    bad = synthetic_raw_df.copy()
    bad.loc[1, "customerID"] = bad.loc[0, "customerID"]
    with pytest.raises(pa_errors.SchemaErrors):
        RawChurnSchema.validate(bad, lazy=True)


def test_split_is_deterministic(synthetic_raw_df: pd.DataFrame) -> None:
    a = stratified_split(synthetic_raw_df, seed=7)
    b = stratified_split(synthetic_raw_df, seed=7)
    pd.testing.assert_frame_equal(a.X_train, b.X_train)
    pd.testing.assert_series_equal(a.y_train, b.y_train)


def test_split_partitions_are_disjoint(synthetic_raw_df: pd.DataFrame) -> None:
    s = stratified_split(synthetic_raw_df, seed=7)
    train_ids = set(s.X_train["customerID"])
    val_ids = set(s.X_val["customerID"])
    test_ids = set(s.X_test["customerID"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(synthetic_raw_df)


def test_split_preserves_class_ratio(synthetic_raw_df: pd.DataFrame) -> None:
    s = stratified_split(synthetic_raw_df, seed=7)
    full_rate = (synthetic_raw_df["Churn"] == "Yes").mean()
    for part in (s.y_train, s.y_val, s.y_test):
        # tolerance is wide because n=20; on real data the gap is much tighter
        assert abs(part.mean() - full_rate) < 0.2
