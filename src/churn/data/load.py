"""Dataset loading with schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.data.schema import RawChurnSchema

RAW_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw(data_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Load and validate the raw Telco churn dataset.

    ``TotalCharges`` ships as a string column with whitespace for customers
    whose tenure is zero — coerce to NaN here so the schema accepts it.
    """
    path = Path(data_dir) / RAW_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run `make data` to download.")

    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return RawChurnSchema.validate(df, lazy=True)


def load_processed(path: Path | str = "data/processed/churn.parquet") -> pd.DataFrame:
    """Load the processed parquet snapshot used downstream of feature engineering."""
    return pd.read_parquet(path)
