"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """A tiny in-memory frame that conforms to RawChurnSchema.

    Used for fast unit tests so we don't depend on the real dataset being
    downloaded in CI.
    """
    n = 20
    return pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "gender": ["Male", "Female"] * (n // 2),
            "SeniorCitizen": [0, 1] * (n // 2),
            "Partner": ["Yes", "No"] * (n // 2),
            "Dependents": ["No"] * n,
            "tenure": list(range(1, n + 1)),
            "PhoneService": ["Yes"] * n,
            "MultipleLines": ["No"] * n,
            "InternetService": ["Fiber optic"] * n,
            "OnlineSecurity": ["No"] * n,
            "OnlineBackup": ["No"] * n,
            "DeviceProtection": ["No"] * n,
            "TechSupport": ["No"] * n,
            "StreamingTV": ["No"] * n,
            "StreamingMovies": ["No"] * n,
            "Contract": ["Month-to-month"] * 14 + ["One year"] * 3 + ["Two year"] * 3,
            "PaperlessBilling": ["Yes"] * n,
            "PaymentMethod": ["Electronic check"] * n,
            "MonthlyCharges": [50.0 + i for i in range(n)],
            "TotalCharges": [100.0 * (i + 1) for i in range(n)],
            "Churn": ["Yes"] * 6 + ["No"] * 14,
        }
    )
