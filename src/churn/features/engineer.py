"""Stateless engineered features.

These are deterministic transformations of the raw columns — no fitted
state, no risk of train/inference drift. The pipeline applies them at
both train and prediction time identically, so they are safe to compute
outside the sklearn ColumnTransformer.
"""

from __future__ import annotations

import pandas as pd

TENURE_BUCKETS: list[float] = [-0.1, 6, 12, 24, 48, 72]
TENURE_LABELS: list[str] = ["0-6", "7-12", "13-24", "25-48", "49+"]

# Service columns whose affirmative value implies the customer opted in.
# Note "No phone service" / "No internet service" are *negatives* — we count
# only true affirmatives and any-internet for InternetService.
SERVICE_COLUMNS: list[str] = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered columns appended.

    Adds:
      - ``tenure_bucket``: 5-bucket discretization motivated by the EDA
        (0-6, 7-12, 13-24, 25-48, 49+ months).
      - ``charge_ratio``: ``MonthlyCharges / (tenure + 1)`` — a proxy for
        cost-per-month-of-relationship; tenure+1 avoids div-by-zero.
      - ``total_services``: count of opted-in services (0-9). New customers
        with high service count are a high-risk segment per the EDA.
    """
    out = df.copy()

    out["tenure_bucket"] = pd.cut(
        out["tenure"],
        bins=TENURE_BUCKETS,
        labels=TENURE_LABELS,
    ).astype(str)

    out["charge_ratio"] = out["MonthlyCharges"] / (out["tenure"] + 1)

    services = (out[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    services = services + (out["InternetService"] != "No").astype(int)
    out["total_services"] = services.astype(int)

    return out
