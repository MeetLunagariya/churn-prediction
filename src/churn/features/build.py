"""Feature engineering for churn prediction.

The baseline preprocessor is intentionally minimal: standardize the two
numeric features the EDA identified as primary signal (``tenure``,
``MonthlyCharges``) and one-hot encode the rest. ``TotalCharges`` is
*omitted* — it's ~deterministic in ``tenure * MonthlyCharges`` (median
delta $0, IQR ±$29) and adding it inflates collinearity without
contributing independent signal in the baseline LR.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
# customerID (identifier) and TotalCharges (collinear with tenure*charges)
# are intentionally dropped.
DROPPED_FEATURES = ["customerID", "TotalCharges"]


def build_preprocessor() -> ColumnTransformer:
    """Build the baseline ColumnTransformer.

    Returns a transformer compatible with sklearn ``Pipeline`` that emits
    a dense float array. ``handle_unknown='ignore'`` on the encoder means
    new categories at inference time pass through as all-zero rows for
    that feature rather than raising — a property we test for.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
