"""Feature engineering for churn prediction.

The baseline preprocessor uses the raw columns identified by EDA as
primary signal (``tenure``, ``MonthlyCharges``, 16 categoricals).
``TotalCharges`` is *omitted* — it's ~deterministic in
``tenure * MonthlyCharges`` (median delta $0, IQR ±$29) and adding it
inflates collinearity without contributing independent signal in the LR
baseline.

``build_preprocessor(use_engineered=True)`` additionally consumes the
columns added by :func:`churn.features.engineer.engineer_features` —
``tenure_bucket``, ``charge_ratio``, ``total_services``.
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

# Added by engineer_features()
ENGINEERED_NUMERIC = ["charge_ratio", "total_services"]
ENGINEERED_CATEGORICAL = ["tenure_bucket"]


def build_preprocessor(use_engineered: bool = False) -> ColumnTransformer:
    """Build the preprocessing ColumnTransformer.

    Returns a transformer compatible with sklearn ``Pipeline`` that emits
    a dense float array. ``handle_unknown='ignore'`` on the encoder means
    new categories at inference time pass through as all-zero rows for
    that feature rather than raising — a property we test for.

    Parameters
    ----------
    use_engineered:
        If True, include the engineered columns added by
        :func:`engineer_features`. Caller is responsible for invoking
        that function before fitting.
    """
    numeric = list(NUMERIC_FEATURES)
    categorical = list(CATEGORICAL_FEATURES)
    if use_engineered:
        numeric = numeric + ENGINEERED_NUMERIC
        categorical = categorical + ENGINEERED_CATEGORICAL

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
