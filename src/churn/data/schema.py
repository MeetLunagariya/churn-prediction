"""Data schemas for raw and processed Telco churn data.

Schemas double as documentation of the data contract: any drift in the
upstream feed surfaces as a validation error rather than a silent model bug.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

CONTRACTS = ("Month-to-month", "One year", "Two year")
PAYMENT_METHODS = (
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
)
INTERNET_SERVICES = ("DSL", "Fiber optic", "No")
YES_NO = ("Yes", "No")
YES_NO_NO_SERVICE = ("Yes", "No", "No internet service")
YES_NO_NO_PHONE = ("Yes", "No", "No phone service")


class RawChurnSchema(pa.DataFrameModel):
    """Schema for the raw IBM Telco churn dataset."""

    customerID: Series[str] = pa.Field(unique=True)
    gender: Series[str] = pa.Field(isin=("Male", "Female"))
    SeniorCitizen: Series[int] = pa.Field(isin=(0, 1))
    Partner: Series[str] = pa.Field(isin=YES_NO)
    Dependents: Series[str] = pa.Field(isin=YES_NO)
    tenure: Series[int] = pa.Field(ge=0, le=100)
    PhoneService: Series[str] = pa.Field(isin=YES_NO)
    MultipleLines: Series[str] = pa.Field(isin=YES_NO_NO_PHONE)
    InternetService: Series[str] = pa.Field(isin=INTERNET_SERVICES)
    OnlineSecurity: Series[str] = pa.Field(isin=YES_NO_NO_SERVICE)
    OnlineBackup: Series[str] = pa.Field(isin=YES_NO_NO_SERVICE)
    DeviceProtection: Series[str] = pa.Field(isin=YES_NO_NO_SERVICE)
    TechSupport: Series[str] = pa.Field(isin=YES_NO_NO_SERVICE)
    StreamingTV: Series[str] = pa.Field(isin=YES_NO_NO_SERVICE)
    StreamingMovies: Series[str] = pa.Field(isin=YES_NO_NO_SERVICE)
    Contract: Series[str] = pa.Field(isin=CONTRACTS)
    PaperlessBilling: Series[str] = pa.Field(isin=YES_NO)
    PaymentMethod: Series[str] = pa.Field(isin=PAYMENT_METHODS)
    MonthlyCharges: Series[float] = pa.Field(ge=0, le=200)
    TotalCharges: Series[float] = pa.Field(ge=0, nullable=True)
    Churn: Series[str] = pa.Field(isin=YES_NO)

    class Config:
        strict = False
        coerce = True
