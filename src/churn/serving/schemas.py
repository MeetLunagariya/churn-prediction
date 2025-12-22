"""Pydantic v2 request and response schemas for the FastAPI service.

Field types use ``Literal`` instead of enums to keep the OpenAPI spec
concise and to surface invalid categorical values as 422s at the
request boundary — long before they reach the sklearn pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GenderType = Literal["Male", "Female"]
YesNoType = Literal["Yes", "No"]
SeniorType = Literal[0, 1]
PhoneLinesType = Literal["Yes", "No", "No phone service"]
InternetType = Literal["DSL", "Fiber optic", "No"]
ServiceType = Literal["Yes", "No", "No internet service"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
PaymentType = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class Customer(BaseModel):
    """One customer row, exactly mirroring the raw Telco schema."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    customerID: str = Field(..., min_length=1, max_length=32)
    gender: GenderType
    SeniorCitizen: SeniorType
    Partner: YesNoType
    Dependents: YesNoType
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: YesNoType
    MultipleLines: PhoneLinesType
    InternetService: InternetType
    OnlineSecurity: ServiceType
    OnlineBackup: ServiceType
    DeviceProtection: ServiceType
    TechSupport: ServiceType
    StreamingTV: ServiceType
    StreamingMovies: ServiceType
    Contract: ContractType
    PaperlessBilling: YesNoType
    PaymentMethod: PaymentType
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0)
    TotalCharges: float | None = Field(default=None, ge=0.0)


class PredictResponse(BaseModel):
    customerID: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_class: int = Field(..., ge=0, le=1)
    threshold_used: float = Field(..., ge=0.0, le=1.0)
    recommended_action: Literal["retain", "monitor"]
    model_version: str


class BatchPredictRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    abs_value: float


class ExplainResponse(BaseModel):
    customerID: str
    churn_probability: float
    churn_class: int
    base_value: float
    top_features: list[FeatureContribution]
    model_version: str


class ModelInfo(BaseModel):
    name: str
    version: str
    created_at: str
    model: str
    feature_set: str
    decision_threshold: float
    threshold_sensitivity_range: list[float]
    training_metrics: dict[str, float | int]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str | None = None
