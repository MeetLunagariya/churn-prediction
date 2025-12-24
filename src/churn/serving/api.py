"""FastAPI service for churn prediction.

Endpoints:
  GET  /health        — liveness + model-loaded check
  GET  /metrics       — Prometheus counters and histograms
  GET  /model/info    — version, decision threshold, training metrics
  POST /predict       — single customer
  POST /predict/batch — up to 1000 customers
  POST /explain       — per-customer SHAP contributions

Pydantic v2 validation rejects malformed inputs with 422 at the request
boundary; the model never sees invalid categories.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from churn.serving.predictor import predictor
from churn.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Customer,
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    ModelInfo,
    PredictResponse,
)

log = structlog.get_logger()

REQUESTS = Counter("churn_requests_total", "Request count", ["endpoint", "status"])
LATENCY = Histogram(
    "churn_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
PREDICTIONS_POSITIVE = Counter(
    "churn_predictions_positive_total",
    "Number of predict calls that returned class=1 (above threshold)",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Eagerly load the model on startup so the first request is fast."""
    try:
        predictor.load()
        log.info("model_loaded", version=predictor.version)
    except FileNotFoundError as exc:
        log.warning("model_artifact_missing", error=str(exc))
    yield


app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Predict telco customer churn with calibrated probabilities, "
        "cost-aware decisioning, and SHAP-based per-customer explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_and_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    log_bound = log.bind(request_id=rid, path=request.url.path, method=request.method)
    log_bound.info("request_start")
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    LATENCY.labels(endpoint=request.url.path).observe(elapsed)
    REQUESTS.labels(endpoint=request.url.path, status=str(response.status_code)).inc()
    response.headers["x-request-id"] = rid
    log_bound.info(
        "request_end",
        status=response.status_code,
        elapsed_ms=round(elapsed * 1000, 2),
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        model_version=predictor.version if predictor.is_loaded else None,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    if not predictor.is_loaded:
        try:
            predictor.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ModelInfo(**predictor.info())


def _to_response(p: Any) -> PredictResponse:
    return PredictResponse(
        customerID=p.customer_id,
        churn_probability=p.probability,
        churn_class=p.class_label,
        threshold_used=p.threshold_used,
        recommended_action=p.recommended_action,
        model_version=p.model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_one(customer: Customer) -> PredictResponse:
    try:
        predictions = predictor.predict([customer.model_dump()])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    p = predictions[0]
    if p.class_label == 1:
        PREDICTIONS_POSITIVE.inc()
    return _to_response(p)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    try:
        predictions = predictor.predict([c.model_dump() for c in payload.customers])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    out = [_to_response(p) for p in predictions]
    PREDICTIONS_POSITIVE.inc(sum(p.churn_class for p in out))
    return BatchPredictResponse(predictions=out)


@app.post("/explain", response_model=ExplainResponse)
def explain(customer: Customer, top_k: int = 5) -> ExplainResponse:
    try:
        explanation = predictor.explain(customer.model_dump(), top_k=top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ExplainResponse(
        customerID=explanation.customer_id,
        churn_probability=explanation.probability,
        churn_class=explanation.class_label,
        base_value=explanation.base_value,
        top_features=[
            FeatureContribution(feature=name, shap_value=v, abs_value=abs_v)
            for (name, v, abs_v) in explanation.top_features
        ],
        model_version=explanation.model_version,
    )
