"""API contract tests using FastAPI's TestClient.

These exercise the live app (including model loading on startup), so
the artifact at ``models/churn_v1.joblib`` must exist — produced by
``scripts/build_artifact.py``. CI runs that build step before testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from churn.serving.api import app

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_customer.json"
ARTIFACT_PATH = Path("models/churn_v1.joblib")
pytestmark = pytest.mark.skipif(
    not ARTIFACT_PATH.exists(),
    reason="run `make build` to produce models/churn_v1.joblib",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_customer() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def test_health_returns_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "1.0.0"


def test_model_info_returns_expected_keys(client: TestClient) -> None:
    r = client.get("/model/info")
    assert r.status_code == 200
    info = r.json()
    for key in (
        "name",
        "version",
        "created_at",
        "model",
        "feature_set",
        "decision_threshold",
        "threshold_sensitivity_range",
        "training_metrics",
    ):
        assert key in info, f"missing key {key}"
    assert 0.0 < info["decision_threshold"] < 1.0


def test_predict_returns_valid_response(client: TestClient, sample_customer: dict) -> None:
    r = client.post("/predict", json=sample_customer)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["customerID"] == sample_customer["customerID"]
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_class"] in (0, 1)
    assert body["recommended_action"] in ("retain", "monitor")
    assert body["model_version"] == "1.0.0"


def test_predict_rejects_invalid_contract(client: TestClient, sample_customer: dict) -> None:
    bad = sample_customer | {"Contract": "Lifetime"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_missing_field(client: TestClient, sample_customer: dict) -> None:
    bad = sample_customer.copy()
    del bad["tenure"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_extra_field(client: TestClient, sample_customer: dict) -> None:
    bad = sample_customer | {"creditScore": 700}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422  # Customer is extra='forbid'


def test_predict_batch_returns_one_per_input(client: TestClient, sample_customer: dict) -> None:
    payload = {
        "customers": [
            sample_customer,
            sample_customer | {"customerID": "demo-002", "tenure": 48},
        ]
    }
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["customerID"] == "demo-001"
    assert preds[1]["customerID"] == "demo-002"


def test_explain_returns_top_features(client: TestClient, sample_customer: dict) -> None:
    r = client.post("/explain?top_k=5", json=sample_customer)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["customerID"] == sample_customer["customerID"]
    assert len(body["top_features"]) == 5
    # SHAP values should be sorted by absolute value descending
    abs_vals = [f["abs_value"] for f in body["top_features"]]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_metrics_endpoint_returns_prometheus_format(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "churn_requests_total" in r.text
