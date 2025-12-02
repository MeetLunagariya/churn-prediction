"""Tests for the baseline training pipeline.

These tests use a *real* (small) synthetic dataset rather than mocking out
the classifier — the property being tested is that the same data + same
seed yields the same probabilities, which is the foundation of every
later comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.data.splits import stratified_split
from churn.models.evaluate import compute_metrics
from churn.models.train import build_baseline_pipeline, train_baseline


def _bigger_synth_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """200-row synthetic frame with realistic conditional dependence
    between tenure and churn so LR can learn something meaningful."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, size=n)
    monthly = rng.uniform(20, 120, size=n).round(2)
    # higher churn rate for short tenure, simulating the EDA finding
    p_churn = 1 / (1 + np.exp(-(2.0 - 0.08 * tenure - 0.005 * (monthly - 60))))
    churn = (rng.random(n) < p_churn).astype(int)
    return pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["Yes", "No"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["Yes", "No"], n),
            "OnlineBackup": rng.choice(["Yes", "No"], n),
            "DeviceProtection": rng.choice(["Yes", "No"], n),
            "TechSupport": rng.choice(["Yes", "No"], n),
            "StreamingTV": rng.choice(["Yes", "No"], n),
            "StreamingMovies": rng.choice(["Yes", "No"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": (tenure * monthly).astype(float),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


def test_build_baseline_pipeline_is_unfitted() -> None:
    pipeline = build_baseline_pipeline()
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict_proba")
    # named_steps available pre-fit
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_baseline_is_reproducible_from_seed() -> None:
    df = _bigger_synth_frame(n=200, seed=0)
    split = stratified_split(df, seed=7)

    a = train_baseline(split, seed=42)
    b = train_baseline(split, seed=42)

    proba_a = a.pipeline.predict_proba(split.X_test)[:, 1]
    proba_b = b.pipeline.predict_proba(split.X_test)[:, 1]
    np.testing.assert_allclose(proba_a, proba_b, rtol=1e-12, atol=1e-12)
    assert a.val_metrics == b.val_metrics
    assert a.test_metrics == b.test_metrics


def test_baseline_beats_marginal_on_easy_synthetic_signal() -> None:
    """With tenure baked in as the dominant signal, ROC-AUC should be well above 0.5."""
    df = _bigger_synth_frame(n=400, seed=1)
    split = stratified_split(df, seed=7)
    result = train_baseline(split, seed=42)
    # generous threshold — point is to catch a wholesale pipeline break
    assert result.val_metrics["roc_auc"] > 0.7
    assert result.test_metrics["roc_auc"] > 0.7


def test_compute_metrics_returns_expected_keys() -> None:
    y_true = [0, 0, 1, 1, 0, 1]
    y_proba = [0.1, 0.2, 0.8, 0.7, 0.3, 0.6]
    m = compute_metrics(y_true, y_proba)
    assert set(m.keys()) == {
        "roc_auc",
        "pr_auc",
        "brier",
        "log_loss",
        "f1_at_0_5",
        "positive_rate",
        "n",
    }
    assert m["n"] == 6
    assert 0.0 <= m["roc_auc"] <= 1.0
