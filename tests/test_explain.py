"""Tests for the SHAP explainer wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.data.splits import stratified_split
from churn.explain.shap_explainer import HGBExplainer
from churn.models.train import train_model
from tests.test_models import _bigger_synth_frame


def test_explainer_runs_on_fitted_pipeline() -> None:
    df = _bigger_synth_frame(n=300, seed=2)
    split = stratified_split(df, seed=7)
    result = train_model("hgb", split, seed=42, use_engineered=False)
    explainer = HGBExplainer(result.pipeline)

    rows = split.X_test.head(30)
    explanation = explainer.explain(rows)
    assert explanation.shap_values.shape[0] == len(rows)
    assert explanation.shap_values.shape[1] == len(explanation.feature_names)
    assert isinstance(explanation.base_value, float)


def test_top_features_returns_k_entries() -> None:
    df = _bigger_synth_frame(n=300, seed=3)
    split = stratified_split(df, seed=7)
    result = train_model("hgb", split, seed=42)
    explainer = HGBExplainer(result.pipeline)
    explanation = explainer.explain(split.X_test.head(50))
    top = explainer.top_features(explanation, k=5)
    assert len(top) == 5
    # Ordered descending by absolute SHAP magnitude
    mags = [m for _, m in top]
    assert mags == sorted(mags, reverse=True)


def test_explanation_attribution_sums_recover_logit(synthetic_raw_df: pd.DataFrame) -> None:
    """For each row, base + sum(shap) should equal the model's raw output (within fp tolerance)."""
    df = _bigger_synth_frame(n=400, seed=4)
    split = stratified_split(df, seed=7)
    result = train_model("hgb", split, seed=42)
    explainer = HGBExplainer(result.pipeline)
    rows = split.X_test.head(10)
    explanation = explainer.explain(rows)
    x_proc = explanation.X_processed
    classifier = result.pipeline.named_steps["classifier"]
    raw = classifier.decision_function(x_proc)
    reconstructed = explanation.base_value + explanation.shap_values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, raw, atol=1e-4)
