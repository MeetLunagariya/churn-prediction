"""SHAP-based explainability for the HGB churn model.

The HGB classifier is tree-based, so we use SHAP's TreeExplainer for
exact, fast Shapley values. The challenge: SHAP operates on the
preprocessed feature matrix (post-OHE and scaling), not the raw
dataframe. This module handles the plumbing so callers pass raw rows
and get back attributions in human-readable feature space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Explanation:
    feature_names: list[str]
    shap_values: NDArray[np.float64]  # shape (n_rows, n_features)
    base_value: float
    X_processed: NDArray[np.float64]


class HGBExplainer:
    """Wraps a fitted churn pipeline (preprocessor + classifier) with SHAP.

    Usage::

        explainer = HGBExplainer(pipeline)
        global_exp = explainer.explain(split.X_test)
        explainer.save_global_summary(global_exp, "reports/figures/shap_summary.png")
        explainer.save_waterfall(global_exp, row=0, path="reports/figures/shap_row0.png")
    """

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self._preprocessor = pipeline.named_steps["preprocessor"]
        self._classifier = pipeline.named_steps["classifier"]
        self._explainer = shap.TreeExplainer(self._classifier)

    def explain(self, X: pd.DataFrame) -> Explanation:
        x_processed = self._preprocessor.transform(X)
        feature_names = list(self._preprocessor.get_feature_names_out())
        exp = self._explainer(x_processed)
        # shap may return Explanation with a different shape based on output
        shap_values = np.asarray(exp.values, dtype=np.float64)
        base = exp.base_values
        # For binary HGB, base_values can be scalar or 1-d
        base_value = float(np.asarray(base).flat[0])
        return Explanation(
            feature_names=feature_names,
            shap_values=shap_values,
            base_value=base_value,
            X_processed=np.asarray(x_processed, dtype=np.float64),
        )

    def save_global_summary(
        self, explanation: Explanation, path: str, max_display: int = 20
    ) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            explanation.shap_values,
            explanation.X_processed,
            feature_names=explanation.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()

    def save_waterfall(
        self, explanation: Explanation, row: int, path: str, max_display: int = 12
    ) -> None:
        import matplotlib.pyplot as plt

        single = shap.Explanation(
            values=explanation.shap_values[row],
            base_values=explanation.base_value,
            data=explanation.X_processed[row],
            feature_names=explanation.feature_names,
        )
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(single, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()

    def top_features(self, explanation: Explanation, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k features by mean absolute SHAP value."""
        mean_abs = np.abs(explanation.shap_values).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:k]
        return [(explanation.feature_names[i], float(mean_abs[i])) for i in idx]
