"""Model loading and prediction for the FastAPI service.

A single ``ChurnPredictor`` instance owns the joblib-serialized
calibrated pipeline plus the cost-aware decision threshold. The
instance is held in a module-level singleton that's instantiated on
first access — so unit-tests can swap in a stub without forcing model
load on import.

SHAP explanations are computed lazily: the TreeExplainer is built on
demand once the pipeline is loaded, then reused.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin

from churn.explain.shap_explainer import HGBExplainer
from churn.features.engineer import engineer_features

DEFAULT_ARTIFACT = Path("models/churn_v1.joblib")
DEFAULT_METADATA = Path("models/churn_v1.json")


@dataclass
class Prediction:
    customer_id: str
    probability: float
    class_label: int
    threshold_used: float
    recommended_action: str
    model_version: str


@dataclass
class Explanation:
    customer_id: str
    probability: float
    class_label: int
    base_value: float
    top_features: list[tuple[str, float, float]]  # (feature, shap, abs(shap))
    model_version: str


class ChurnPredictor:
    """Wraps a fitted+calibrated pipeline behind a JSON-friendly interface."""

    def __init__(
        self,
        artifact_path: Path | str = DEFAULT_ARTIFACT,
        metadata_path: Path | str = DEFAULT_METADATA,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.metadata_path = Path(metadata_path)
        self._model: ClassifierMixin | None = None
        self._metadata: dict[str, Any] | None = None
        self._explainer: HGBExplainer | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not self.artifact_path.exists():
                raise FileNotFoundError(
                    f"Model artifact not found at {self.artifact_path}. "
                    "Run `make build` to produce it."
                )
            self._model = joblib.load(self.artifact_path)
            self._metadata = json.loads(self.metadata_path.read_text())

    def info(self) -> dict[str, Any]:
        if not self._metadata:
            self.load()
        assert self._metadata is not None
        return self._metadata

    @property
    def threshold(self) -> float:
        return float(self.info()["decision_threshold"])

    @property
    def version(self) -> str:
        return str(self.info()["version"])

    def _frame(self, customers: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(customers)
        # Engineer features in-place to mirror the training pipeline.
        return engineer_features(df)

    def predict(self, customers: list[dict[str, Any]]) -> list[Prediction]:
        if not self.is_loaded:
            self.load()
        assert self._model is not None
        df = self._frame(customers)
        proba = self._model.predict_proba(df)[:, 1]
        threshold = self.threshold
        version = self.version
        out: list[Prediction] = []
        for i, c in enumerate(customers):
            p = float(proba[i])
            cls = int(p >= threshold)
            out.append(
                Prediction(
                    customer_id=str(c.get("customerID", f"row-{i}")),
                    probability=p,
                    class_label=cls,
                    threshold_used=threshold,
                    recommended_action="retain" if cls == 1 else "monitor",
                    model_version=version,
                )
            )
        return out

    def explain(self, customer: dict[str, Any], top_k: int = 5) -> Explanation:
        if not self.is_loaded:
            self.load()
        assert self._model is not None
        # The serialized object is a CalibratedClassifierCV. The underlying
        # uncalibrated pipeline lives at .estimator (FrozenEstimator wrapper).
        if self._explainer is None:
            underlying = self._unwrap_pipeline()
            self._explainer = HGBExplainer(underlying)

        df = self._frame([customer])
        explanation = self._explainer.explain(df)
        # For a single row we need each feature's signed SHAP value, not the
        # global mean abs that top_features returns.
        single_row = explanation.shap_values[0]
        ranked = sorted(
            zip(explanation.feature_names, single_row, strict=False),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:top_k]

        proba = float(self._model.predict_proba(df)[0, 1])
        cls = int(proba >= self.threshold)
        return Explanation(
            customer_id=str(customer.get("customerID", "row-0")),
            probability=proba,
            class_label=cls,
            base_value=explanation.base_value,
            top_features=[(name, float(v), abs(float(v))) for name, v in ranked],
            model_version=self.version,
        )

    def _unwrap_pipeline(self) -> Any:
        """Return the fitted (preprocessor + classifier) pipeline.

        ``CalibratedClassifierCV(FrozenEstimator(pipeline))`` stores the
        wrapped pipeline at ``.estimator.estimator``.
        """
        assert self._model is not None
        inner = getattr(self._model, "estimator", None)
        # FrozenEstimator unwraps via .estimator as well
        if inner is not None and hasattr(inner, "estimator"):
            return inner.estimator
        return inner


# Module-level singleton — instantiated but not loaded.
predictor = ChurnPredictor()
