"""FastAPI inference service."""

from churn.serving.predictor import ChurnPredictor, Explanation, Prediction, predictor

__all__ = ["ChurnPredictor", "Explanation", "Prediction", "predictor"]
