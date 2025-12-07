"""Training, evaluation, calibration, tuning."""

from churn.models.calibrate import (
    CalibrationMethod,
    calibrate,
    expected_calibration_error,
    reliability_curve,
)
from churn.models.evaluate import BinaryMetrics, compute_metrics, format_metrics
from churn.models.train import (
    DEFAULT_SEED,
    SUPPORTED_ESTIMATORS,
    EstimatorName,
    TrainingResult,
    build_baseline_pipeline,
    build_pipeline,
    engineer_split,
    fit_evaluate,
    train_baseline,
    train_model,
)
from churn.models.tune import tune_hgb

__all__ = [
    "DEFAULT_SEED",
    "SUPPORTED_ESTIMATORS",
    "BinaryMetrics",
    "CalibrationMethod",
    "EstimatorName",
    "TrainingResult",
    "build_baseline_pipeline",
    "build_pipeline",
    "calibrate",
    "compute_metrics",
    "engineer_split",
    "expected_calibration_error",
    "fit_evaluate",
    "format_metrics",
    "reliability_curve",
    "train_baseline",
    "train_model",
    "tune_hgb",
]
