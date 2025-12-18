"""Training, evaluation, calibration, tuning, threshold, bootstrap, slices."""

from churn.models.bootstrap import BootstrapResult, bootstrap_metric
from churn.models.calibrate import (
    CalibrationMethod,
    calibrate,
    expected_calibration_error,
    reliability_curve,
)
from churn.models.evaluate import BinaryMetrics, compute_metrics, format_metrics
from churn.models.slices import SliceMetric, compute_slice_metrics, slices_to_dataframe
from churn.models.threshold import (
    ThresholdSearch,
    expected_utility,
    optimize_threshold,
)
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
    "BootstrapResult",
    "CalibrationMethod",
    "EstimatorName",
    "SliceMetric",
    "ThresholdSearch",
    "TrainingResult",
    "bootstrap_metric",
    "build_baseline_pipeline",
    "build_pipeline",
    "calibrate",
    "compute_metrics",
    "compute_slice_metrics",
    "engineer_split",
    "expected_calibration_error",
    "expected_utility",
    "fit_evaluate",
    "format_metrics",
    "optimize_threshold",
    "reliability_curve",
    "slices_to_dataframe",
    "train_baseline",
    "train_model",
    "tune_hgb",
]
