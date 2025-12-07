"""Training entrypoints supporting multiple estimators.

Two estimators are wired up by default:

- ``lr``: Logistic Regression (the baseline).
- ``hgb``: scikit-learn's ``HistGradientBoostingClassifier`` — an
  in-tree GBM with no native (libomp) dependency, ~1-2% AUC behind
  LightGBM on tabular data but installable anywhere.

The split of responsibility:

- :func:`build_pipeline` is a pure factory — given (estimator name, seed,
  hyperparams), returns an unfitted ``sklearn.pipeline.Pipeline``.
- :func:`fit_evaluate` takes a pipeline and a split, fits, and returns
  ``TrainingResult`` (pipeline + val/test metrics). No engineering or
  estimator choice baked in.
- :func:`engineer_split` is the convenience that applies
  :func:`engineer_features` to all three partitions of a split.

MLflow side effects live in the CLI wrapper, not here, so unit tests
don't accidentally write to ``mlruns/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.data.splits import Split
from churn.features.build import build_preprocessor
from churn.features.engineer import engineer_features
from churn.models.evaluate import BinaryMetrics, compute_metrics

DEFAULT_SEED = 42

EstimatorName = Literal["lr", "hgb"]
SUPPORTED_ESTIMATORS: tuple[EstimatorName, ...] = ("lr", "hgb")


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    val_metrics: BinaryMetrics
    test_metrics: BinaryMetrics


def _make_estimator(
    name: EstimatorName,
    *,
    seed: int = DEFAULT_SEED,
    class_weight: Any = None,
    params: dict[str, Any] | None = None,
) -> BaseEstimator:
    """Construct an unfitted classifier of the requested kind."""
    params = dict(params or {})
    if name == "lr":
        params.setdefault("max_iter", 1000)
        params.setdefault("solver", "lbfgs")
        return LogisticRegression(class_weight=class_weight, random_state=seed, **params)
    if name == "hgb":
        params.setdefault("max_iter", 400)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("max_leaf_nodes", 31)
        params.setdefault("early_stopping", False)
        return HistGradientBoostingClassifier(
            class_weight=class_weight, random_state=seed, **params
        )
    raise ValueError(f"unknown estimator: {name!r}; expected one of {SUPPORTED_ESTIMATORS}")


def build_pipeline(
    estimator_name: EstimatorName,
    *,
    seed: int = DEFAULT_SEED,
    use_engineered: bool = False,
    class_weight: Any = None,
    estimator_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Return an unfitted preprocessor + classifier pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(use_engineered=use_engineered)),
            (
                "classifier",
                _make_estimator(
                    estimator_name,
                    seed=seed,
                    class_weight=class_weight,
                    params=estimator_params,
                ),
            ),
        ]
    )


def engineer_split(split: Split) -> Split:
    """Apply :func:`engineer_features` to every partition of ``split``."""
    return Split(
        X_train=engineer_features(split.X_train),
        X_val=engineer_features(split.X_val),
        X_test=engineer_features(split.X_test),
        y_train=split.y_train,
        y_val=split.y_val,
        y_test=split.y_test,
    )


def fit_evaluate(pipeline: Pipeline, split: Split) -> TrainingResult:
    """Fit ``pipeline`` on train, return val + test metrics."""
    pipeline.fit(split.X_train, split.y_train)
    val_proba = pipeline.predict_proba(split.X_val)[:, 1]
    test_proba = pipeline.predict_proba(split.X_test)[:, 1]
    return TrainingResult(
        pipeline=pipeline,
        val_metrics=compute_metrics(split.y_val, val_proba),
        test_metrics=compute_metrics(split.y_test, test_proba),
    )


def build_baseline_pipeline(seed: int = DEFAULT_SEED) -> Pipeline:
    """Original LR baseline (no engineered features). Preserved for back-compat."""
    return build_pipeline("lr", seed=seed)


def train_baseline(split: Split, seed: int = DEFAULT_SEED) -> TrainingResult:
    """Back-compat: original LR baseline."""
    return fit_evaluate(build_baseline_pipeline(seed=seed), split)


def train_model(
    estimator_name: EstimatorName,
    split: Split,
    *,
    seed: int = DEFAULT_SEED,
    use_engineered: bool = False,
    class_weight: Any = None,
    estimator_params: dict[str, Any] | None = None,
) -> TrainingResult:
    """End-to-end: optionally engineer features, build pipeline, fit, evaluate."""
    working_split = engineer_split(split) if use_engineered else split
    pipeline = build_pipeline(
        estimator_name,
        seed=seed,
        use_engineered=use_engineered,
        class_weight=class_weight,
        estimator_params=estimator_params,
    )
    return fit_evaluate(pipeline, working_split)
