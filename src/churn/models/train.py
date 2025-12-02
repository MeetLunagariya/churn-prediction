"""Baseline training entrypoints.

Provides ``build_baseline_pipeline`` (model factory) and ``train_baseline``
(end-to-end fit + eval). MLflow tracking lives in the CLI wrapper at
``scripts/train.py`` so the core training code stays importable without
side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.data.splits import Split
from churn.features.build import build_preprocessor
from churn.models.evaluate import BinaryMetrics, compute_metrics

DEFAULT_SEED = 42


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    val_metrics: BinaryMetrics
    test_metrics: BinaryMetrics


def build_baseline_pipeline(seed: int = DEFAULT_SEED) -> Pipeline:
    """Return an unfitted baseline pipeline: preprocessor -> LogisticRegression.

    ``class_weight`` is intentionally ``None`` — class-weighting is a Week 2
    experiment, not a baseline default (see ADR 0001). ``max_iter=1000`` is
    set high enough that lbfgs converges on this dataset without warnings.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=None,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_baseline(split: Split, seed: int = DEFAULT_SEED) -> TrainingResult:
    """Fit the baseline on train and evaluate on val + test."""
    pipeline = build_baseline_pipeline(seed=seed)
    pipeline.fit(split.X_train, split.y_train)

    val_proba = pipeline.predict_proba(split.X_val)[:, 1]
    test_proba = pipeline.predict_proba(split.X_test)[:, 1]

    return TrainingResult(
        pipeline=pipeline,
        val_metrics=compute_metrics(split.y_val, val_proba),
        test_metrics=compute_metrics(split.y_test, test_proba),
    )
