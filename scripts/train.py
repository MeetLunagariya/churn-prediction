"""Train a churn model end-to-end and log to MLflow.

Usage::

    uv run python scripts/train.py --config configs/train.yaml      # LR baseline
    uv run python scripts/train.py --config configs/train_hgb.yaml  # HGB tuned + calibrated

Reads everything from the config: model name, engineered-feature flag,
optional Optuna tuning, optional calibration. The CLI is intentionally
thin — model code lives in ``churn.models``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import mlflow
import typer
import yaml
from mlflow.models.signature import infer_signature
from rich.console import Console

from churn.data.load import load_raw
from churn.data.splits import stratified_split
from churn.models.calibrate import calibrate, expected_calibration_error
from churn.models.evaluate import compute_metrics, format_metrics
from churn.models.train import (
    EstimatorName,
    build_pipeline,
    engineer_split,
    fit_evaluate,
)
from churn.models.tune import tune_hgb

app = typer.Typer(add_completion=False)
console = Console()


def _log_metrics(metrics: dict[str, Any], prefix: str) -> None:
    for k, v in metrics.items():
        if isinstance(v, int | float):
            mlflow.log_metric(f"{prefix}_{k}", float(v))


@app.command()
def main(
    config: Path = typer.Option(  # noqa: B008
        Path("configs/train.yaml"), help="Path to training config YAML."
    ),
    data_dir: Path = typer.Option(Path("data/raw"), help="Raw data dir."),  # noqa: B008
    tracking_uri: str = typer.Option("file:./mlruns", help="MLflow tracking URI."),
) -> None:
    cfg: dict[str, Any] = yaml.safe_load(config.read_text())

    console.print(f"[bold]Loading data from {data_dir}[/bold]")
    df = load_raw(data_dir)

    split_cfg = cfg["split"]
    split = stratified_split(
        df,
        val_size=split_cfg["val_size"],
        test_size=split_cfg["test_size"],
        seed=split_cfg["seed"],
    )
    use_engineered = bool(cfg.get("features", {}).get("use_engineered", False))
    if use_engineered:
        split = engineer_split(split)
    console.print(
        f"split sizes: train={len(split.X_train)}, val={len(split.X_val)}, "
        f"test={len(split.X_test)}, engineered={use_engineered}"
    )

    model_cfg = cfg["model"]
    estimator_name = cast(EstimatorName, model_cfg["name"])
    class_weight = model_cfg.get("class_weight")
    estimator_params: dict[str, Any] = dict(model_cfg.get("params") or {})

    tuning_cfg = cfg.get("tuning") or {}
    cal_cfg = cfg.get("calibration") or {}

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        mlflow.log_params(
            {
                "model": estimator_name,
                "class_weight": str(class_weight),
                "seed": cfg["seed"],
                "use_engineered": use_engineered,
                "train_size": len(split.X_train),
                "val_size": len(split.X_val),
                "test_size": len(split.X_test),
                "tuning_enabled": bool(tuning_cfg.get("enabled", False)),
                "calibration_enabled": bool(cal_cfg.get("enabled", False)),
            }
        )

        if tuning_cfg.get("enabled") and estimator_name == "hgb":
            n_trials = int(tuning_cfg.get("n_trials", 30))
            n_splits = int(tuning_cfg.get("n_splits", 5))
            console.print(
                f"[bold]Running Optuna search ({n_trials} trials, {n_splits}-fold CV)…[/bold]"
            )
            best_params, study = tune_hgb(
                split.X_train,
                split.y_train,
                use_engineered=use_engineered,
                n_trials=n_trials,
                n_splits=n_splits,
                seed=cfg["seed"],
            )
            console.print(f"best CV PR-AUC: {study.best_value:.4f}")
            console.print(f"best params: {best_params}")
            estimator_params.update(best_params)
            mlflow.log_metric("tuning_best_cv_pr_auc", study.best_value)
            mlflow.log_dict(best_params, "best_params.json")

        for k, v in estimator_params.items():
            mlflow.log_param(f"estimator_{k}", v)

        pipeline = build_pipeline(
            estimator_name,
            seed=cfg["seed"],
            use_engineered=use_engineered,
            class_weight=class_weight,
            estimator_params=estimator_params,
        )
        result = fit_evaluate(pipeline, split)
        console.print(format_metrics(result.val_metrics, prefix="[val ]"))
        console.print(format_metrics(result.test_metrics, prefix="[test]"))
        _log_metrics(dict(result.val_metrics), "val")
        _log_metrics(dict(result.test_metrics), "test")

        val_proba_raw = result.pipeline.predict_proba(split.X_val)[:, 1]
        test_proba_raw = result.pipeline.predict_proba(split.X_test)[:, 1]
        ece_val_raw = expected_calibration_error(split.y_val, val_proba_raw)
        ece_test_raw = expected_calibration_error(split.y_test, test_proba_raw)
        mlflow.log_metric("val_ece", ece_val_raw)
        mlflow.log_metric("test_ece", ece_test_raw)
        console.print(f"  uncalibrated ECE: val={ece_val_raw:.4f} test={ece_test_raw:.4f}")

        final_pipeline = result.pipeline

        if cal_cfg.get("enabled"):
            method = cal_cfg.get("method", "isotonic")
            console.print(f"[bold]Calibrating with method={method}[/bold]")
            calibrator = calibrate(result.pipeline, split.X_val, split.y_val, method=method)
            cal_val_proba = calibrator.predict_proba(split.X_val)[:, 1]
            cal_test_proba = calibrator.predict_proba(split.X_test)[:, 1]
            val_metrics_cal = compute_metrics(split.y_val, cal_val_proba)
            test_metrics_cal = compute_metrics(split.y_test, cal_test_proba)
            ece_val_cal = expected_calibration_error(split.y_val, cal_val_proba)
            ece_test_cal = expected_calibration_error(split.y_test, cal_test_proba)
            _log_metrics(dict(val_metrics_cal), "val_calibrated")
            _log_metrics(dict(test_metrics_cal), "test_calibrated")
            mlflow.log_metric("val_calibrated_ece", ece_val_cal)
            mlflow.log_metric("test_calibrated_ece", ece_test_cal)
            console.print(format_metrics(val_metrics_cal, prefix="[val cal ]"))
            console.print(format_metrics(test_metrics_cal, prefix="[test cal]"))
            console.print(f"  calibrated   ECE: val={ece_val_cal:.4f} test={ece_test_cal:.4f}")
            final_pipeline = calibrator

        sample = split.X_val.head(5)
        signature = infer_signature(sample, final_pipeline.predict_proba(sample))
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            name="model",
            signature=signature,
            input_example=sample,
        )

        console.print(f"\n[dim]MLflow run id: {run.info.run_id}[/dim]")
        console.print(f"[dim]Tracking URI:  {tracking_uri}[/dim]")


if __name__ == "__main__":
    app()
