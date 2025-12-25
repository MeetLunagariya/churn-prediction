"""Build the production model artifact for serving.

Trains the calibrated HGB with the Optuna-tuned best parameters and
saves the entire pipeline (preprocessor + classifier + isotonic
calibrator) plus a small metadata sidecar to ``models/churn_v1.joblib``
/ ``models/churn_v1.json``.

Run before ``make serve``::

    uv run python scripts/build_artifact.py

Idempotent: overwrites the existing artifact.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import typer
from rich.console import Console

from churn.data.load import load_raw
from churn.data.splits import stratified_split
from churn.models.calibrate import calibrate, expected_calibration_error
from churn.models.evaluate import compute_metrics
from churn.models.threshold import optimize_threshold
from churn.models.train import build_pipeline, engineer_split, fit_evaluate

DEFAULT_BEST_PARAMS = {
    "max_iter": 200,
    "learning_rate": 0.03213803416203723,
    "max_leaf_nodes": 88,
    "max_depth": 3,
    "min_samples_leaf": 54,
    "l2_regularization": 0.017127436834329976,
}

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    out_dir: Path = typer.Option(Path("models"), help="Output directory."),  # noqa: B008
    artifact_name: str = typer.Option("churn_v1", help="Artifact base name."),
    seed: int = typer.Option(42, help="Random seed."),
    savings_per_save: float = typer.Option(500.0, help="$ revenue per save."),
    retention_cost: float = typer.Option(50.0, help="$ cost per retention offer."),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Building production artifact[/bold]")
    df = load_raw()
    split = engineer_split(stratified_split(df, seed=seed))

    pipeline = build_pipeline(
        "hgb", seed=seed, use_engineered=True, estimator_params=DEFAULT_BEST_PARAMS
    )
    result = fit_evaluate(pipeline, split)
    console.print(f"  uncalibrated test ROC-AUC {result.test_metrics['roc_auc']:.4f}")

    calibrator = calibrate(result.pipeline, split.X_val, split.y_val, method="isotonic")
    test_proba = calibrator.predict_proba(split.X_test)[:, 1]
    val_proba = calibrator.predict_proba(split.X_val)[:, 1]

    cal_test = compute_metrics(split.y_test, test_proba)
    ece_test = expected_calibration_error(split.y_test, test_proba)
    console.print(f"  calibrated   test ROC-AUC {cal_test['roc_auc']:.4f}  ECE {ece_test:.4f}")

    threshold_search = optimize_threshold(
        split.y_val.to_numpy(),
        val_proba,
        savings_per_save=savings_per_save,
        retention_cost=retention_cost,
    )
    console.print(
        f"  cost-optimal threshold (chosen on val): {threshold_search.best_threshold:.3f}"
    )

    artifact_path = out_dir / f"{artifact_name}.joblib"
    metadata_path = out_dir / f"{artifact_name}.json"

    joblib.dump(calibrator, artifact_path, compress=("zlib", 3))
    metadata: dict[str, Any] = {
        "name": artifact_name,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "model": "HistGradientBoostingClassifier + IsotonicCalibration",
        "feature_set": "engineered",
        "seed": seed,
        "best_params": DEFAULT_BEST_PARAMS,
        "decision_threshold": threshold_search.best_threshold,
        "threshold_sensitivity_range": list(threshold_search.sensitivity_range),
        "cost_matrix": {
            "savings_per_save": savings_per_save,
            "retention_cost": retention_cost,
        },
        "training_metrics": {
            "test_roc_auc": float(cal_test["roc_auc"]),
            "test_pr_auc": float(cal_test["pr_auc"]),
            "test_brier": float(cal_test["brier"]),
            "test_ece": ece_test,
            "n_train": len(split.X_train),
            "n_val": len(split.X_val),
            "n_test": len(split.X_test),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    console.print(
        f"\n[green]Saved[/green] {artifact_path} ({artifact_path.stat().st_size / 1024:.0f} KB)"
    )
    console.print(f"[green]Saved[/green] {metadata_path}")


if __name__ == "__main__":
    app()
