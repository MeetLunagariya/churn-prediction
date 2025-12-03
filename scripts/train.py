"""Train the baseline churn model and log to MLflow.

Usage::

    uv run python scripts/train.py --config configs/train.yaml

The CLI is intentionally thin — the heavy lifting lives in
``churn.models.train``. Keeping MLflow side effects out of the importable
package means tests don't accidentally write to ``mlruns/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import typer
import yaml
from mlflow.models.signature import infer_signature
from rich.console import Console

from churn.data.load import load_raw
from churn.data.splits import stratified_split
from churn.models.evaluate import format_metrics
from churn.models.train import train_baseline

app = typer.Typer(add_completion=False)
console = Console()


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
    console.print(
        f"split sizes: train={len(split.X_train)}, val={len(split.X_val)}, "
        f"test={len(split.X_test)}"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        mlflow.log_params(
            {
                "model": cfg["model"]["name"],
                "max_iter": cfg["model"]["max_iter"],
                "class_weight": str(cfg["model"]["class_weight"]),
                "seed": cfg["seed"],
                "train_size": len(split.X_train),
                "val_size": len(split.X_val),
                "test_size": len(split.X_test),
            }
        )

        result = train_baseline(split, seed=cfg["seed"])

        for metric_name, value in result.val_metrics.items():
            if isinstance(value, int | float):
                mlflow.log_metric(f"val_{metric_name}", float(value))
        for metric_name, value in result.test_metrics.items():
            if isinstance(value, int | float):
                mlflow.log_metric(f"test_{metric_name}", float(value))

        sample = split.X_val.head(5)
        signature = infer_signature(sample, result.pipeline.predict_proba(sample))
        mlflow.sklearn.log_model(
            sk_model=result.pipeline,
            name="model",
            signature=signature,
            input_example=sample,
        )

        console.print(format_metrics(result.val_metrics, prefix="[val]"))
        console.print(format_metrics(result.test_metrics, prefix="[test]"))
        console.print(f"\n[dim]MLflow run id: {run.info.run_id}[/dim]")
        console.print(f"[dim]Tracking URI:  {tracking_uri}[/dim]")


if __name__ == "__main__":
    app()
