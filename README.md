# Customer Churn Prediction with Explainable Insights

> Predict which telco customers will churn next cycle, explain *why*, and recommend retention actions — exposed as a FastAPI inference service and a Streamlit ops dashboard.

[![CI](https://github.com/meetlunagariya/churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/meetlunagariya/churn-prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live demo:** _coming v1.0_ · **API docs:** _coming v1.0_ · **Model card:** [reports/model_card.md](reports/model_card.md)

## Why this project

Retention teams need two things: an accurate signal of who is about to churn, and a *reason* they can act on. This project ships both, with the rigor expected of a production model: calibrated probabilities, a business-cost-aware decision threshold, slice-level evaluation, and drift monitoring.

## Results

_Populated at end of Week 3._

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | TBD | (TBD, TBD) |
| PR-AUC | TBD | (TBD, TBD) |
| Brier score (calibrated) | TBD | — |
| Expected utility @ optimal threshold | TBD | — |
| Worst-slice recall | TBD | — |

Full breakdown: [reports/model_card.md](reports/model_card.md).

## Approach

1. **Data contract** with [pandera](https://pandera.readthedocs.io/); deterministic stratified splits.
2. **Baseline** logistic regression; **production model** LightGBM tuned via Optuna with nested CV.
3. **Calibration** with isotonic regression; ECE reported pre/post.
4. **Decision threshold** chosen to maximize expected utility under an explicit cost matrix.
5. **Explainability** via SHAP — global feature impact and per-customer waterfalls.
6. **Drift monitoring** with [Evidently](https://www.evidentlyai.com/) on a simulated temporal split.
7. **Serving** via FastAPI; **demo** via Streamlit; both containerized.

## Quickstart

```bash
git clone https://github.com/meetlunagariya/churn-prediction.git
cd churn-prediction
make install     # uv sync + pre-commit hooks
make data        # download dataset (~1 MB)
make test        # run unit tests
```

Later milestones add:

```bash
make train       # train + log to MLflow
make serve       # FastAPI on :8000
make app         # Streamlit on :8501
make docker      # docker compose stack
```

## Project structure

```
src/churn/        # importable package
  data/           # loaders, schemas, splits
  features/       # feature engineering
  models/         # training, evaluation, calibration, thresholding
  explain/        # SHAP wrappers
  monitoring/     # drift reports
  serving/        # FastAPI app
app/              # Streamlit dashboard
scripts/          # CLI entrypoints
tests/            # pytest suite
notebooks/        # exploration & diagnostics
reports/          # figures, model card, drift report
docs/             # architecture, ADRs
configs/          # YAML configs (data, train, serve)
```

## Reproducibility

- Single seed (`42`) threaded through splits, CV, and model init.
- All experiments tracked in MLflow (`mlruns/`, gitignored).
- Lockfile (`uv.lock`) committed.
- Dataset SHA-256 checked on download.
- Docker image pinned by digest in `docker-compose.yml`.

## Limitations

- Dataset is a single snapshot; "drift" is simulated, not observed.
- Cost matrix in the model card is illustrative; production deployment requires retention-team-validated costs.
- No causal inference — predictions are correlational; uplift modeling is listed as future work.

## License

[MIT](LICENSE)
