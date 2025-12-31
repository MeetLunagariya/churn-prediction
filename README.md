# Customer Churn Prediction with Explainable Insights

> Predict which telco customers will churn next cycle, explain *why*, and recommend retention actions — exposed as a FastAPI inference service and a Streamlit ops dashboard.

[![CI](https://github.com/meetlunagariya/churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/meetlunagariya/churn-prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live demo:** _deploy your own via_ [`docs/deployment.md`](docs/deployment.md) (Streamlit Community Cloud — 3 min, free). **Local:** `make docker` → :8000 + :8501. **API docs:** `/docs` (OpenAPI). **Model card:** [reports/model_card.md](reports/model_card.md).

## Why this project

Retention teams need two things: an accurate signal of who is about to churn, and a *reason* they can act on. This project ships both, with the rigor expected of a production model: calibrated probabilities, a business-cost-aware decision threshold, slice-level evaluation, and drift monitoring.

## Results

Both models trained on a 70/15/15 stratified split (seed 42), 4,929
training rows. 95% CIs from 1,000-sample bootstrap on the test set.

| Model | ROC-AUC (test) | PR-AUC (test) | Brier | ECE |
|---|---|---|---|---|
| LR baseline (`v0.1.0`) | 0.847 | 0.638 | 0.137 | 0.026 |
| **HGB tuned + calibrated (`v0.2.0`)** | **0.853** [0.831, 0.876] | 0.641 [0.584, 0.696] | **0.134** [0.122, 0.146] | **0.018** |

### Cost-weighted decision (`v0.3.0`)

Under an illustrative cost matrix (`S=$500` revenue per save, `R=$50`
retention offer — see [ADR 0003](docs/decisions/0003-cost-matrix.md)):

| Threshold | Predicted positives | Expected utility | vs. naive |
|---|---|---|---|
| **0.120 (optimal)** | larger cohort | **$100,500** | **+49.9%** |
| 0.500 (naive) | smaller cohort | $67,050 | — |

Sensitivity: thresholds in [0.03, 0.15] are all within 5% of the
optimum, so the operating point is not fragile.

### Slice analysis

Weakest slice on test: **`Contract = One year`** (n=224, ROC-AUC 0.718
— 14 points below the 0.853 global). Strongest: `Contract = Two year`
(0.882) and `InternetService = No` (0.909). Full table in
[reports/model_card.md](reports/model_card.md).

### Explainability

SHAP global ranking (top 5 by mean |SHAP|): `Contract_Month-to-month`,
`charge_ratio` (engineered), `OnlineSecurity_No`, `TechSupport_No`,
`Contract_Two year`. Per-customer waterfalls in
[reports/figures/](reports/figures/).

- HGB picks up +0.6 ROC-AUC and a 30% reduction in ECE versus the
  baseline on test; the bulk of the lift comes from engineered features
  + Optuna tuning. Calibration buys lower Brier and ECE at a tiny
  PR-AUC cost. Best params:
  `learning_rate=0.032, max_iter=200, max_leaf_nodes=88, max_depth=3,
  min_samples_leaf=54, l2_regularization=0.017`.
- scikit-learn's `HistGradientBoostingClassifier` is used in place of
  LightGBM/XGBoost — see [ADR 0002](docs/decisions/0002-hgb-over-lightgbm.md).

Reproduce:

```bash
make data
make train                                                       # LR baseline
uv run python scripts/train.py --config configs/train_hgb.yaml   # tuned HGB
uv run python notebooks/03_model_diagnostics.py                  # threshold, slices, SHAP, drift
```

Full breakdown: [reports/model_card.md](reports/model_card.md).

## Serving (`v1.0.0`)

### Architecture

```
┌──────────────────┐   /predict      ┌────────────────────┐
│ Streamlit (8501) │ ──────────────▶ │  FastAPI (8000)    │
│  customer lookup │ ◀────────────── │  + Prometheus      │
│  what-if         │   /explain      │  + structlog       │
│  about           │                  └────────┬───────────┘
└──────────────────┘                           │
                                               ▼
                                       ┌───────────────┐
                                       │ ChurnPredictor│
                                       │ joblib model  │
                                       │ + SHAP        │
                                       └───────────────┘
```

### Run locally

```bash
make build                # train + save calibrated HGB to models/churn_v1.joblib
make serve                # FastAPI on :8000 (rebuilds artifact first)
make app                  # Streamlit on :8501 (separate terminal)
# or, in one command:
make docker               # docker compose up --build
```

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | Liveness + model-loaded check |
| GET  | `/metrics` | Prometheus counters and latency histograms |
| GET  | `/model/info` | Version, decision threshold, training metrics |
| POST | `/predict` | Single customer → probability + class + recommended action |
| POST | `/predict/batch` | Up to 1000 customers in one call |
| POST | `/explain?top_k=5` | Per-customer SHAP top-N contributions |

Pydantic v2 schemas reject malformed inputs with 422 at the request boundary.
Structured JSON logs via structlog include request IDs and latency.

### Example

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_customer.json | jq
```

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
make test        # run unit tests (15 tests, 93% coverage)
make train       # train baseline LR + log to MLflow
```

Inspect runs:

```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

Later milestones add:

```bash
make serve       # FastAPI on :8000   (Week 4)
make app         # Streamlit on :8501 (Week 4)
make docker      # docker compose stack
```

## Notebooks

The EDA is authored as a jupytext-paired script: edit
[`notebooks/01_eda.py`](notebooks/01_eda.py), regenerate the .ipynb with
`uv run jupytext --to ipynb --execute notebooks/01_eda.py`. The committed
.ipynb is what renders on GitHub.

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
