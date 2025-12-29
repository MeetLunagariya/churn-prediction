# Model Card — Churn Prediction

Status: **v1.0.0** — full lifecycle landed: LR baseline, HGB calibrated +
threshold-optimized, slice analysis, SHAP explainability, drift harness,
FastAPI inference service, Streamlit ops dashboard, Dockerized.

## Intended use

Predict the probability that a Telco customer churns in the next billing
cycle, and surface ranked retention candidates to the retention team. Not
intended for fully automated retention spend; a human is in the loop for
any action.

**Out of scope:** real-time individual customer decisions without
calibration check, any consumer-facing decisioning, lending or insurance
decisioning (the model is not trained for those distributions).

## Training data

- IBM Telco Customer Churn (public dataset, 7,043 rows, 21 columns).
- SHA-256 of the raw CSV is pinned in [scripts/download_data.py](../scripts/download_data.py).
- 70/15/15 stratified split with seed 42.
- Class balance: 26.5% positive (churn).

Caveats:
- Single snapshot — no temporal validation; the drift simulation in
  Week 3 is artificial.
- Single geography / market segment — generalization to other telco
  populations is unverified.

## Features

- 2 numeric (`tenure`, `MonthlyCharges`).
- 16 categorical, one-hot encoded with `handle_unknown='ignore'`.
- 2 engineered numeric (`charge_ratio`, `total_services`) and 1
  engineered categorical (`tenure_bucket`) for the HGB model.
- `customerID` (identifier) and `TotalCharges` (collinear with
  `tenure * MonthlyCharges`) are dropped.

## Models

| Model | Engineered | Tuned | Calibrated | Seed |
|---|---|---|---|---|
| LR baseline | no | no | no | 42 |
| HGB | yes | Optuna 30 trials × 5-fold CV (PR-AUC) | isotonic on val | 42 |

## Metrics

_Filled in by the Week 2 commit that closes v0.2.0._

| Model | Val ROC-AUC | Val PR-AUC | Test ROC-AUC | Test PR-AUC | Test Brier | Test ECE |
|---|---|---|---|---|---|---|
| LR baseline | 0.828 | 0.629 | 0.847 | 0.638 | 0.137 | 0.026 |
| HGB tuned (uncalibrated) | 0.836 | 0.644 | 0.854 | 0.670 | 0.137 | 0.029 |
| **HGB tuned + isotonic-calibrated** | **0.843** | **0.643** | **0.853** | **0.641** | **0.134** | **0.018** |

95% bootstrap CIs on the calibrated HGB test set (n=1,057, 1,000
resamples, seed=42):

| Metric | Point | 95% CI |
|---|---|---|
| ROC-AUC | 0.8530 | [0.8305, 0.8758] |
| PR-AUC  | 0.6406 | [0.5841, 0.6963] |
| Brier   | 0.1340 | [0.1219, 0.1462] |

## Cost-weighted decision

Under the illustrative cost matrix from
[ADR 0003](../docs/decisions/0003-cost-matrix.md) (`S=$500, R=$50`):

| Threshold | Expected utility | Vs naive 0.5 |
|---|---|---|
| **0.120 (optimum)** | **$100,500** | **+49.9%** |
| 0.5 (naive) | $67,050 | — |

Sensitivity: thresholds in [0.03, 0.15] are all within 5% of the
optimum.

## Slice analysis (calibrated HGB, test set)

ROC-AUC by subgroup, n ≥ 30:

| Slice | Value | n | Positive rate | ROC-AUC |
|---|---|---|---|---|
| Contract | Month-to-month | 583 | 0.424 | 0.767 |
| Contract | One year | 224 | 0.125 | **0.718** ← weakest |
| Contract | Two year | 250 | 0.020 | 0.882 |
| tenure_bucket | 0-6 | 225 | 0.573 | 0.766 |
| tenure_bucket | 7-12 | 102 | 0.284 | 0.739 |
| tenure_bucket | 13-24 | 154 | 0.292 | 0.807 |
| tenure_bucket | 25-48 | 239 | 0.213 | 0.790 |
| tenure_bucket | 49+ | 337 | 0.077 | 0.883 |
| InternetService | DSL | 358 | 0.184 | 0.830 |
| InternetService | Fiber optic | 480 | 0.413 | 0.787 |
| InternetService | No | 219 | 0.073 | 0.909 |
| PaymentMethod | Bank transfer | 237 | 0.190 | 0.902 |
| PaymentMethod | Credit card | 222 | 0.176 | 0.859 |
| PaymentMethod | Electronic check | 363 | 0.419 | 0.790 |
| PaymentMethod | Mailed check | 235 | 0.187 | 0.828 |

The weakest slice (One year contracts) has too few churners (n=28) for
the ranking task to be easy. Worth flagging to the retention team but
not necessarily a model defect.

## Explainability (SHAP)

Top 10 features by mean |SHAP value| on the test set:

| Rank | Feature | Mean abs SHAP |
|---|---|---|
| 1 | Contract_Month-to-month | 0.681 |
| 2 | charge_ratio *(engineered)* | 0.518 |
| 3 | OnlineSecurity_No | 0.195 |
| 4 | TechSupport_No | 0.192 |
| 5 | Contract_Two year | 0.161 |
| 6 | MonthlyCharges | 0.150 |
| 7 | InternetService_Fiber optic | 0.150 |
| 8 | PaymentMethod_Electronic check | 0.141 |
| 9 | PaperlessBilling_No | 0.096 |
| 10 | tenure | 0.076 |

Notable: the engineered `charge_ratio` is the #2 feature globally,
ahead of either of its raw inputs. The retention team's intuition that
"contract type matters most" is confirmed quantitatively.

Per-customer waterfalls for the highest- and lowest-risk test rows
are saved as `reports/figures/12_shap_high_risk.png` and
`reports/figures/13_shap_low_risk.png`.

## Drift monitoring

Simulated by splitting the test set 70/30 by index (placeholder for
real timestamped batches). All six monitored signals (tenure,
MonthlyCharges, charge_ratio, Contract, InternetService, and the
prediction score itself) classify as `stable` — expected, since both
windows come from the same distribution. The harness lives in
`src/churn/monitoring/drift.py` and is wired up to be replaced by real
timestamped batches in production.

Optuna best params (30 trials × 5-fold stratified CV, PR-AUC objective):
- `learning_rate`: 0.032
- `max_iter`: 200
- `max_leaf_nodes`: 88
- `max_depth`: 3
- `min_samples_leaf`: 54
- `l2_regularization`: 0.017
- Best CV PR-AUC: 0.674

Observations:
- Tree depth landed shallow (max_depth=3, min_samples_leaf=54) — the
  search prefers regularization over flexibility on this dataset, which
  is consistent with the linear-friendly nature of the dominant signal
  (tenure).
- Calibration trades 0.029 PR-AUC for 0.011 ECE (lower is better). The
  trade is worthwhile because the Week-3 threshold optimization is
  probability-sensitive — a model that ranks well but is poorly
  calibrated picks worse thresholds.

95% bootstrap confidence intervals land in Week 3.

## Calibration

Isotonic calibration fit on the val partition only; never on train (would
overfit) and never on test (would leak). ECE is reported pre/post.

## Known limitations

1. **Snapshot data, no real drift.** Drift monitoring (Week 3) is a
   demonstration of the harness, not evidence of live drift handling.
2. **No causal inference.** The model predicts who *will* churn, not who
   *would respond to retention*. Uplift modeling is future work.
3. **Cost matrix is illustrative.** The Week-3 cost-weighted threshold
   uses values that should be re-validated with the retention team
   before any real deployment.
4. **No fairness audit yet.** Slice analysis in Week 3 will surface any
   subgroup gaps (e.g., by tenure bucket, contract type); demographic
   fairness slicing is out of scope on this dataset (no sensitive
   attributes labeled).

## Reproducibility

```bash
make data        # SHA-pinned dataset
make train       # LR baseline -> mlruns/
uv run python scripts/train.py --config configs/train_hgb.yaml
```

All seeds threaded from configs. Lockfile (`uv.lock`) committed. Python
3.11 pinned via `.python-version`.

## Maintainers

Project maintainer: [meetlunagariya47@gmail.com](mailto:meetlunagariya47@gmail.com).
