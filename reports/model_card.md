# Model Card — Churn Prediction

Status: **v0.2-modeling** (LR baseline + HGB Week 2 model). Calibration
done, threshold optimization and slice analysis arrive in Week 3.

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
