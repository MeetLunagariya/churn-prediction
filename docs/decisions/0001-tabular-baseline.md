# 0001 — Baseline modeling approach for churn

- **Date:** 2026-05-13
- **Status:** Accepted

## Context

The project predicts customer churn from the IBM Telco dataset (~7k rows,
mixed numeric and categorical features, ~26% positive class). The baseline
needs to satisfy three constraints:

1. Establish a reproducible reference metric to gate future modeling work.
2. Be simple enough to debug end-to-end on Day 5.
3. Produce probabilities, not just labels, so calibration can be measured
   later.

## Decision

The baseline is **logistic regression** inside a scikit-learn `Pipeline`:

- `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")` for
  categoricals and `StandardScaler` for numerics.
- `LogisticRegression(max_iter=1000, class_weight=None)` so the baseline
  reflects the *uncorrected* class balance — class-weighting is a Week 2
  experiment, not a baseline default.
- Trained on the train split, evaluated on val, final report on test.

## Consequences

- Establishes a metric floor (expected ROC-AUC ~0.83 on this dataset).
- Makes the comparison against LightGBM/XGBoost in Week 2 directly
  attributable to the model, not the preprocessing.
- The `Pipeline` shape is reused by every downstream model, so feature
  changes propagate automatically.

## Alternatives considered

- **Decision tree** — interpretable, but doesn't produce well-ranked
  probabilities and skews comparisons.
- **Naive frequency model** — too weak a floor; doesn't exercise the
  preprocessing pipeline.
- **LightGBM straight away** — skips the "is the data pipeline correct?"
  check that a linear baseline forces.
