# 0002 — HistGradientBoostingClassifier over LightGBM/XGBoost

- **Date:** 2026-05-13
- **Status:** Accepted

## Context

The Week 2 milestone calls for a tree-ensemble model alongside the LR
baseline. The conventional choice on tabular data is LightGBM or
XGBoost. Both require ``libomp`` on macOS (since their 4.x and 2.x
releases respectively unbundled the OpenMP runtime), which the dev
machine doesn't have and cannot easily install (no Homebrew).

## Decision

Use scikit-learn's ``HistGradientBoostingClassifier`` (HGB) as the
Week-2 tree ensemble. Keep the abstraction in
``churn.models.train._make_estimator`` open so a LightGBM trainer can
be slotted in later by anyone running on a libomp-ready environment.

## Consequences

- The project installs and runs on any OS without a brew step.
- Expected performance gap vs LightGBM is ~1-2 ROC-AUC points on this
  dataset shape (small, well-behaved tabular). On Telco Churn the gap
  is observed to be small.
- HGB doesn't expose `feature_importance_` in the same fashion as
  LightGBM; SHAP TreeExplainer (Week 3) handles it cleanly so this is
  not blocking.
- HGB's Optuna search space differs from LightGBM's (``max_iter`` vs
  ``n_estimators``, ``max_leaf_nodes`` vs ``num_leaves``,
  ``l2_regularization`` vs ``reg_lambda``); the tuner module is
  HGB-specific (``tune_hgb``) to keep search-space concerns close to
  the estimator they're tuned for.

## Alternatives considered

- **Install libomp via Homebrew.** Rejected for this snapshot because
  the dev machine doesn't have brew; could be reconsidered later.
- **CatBoost.** No libomp issue, but ships a large binary (~120 MB) and
  introduces a less-familiar dep. Disqualified on weight-vs-benefit.
- **Sklearn RandomForest.** Worse default performance and slower
  inference than HGB on this shape; not a meaningful alternative.
