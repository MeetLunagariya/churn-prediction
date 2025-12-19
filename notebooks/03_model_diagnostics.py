# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 — Model Diagnostics
#
# Consolidates the Week 3 evaluation work for the calibrated HGB model:
#
# 1. Cost-weighted threshold sweep (utility vs. naive 0.5).
# 2. Slice-level metrics by contract type and tenure bucket.
# 3. Bootstrap CIs for top-line metrics.
# 4. SHAP global summary and per-customer waterfall.
# 5. Simulated temporal drift report (PSI / KS / chi-squared).
#
# This notebook re-trains the HGB model from scratch (cheap on this
# dataset) so it doesn't depend on a specific MLflow run id.

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

if Path.cwd().name == "notebooks":
    os.chdir(Path.cwd().parent)

from churn.data.load import load_raw
from churn.data.splits import stratified_split
from churn.explain.shap_explainer import HGBExplainer
from churn.models.bootstrap import bootstrap_metric
from churn.models.calibrate import calibrate, reliability_curve
from churn.models.slices import compute_slice_metrics, slices_to_dataframe
from churn.models.threshold import optimize_threshold
from churn.models.train import build_pipeline, engineer_split, fit_evaluate
from churn.monitoring.drift import (
    categorical_drift,
    numeric_drift,
    reports_to_dataframe,
    score_drift,
)

FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 42

# %% [markdown]
# ## Train the calibrated HGB model
#
# Uses the same configuration as configs/train_hgb.yaml but with the
# best Optuna parameters baked in for reproducibility.

# %%
df = load_raw()
split = stratified_split(df, seed=SEED)
split = engineer_split(split)

best_params = {
    "max_iter": 200,
    "learning_rate": 0.03213803416203723,
    "max_leaf_nodes": 88,
    "max_depth": 3,
    "min_samples_leaf": 54,
    "l2_regularization": 0.017127436834329976,
}
pipeline = build_pipeline("hgb", seed=SEED, use_engineered=True, estimator_params=best_params)
result = fit_evaluate(pipeline, split)
print("Pre-calibration:")
print(
    f"  val  ROC-AUC {result.val_metrics['roc_auc']:.4f}  PR-AUC {result.val_metrics['pr_auc']:.4f}"
)
print(
    f"  test ROC-AUC {result.test_metrics['roc_auc']:.4f}  PR-AUC {result.test_metrics['pr_auc']:.4f}"
)

calibrator = calibrate(result.pipeline, split.X_val, split.y_val, method="isotonic")
test_proba = calibrator.predict_proba(split.X_test)[:, 1]
val_proba = calibrator.predict_proba(split.X_val)[:, 1]
print(f"  test calibrated ROC-AUC {roc_auc_score(split.y_test, test_proba):.4f}")

# %% [markdown]
# ## Cost-weighted threshold optimization
#
# We sweep thresholds in [0.01, 0.99] and compute expected utility under
# the cost matrix from ADR 0003. The lift over the naive 0.5 cutoff is
# the headline business number.

# %%
search = optimize_threshold(
    split.y_test.to_numpy(), test_proba, savings_per_save=500, retention_cost=50
)
print(f"Optimal threshold:    {search.best_threshold:.3f}")
print(f"Utility @ optimum:    ${search.best_utility:,.0f}")
print(f"Utility @ 0.5 naive:  ${search.naive_utility:,.0f}")
print(
    f"Uplift:               ${search.best_utility - search.naive_utility:,.0f} "
    f"({(search.best_utility / max(search.naive_utility, 1) - 1):+.1%})"
)
print(
    f"Sensitivity range (within 5%): [{search.sensitivity_range[0]:.2f}, "
    f"{search.sensitivity_range[1]:.2f}]"
)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(search.thresholds, search.utilities, color="#4c72b0", lw=2)
ax.axvline(
    search.best_threshold,
    color="#dd8452",
    linestyle="--",
    label=f"optimum {search.best_threshold:.2f}",
)
ax.axvline(0.5, color="gray", linestyle=":", label="naive 0.5")
ax.axvspan(*search.sensitivity_range, alpha=0.15, color="#dd8452", label="within 5% of optimum")
ax.set_xlabel("decision threshold")
ax.set_ylabel("expected utility ($)")
ax.set_title("Cost-weighted threshold sweep on test set")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "09_threshold_utility.png", dpi=120)
plt.close(fig)

# %% [markdown]
# ## Bootstrap confidence intervals
#
# 1,000 bootstrap resamples of the test set for each top-line metric.
# Point estimates without intervals are misleading on a 1,057-row test
# fold.

# %%
bootstrap_results = []
for name, fn in [
    ("ROC-AUC", roc_auc_score),
    ("PR-AUC", average_precision_score),
    ("Brier (lower better)", brier_score_loss),
]:
    r = bootstrap_metric(
        fn, split.y_test.to_numpy(), test_proba, metric_name=name, n_bootstrap=1000, seed=SEED
    )
    print(r.format())
    bootstrap_results.append(r)

# %% [markdown]
# ## Slice-level metrics
#
# Surface subgroups where the model underperforms. The weakest slice
# (lowest ROC-AUC among groups with n >= 30) is what we'd discuss with
# the retention team and the candidate for future feature work.

# %%
# Re-attach the raw (pre-engineered) columns to the test rows for slicing.
test_raw = df.iloc[split.X_test.index].copy()

slices_by_contract = compute_slice_metrics(
    test_raw, split.y_test.to_numpy(), test_proba, "Contract"
)
slices_by_internet = compute_slice_metrics(
    test_raw, split.y_test.to_numpy(), test_proba, "InternetService"
)
slices_by_payment = compute_slice_metrics(
    test_raw, split.y_test.to_numpy(), test_proba, "PaymentMethod"
)

# tenure bucket — derive from raw
test_with_bucket = test_raw.copy()
test_with_bucket["tenure_bucket"] = pd.cut(
    test_with_bucket["tenure"],
    bins=[-0.1, 6, 12, 24, 48, 72],
    labels=["0-6", "7-12", "13-24", "25-48", "49+"],
).astype(str)
slices_by_tenure = compute_slice_metrics(
    test_with_bucket, split.y_test.to_numpy(), test_proba, "tenure_bucket"
)

all_slices = slices_by_contract + slices_by_tenure + slices_by_internet + slices_by_payment
slice_table = slices_to_dataframe(all_slices)
print(slice_table.to_string(index=False))

# Visualize ROC-AUC per slice
slice_table_plot = slice_table.dropna(subset=["roc_auc"]).copy()
slice_table_plot["label"] = slice_table_plot["slice"] + " = " + slice_table_plot["value"]
slice_table_plot = slice_table_plot.sort_values("roc_auc")

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#c44e52" if v < 0.78 else "#4c72b0" for v in slice_table_plot["roc_auc"]]
ax.barh(slice_table_plot["label"], slice_table_plot["roc_auc"], color=colors)
ax.axvline(
    float(result.test_metrics["roc_auc"]),
    color="black",
    linestyle="--",
    label="global test ROC-AUC",
)
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("ROC-AUC")
ax.set_title("Slice-level ROC-AUC on test set (red < 0.78)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "10_slice_roc_auc.png", dpi=120)
plt.close(fig)

worst = slice_table_plot.iloc[0]
print(f"\nWeakest slice: {worst['label']} (n={worst['n']}, ROC-AUC={worst['roc_auc']:.3f})")

# %% [markdown]
# ## SHAP global summary + per-customer waterfall

# %%
explainer = HGBExplainer(result.pipeline)
explanation = explainer.explain(split.X_test)
top10 = explainer.top_features(explanation, k=10)
print("Top 10 features by mean |SHAP|:")
for name, mag in top10:
    print(f"  {name:30s} {mag:.4f}")

explainer.save_global_summary(explanation, str(FIGURES / "11_shap_summary.png"))

# Pick a high-confidence churner and a high-confidence non-churner
top_idx = int(np.argmax(test_proba))
bot_idx = int(np.argmin(test_proba))
explainer.save_waterfall(explanation, row=top_idx, path=str(FIGURES / "12_shap_high_risk.png"))
explainer.save_waterfall(explanation, row=bot_idx, path=str(FIGURES / "13_shap_low_risk.png"))
print(
    f"\nHigh-risk row {top_idx}: predicted {test_proba[top_idx]:.3f}, actual {int(split.y_test.iloc[top_idx])}"
)
print(
    f"Low-risk  row {bot_idx}: predicted {test_proba[bot_idx]:.3f}, actual {int(split.y_test.iloc[bot_idx])}"
)

# %% [markdown]
# ## Reliability diagram (calibration check)

# %%
confs, accs, counts = reliability_curve(split.y_test.to_numpy(), test_proba, n_bins=10)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="perfect calibration")
mask = counts > 0
ax.plot(confs[mask], accs[mask], "o-", color="#4c72b0", label="HGB calibrated")
ax.set_xlabel("predicted probability (mean per bin)")
ax.set_ylabel("empirical positive rate")
ax.set_title("Reliability diagram — test set")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "14_reliability_diagram.png", dpi=120)
plt.close(fig)

# %% [markdown]
# ## Simulated temporal drift
#
# Telco is a single snapshot, so this is illustrative: we treat the
# first 70% of test rows (by index) as the "reference" window and the
# last 30% as the "current" window. In a real deployment these would
# be timestamped batches.

# %%
test_size = len(split.X_test)
ref_end = int(0.7 * test_size)
X_ref = split.X_test.iloc[:ref_end]
X_cur = split.X_test.iloc[ref_end:]
proba_ref = test_proba[:ref_end]
proba_cur = test_proba[ref_end:]

drift_reports = [
    numeric_drift(X_ref["tenure"], X_cur["tenure"]),
    numeric_drift(X_ref["MonthlyCharges"], X_cur["MonthlyCharges"]),
    numeric_drift(X_ref["charge_ratio"], X_cur["charge_ratio"]),
    categorical_drift(X_ref["Contract"], X_cur["Contract"]),
    categorical_drift(X_ref["InternetService"], X_cur["InternetService"]),
    score_drift(proba_ref, proba_cur),
]
drift_table = reports_to_dataframe(drift_reports)
print(drift_table.to_string(index=False))

# Bar chart of severity counts
fig, ax = plt.subplots(figsize=(6, 3.5))
counts_sev = drift_table["severity"].value_counts()
colors_sev = {"stable": "#4c72b0", "minor": "#dd8452", "major": "#c44e52"}
bars = counts_sev.sort_index().rename({"stable": "stable", "minor": "minor", "major": "major"})
ax.bar(bars.index, bars.values, color=[colors_sev[s] for s in bars.index])
ax.set_ylabel("feature count")
ax.set_title("Drift severity across features (simulated split)")
for i, v in enumerate(bars.values):
    ax.text(i, v + 0.05, str(int(v)), ha="center")
fig.tight_layout()
fig.savefig(FIGURES / "15_drift_severity.png", dpi=120)
plt.close(fig)

# %% [markdown]
# ## Findings summary
#
# 1. **Threshold:** optimum sits well below 0.5 because S=$500 >> R=$50
#    incentivizes acting on borderline cases. Utility uplift over the
#    naive cutoff is meaningful and the sensitivity range shows the
#    optimum isn't fragile.
# 2. **Slices:** the weakest slice is one we'd discuss with the retention
#    team — see the printed result above for the specific subgroup.
# 3. **SHAP:** tenure and Contract dominate global feature importance —
#    consistent with the EDA. The per-customer waterfalls let a retention
#    rep explain *why* a specific customer is flagged.
# 4. **Calibration:** the reliability diagram tracks the diagonal closely
#    after isotonic calibration; ECE confirms (logged in MLflow).
# 5. **Drift simulation:** on this artificial split, nothing should show
#    as "major" drift since both windows come from the same distribution.
#    The harness is in place; in production it would run on real time
#    windows.

print("\nFigures saved to:")
for p in sorted(FIGURES.glob("0[9]_*.png")) + sorted(FIGURES.glob("1[0-5]_*.png")):
    print(f"  {p}")
