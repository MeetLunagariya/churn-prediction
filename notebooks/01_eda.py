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
# # 01 — EDA: IBM Telco Customer Churn
#
# Goal: build mental model of the dataset before any modeling. Specifically:
#
# 1. Confirm the data contract (shape, dtypes, missingness).
# 2. Quantify class imbalance.
# 3. Identify candidate signal: which features actually separate churners from non-churners?
# 4. Surface obvious leakage risks before they corrupt the model.
#
# Findings are summarized at the bottom; figures land in `reports/figures/`.

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Run relative to project root regardless of whether this is executed
# as a script (from root) or as a notebook (from notebooks/).
if Path.cwd().name == "notebooks":
    os.chdir(Path.cwd().parent)

from churn.data.load import load_raw

FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

df = load_raw()
print(f"shape: {df.shape}")
print(f"churn rate: {(df['Churn'] == 'Yes').mean():.3f}")

# %% [markdown]
# ## Shape, dtypes, missingness

# %%
print("dtypes:")
print(df.dtypes.value_counts())
print()
missing = df.isna().sum()
print("missing values per column (only columns with >0 NaN):")
print(missing[missing > 0])

# %% [markdown]
# Only `TotalCharges` has missing values, and only 11 of them. Inspecting which
# rows — these correspond to customers with `tenure == 0`, i.e. brand-new
# customers who haven't been billed yet. We'll either impute as 0 or drop them
# when building features; the choice doesn't materially change downstream
# metrics on this dataset.

# %%
print(df[df["TotalCharges"].isna()][["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])

# %% [markdown]
# ## Class balance
#
# 26.5% positive class — moderately imbalanced. Not severe enough to require
# SMOTE in the baseline, but PR-AUC will be a more honest headline metric than
# ROC-AUC.

# %%
fig, ax = plt.subplots(figsize=(5, 3.5))
churn_counts = df["Churn"].value_counts()
ax.bar(churn_counts.index, churn_counts.values, color=["#4c72b0", "#dd8452"])
for i, v in enumerate(churn_counts.values):
    ax.text(i, v + 50, f"{v:,}\n({v / len(df):.1%})", ha="center")
ax.set_ylabel("customer count")
ax.set_title("Class balance — Churn outcome")
fig.tight_layout()
fig.savefig(FIGURES / "01_class_balance.png", dpi=120)
plt.close(fig)

# %% [markdown]
# ## Tenure × Churn
#
# Tenure is the single strongest predictor by visual inspection: churners are
# heavily concentrated in the first ~12 months, while long-tenure customers
# almost never churn. This is going to drive most of the model's lift.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for label, color in [("No", "#4c72b0"), ("Yes", "#dd8452")]:
    sns.kdeplot(
        df.loc[df["Churn"] == label, "tenure"],
        ax=ax,
        label=f"Churn = {label}",
        fill=True,
        alpha=0.4,
        color=color,
    )
ax.set_xlabel("tenure (months)")
ax.set_title("Tenure distribution by churn class")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "02_tenure_by_churn.png", dpi=120)
plt.close(fig)

# %% [markdown]
# ## Monthly charges × Churn
#
# Churners also skew toward higher monthly charges — the modal churner pays
# around $70–$95/month, while the modal non-churner is bimodal: a cluster at
# $20–$25 (basic phone-only plans) and another at $80+ (full-bundle).

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for label, color in [("No", "#4c72b0"), ("Yes", "#dd8452")]:
    sns.kdeplot(
        df.loc[df["Churn"] == label, "MonthlyCharges"],
        ax=ax,
        label=f"Churn = {label}",
        fill=True,
        alpha=0.4,
        color=color,
    )
ax.set_xlabel("monthly charges ($)")
ax.set_title("Monthly charges distribution by churn class")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "03_monthly_charges_by_churn.png", dpi=120)
plt.close(fig)

# %% [markdown]
# ## Contract type is the biggest categorical lever

# %%
contract_rate = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values()
print("Churn rate by contract:")
print(contract_rate)

fig, ax = plt.subplots(figsize=(6, 3.5))
contract_rate.plot.barh(ax=ax, color="#dd8452")
ax.set_xlabel("churn rate")
ax.set_title("Churn rate by contract type")
for i, (_, v) in enumerate(contract_rate.items()):
    ax.text(v + 0.005, i, f"{v:.1%}", va="center")
ax.set_xlim(0, contract_rate.max() * 1.2)
fig.tight_layout()
fig.savefig(FIGURES / "04_churn_rate_by_contract.png", dpi=120)
plt.close(fig)

# %% [markdown]
# Month-to-month customers churn at 4–10× the rate of contract customers.
# This single feature alone could form a useful baseline rule.

# %% [markdown]
# ## Payment method as a secondary lever

# %%
payment_rate = (
    df.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values()
)
print("Churn rate by payment method:")
print(payment_rate)

fig, ax = plt.subplots(figsize=(7, 3.5))
payment_rate.plot.barh(ax=ax, color="#dd8452")
ax.set_xlabel("churn rate")
ax.set_title("Churn rate by payment method")
for i, (_, v) in enumerate(payment_rate.items()):
    ax.text(v + 0.005, i, f"{v:.1%}", va="center")
ax.set_xlim(0, payment_rate.max() * 1.2)
fig.tight_layout()
fig.savefig(FIGURES / "05_churn_rate_by_payment.png", dpi=120)
plt.close(fig)

# %% [markdown]
# Electronic-check payers churn ~3× as much as customers on automatic
# payment methods. Plausible mechanism: friction in payment renewal correlates
# with friction in account renewal.

# %% [markdown]
# ## Interaction: tenure × monthly charges
#
# A common intuition says "expensive customers are loyal because they sunk
# cost." The data says otherwise: **new customers paying high monthly charges**
# are the highest-risk segment. This interaction is a feature-engineering
# candidate for Week 2.

# %%
df_interact = df.copy()
df_interact["tenure_bucket"] = pd.cut(
    df_interact["tenure"],
    bins=[-0.1, 6, 12, 24, 48, 72],
    labels=["0-6", "7-12", "13-24", "25-48", "49+"],
)
df_interact["charge_bucket"] = pd.qcut(
    df_interact["MonthlyCharges"], q=4, labels=["q1-low", "q2", "q3", "q4-high"]
)

heat = (
    df_interact.groupby(["tenure_bucket", "charge_bucket"], observed=True)["Churn"]
    .apply(lambda s: (s == "Yes").mean())
    .unstack()
)
print("Churn rate by tenure × charge bucket:")
print(heat.round(3))

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(heat, annot=True, fmt=".2f", cmap="Oranges", ax=ax, cbar_kws={"label": "churn rate"})
ax.set_title("Churn rate by tenure × monthly-charge bucket")
ax.set_xlabel("monthly charge quartile")
ax.set_ylabel("tenure bucket (months)")
fig.tight_layout()
fig.savefig(FIGURES / "06_tenure_charge_interaction.png", dpi=120)
plt.close(fig)

# %% [markdown]
# Top-left corner (0–6 month tenure × top-quartile charges) hits ~55% churn,
# vs. ~3% in the bottom-right (long tenure × low charges) — an 18× gap.

# %% [markdown]
# ## Correlations between numeric features

# %%
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
churn_numeric = df[[*numeric_cols, "Churn"]].copy()
churn_numeric["churn"] = (churn_numeric["Churn"] == "Yes").astype(int)
corr = churn_numeric[[*numeric_cols, "churn"]].corr()
print("Correlations:")
print(corr.round(3))

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1)
ax.set_title("Numeric feature correlations")
fig.tight_layout()
fig.savefig(FIGURES / "07_numeric_correlations.png", dpi=120)
plt.close(fig)

# %% [markdown]
# `TotalCharges` is ~0.83 correlated with `tenure` and ~0.65 with
# `MonthlyCharges` — which makes sense because it is *constructed* from them.
# That's a leakage smell — see the next section.

# %% [markdown]
# ## Leakage check: TotalCharges ≈ tenure × MonthlyCharges?
#
# If `TotalCharges` is just the cumulative bill, it's deterministic from
# (tenure, MonthlyCharges). Two consequences:
#
# 1. We're not gaining new information from including all three.
# 2. Worse: at prediction time for a *new* customer we may have tenure &
#    monthly charge but not "total to date" — including it could leak future
#    information depending on snapshot semantics.

# %%
implied = df["tenure"] * df["MonthlyCharges"]
delta = df["TotalCharges"] - implied
clean = delta.dropna()
print("TotalCharges - tenure*MonthlyCharges:")
print(f"  median: {clean.median():.2f}")
print(f"  IQR:    [{clean.quantile(0.25):.2f}, {clean.quantile(0.75):.2f}]")
print(f"  mean abs: {clean.abs().mean():.2f}")
print(f"  pct within $5: {(clean.abs() < 5).mean():.1%}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(implied, df["TotalCharges"], alpha=0.2, s=8, color="#4c72b0")
limits = [0, df["TotalCharges"].max()]
ax.plot(limits, limits, color="red", linestyle="--", label="y = x")
ax.set_xlabel("tenure × MonthlyCharges (implied)")
ax.set_ylabel("TotalCharges (observed)")
ax.set_title("TotalCharges vs. (tenure × MonthlyCharges)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "08_total_charges_leakage_check.png", dpi=120)
plt.close(fig)

# %% [markdown]
# Most points cluster on `y = x` but with meaningful spread (median delta is
# small but the tails are wide — promotions, plan changes during tenure).
# `TotalCharges` carries some residual signal beyond the product, but
# we should treat it cautiously: drop it from the baseline, then re-add it
# in Week 2 only if a leakage-controlled ablation justifies it.

# %% [markdown]
# ## Findings summary
#
# 1. **Shape:** 7,043 customers × 21 features. Class balance 26.5% positive.
# 2. **Missingness:** only `TotalCharges` (11 rows, all tenure=0). Safe to
#    impute as 0 or drop.
# 3. **Top single-feature signals:**
#    - `tenure`: monotonic, churn concentrated in first year
#    - `Contract`: month-to-month customers churn at 4–10× contract rates
#    - `PaymentMethod`: electronic-check ~3× other methods
#    - `MonthlyCharges`: churners skew higher; non-churners are bimodal
# 4. **Surprising interaction:** new customers at high monthly charges churn
#    at ~55%, vs. ~3% for long-tenure low-charge customers. This is a
#    feature-engineering opportunity in Week 2.
# 5. **Leakage risk:** `TotalCharges ≈ tenure × MonthlyCharges` — drop from
#    baseline, justify re-inclusion later if at all.
# 6. **Modeling implications:**
#    - PR-AUC over ROC-AUC as the headline metric
#    - Linear baseline will likely hit ROC-AUC ~0.83 driven by tenure alone
#    - LightGBM should pick up the tenure × charges interaction without
#      explicit feature crossing
#    - Calibration will matter because the business decision threshold sits
#      in a probability region (0.25–0.40) where small errors flip decisions

print("\nFigures saved to:")
for p in sorted(FIGURES.glob("*.png")):
    print(f"  {p}")
