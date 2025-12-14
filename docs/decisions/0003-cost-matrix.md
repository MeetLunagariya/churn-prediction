# 0003 — Cost matrix for retention decisions

- **Date:** 2026-05-13
- **Status:** Accepted (illustrative; production deployment requires
  retention-team-validated values)

## Context

Choosing the model's decision threshold at 0.5 is arbitrary. Retention
decisions are asymmetric:

- Missing a churner (false negative) loses the customer's expected
  lifetime value.
- Wasting a retention offer on a non-churner (false positive) costs
  the offer amount.

We need an explicit cost matrix so the model's operating threshold can
be chosen to maximize business outcomes, not accuracy.

## Decision

Adopt the "utility vs. doing nothing" formulation:

```
utility(threshold) = TP * (S - R) - FP * R
```

with these **illustrative** default values for the demo:

| Symbol | Meaning | Default | Source |
|---|---|---|---|
| S | Expected revenue retained per save (LTV impact) | **$500** | Half of a 24-month telco LTV (~$1000) under assumption of 50% retention-success rate; published industry estimates |
| R | Cost of a retention offer (discount/call) | **$50** | One-month discount on a $50-100 plan |

TN and FN contributions to the utility-vs-baseline formulation are zero
because they match the do-nothing counterfactual.

## Consequences

- The threshold is data-dependent: with $S \\gg R$, the optimum moves
  *below* 0.5 (it's cheap to act on borderline cases). Empirically the
  Telco data + HGB calibrated model lands the optimum around 0.30 — the
  exact value lives in `reports/model_card.md` after `make train`.
- We report **utility uplift over the naive 0.5 cutoff** as the
  decision metric, alongside ROC-AUC and PR-AUC.
- We also report **sensitivity range** (thresholds within 5% of optimum
  utility), which signals whether the optimum is fragile.

## Simplifications (must be revisited for production)

1. **100% retention success rate.** Real retention offers convert only
   a fraction of the time. The right generalization is:

   ```
   TP_value = p_save * (S - R) + (1 - p_save) * (-R) = p_save * S - R
   ```

   Add `p_save` to the cost API when we have an A/B-validated estimate.

2. **Single LTV across customers.** In reality, LTV varies — a long-
   tenure two-year-contract customer is worth more than a one-month
   high-charge customer. Per-customer S would let us prioritize
   highest-value saves.

3. **Single retention offer.** Different offer tiers (discount vs. plan
   upgrade vs. account manager call) have different (cost, conversion)
   profiles. Bandit-style offer selection is a future extension.

## Alternatives considered

- **F-beta maximization.** Tunes the F1-vs-recall balance but doesn't
  translate cleanly to dollars. Rejected — the cost matrix is more
  defensible in a retention-team conversation.
- **Profit curves.** Same idea, different visualization. We render a
  utility-vs-threshold curve in `reports/figures/threshold_utility.png`
  to support this view.
