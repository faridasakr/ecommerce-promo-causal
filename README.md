---
title: Did Free Shipping Actually Work?
emoji: 📦
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
license: mit
---

# Did Free Shipping Actually Work?

**A causal study of an e-commerce promotion — where the obvious analysis overstates the effect by 3.6×, and where the profitable answer is the opposite of what revenue alone suggests.**

---

## The decision

| Segment | Incremental revenue | Gross profit @45% | Shipping subsidy | **Net contribution** | 95% CI | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Low spend | $7.98 | $3.59 | −$2.62 | **+$0.98** | [+0.60, +1.14] | Profitable ✅ |
| Mid spend | $7.12 | $3.21 | −$3.25 | **−$0.04** | [−0.32, +0.28] | **Too close to call** ⚠️ |
| High spend | $4.01 | $1.81 | −$4.11 | **−$2.31** | [−2.89, −1.72] | Loss-making ❌ |

**Recommendation: target the promotion at low-spend customers only. Hold mid-spend pending a larger sample — its contribution interval straddles zero, so the current data cannot tell you which side of break-even it falls on.**

The subsidy scales with how many people *order*, not how much they spend. High-spend customers order more often, so you pay shipping on many purchases that would have happened anyway — which is how a segment can generate real incremental revenue and still lose money.

Two things a revenue-only analysis would have gotten wrong: it ranks mid-spend ($7.12) as nearly as good as low-spend ($7.98), when in contribution terms one is profitable and the other is indistinguishable from zero; and it makes high-spend look worth subsidising when it destroys $2.31 per customer.

The causal CIs are propagated through the margin arithmetic rather than discarded at the handoff — which is what surfaces the mid-spend ambiguity instead of hiding it behind a tidy point estimate.

## The estimate behind it

| Estimator | Target estimand | Estimate | 95% CI | True value | Error |
|---|---|---:|---:|---:|---:|
| Naive difference in means | ATE | $23.03 | [22.40, 23.57] | $5.87 | **+293%** |
| OLS regression adjustment | ATE | $7.09 | [6.60, 7.73] | $5.87 | +21% |
| Propensity score matching | **ATT** | $7.08 | [5.32, 7.76] | **$5.98** | +18% |
| IPW (stabilised) | ATE | $6.38 | [5.68, 7.27] | $5.87 | +9% |
| **AIPW (cross-fitted, doubly robust)** | ATE | **$6.42** | [5.98, 7.16] | $5.87 | +9% |

The **estimated causal effect** is **$6.42 per customer** (95% CI $5.98–$7.16), under conditional exchangeability, positivity, and SUTVA. Because this is synthetic data, the planted truth is known to be **$5.87** — on real data that column would not exist, which is precisely why the diagnostics and stated assumptions carry the weight.

**On the estimand column:** propensity score matching targets the ATT — the effect among customers who *actually used* the promo — and is scored against the true ATT ($5.98), not the ATE ($5.87). Under this DGP they happen to be close, but that is a property of this data, not a licence to conflate them. A policy question about extending the promo to everyone needs the ATE; a question about whether it paid off for the people who took it needs the ATT.

---

## The lesson worth stealing

**The doubly robust estimator was only 9% off — and its 95% confidence interval still excluded the true value.**

That is not a bug. Bootstrap intervals quantify *sampling variability*: how much the estimate would move if you redrew the sample. They say nothing about *model misspecification* — systematic bias from a propensity model that doesn't capture the true assignment mechanism. So a tight interval around a slightly biased estimate is exactly what you should expect, and exactly what a stakeholder will misread as precision.

Only 2 of 5 intervals contain their target. Any analysis that reports a CI as though it bounds total error is making this mistake.

---

## Why this project exists

Most "did our campaign work?" analyses are a difference in group means. That number is almost always wrong, and wrong in a predictable direction, because people who opt into a promotion differ systematically from people who don't.

This project quantifies exactly how wrong — on synthetic data where the true effect is known, so every estimator can be scored rather than argued about — and then carries the estimate through to an actual targeting decision.

## Problem

An online retailer ran a 4-week free-shipping promotion. 37% of customers used it. Leadership wants to know whether to run it again, and for whom.

The obstacle is self-selection: promo uptake is driven by email engagement, prior spend, and tenure — the same variables that drive baseline revenue.

## Data

Synthetic, generated in `src/generate_data.py` using the potential-outcomes framework: every customer gets both Y(0) and Y(1), and only the one matching their actual treatment is revealed. Because both are generated, the true ATE and ATT are known exactly and written to `data/ground_truth/answer_key.json`.

The promotion is planted to work through two channels, both stronger for price-sensitive shoppers:

- **Incidence** — a 5.5pp lift in the probability of purchasing at all
- **Basket size** — a $12 lift among people who would have purchased anyway

Realistic damage is layered on afterward (currency-formatted strings, inconsistent categorical casing, missing engagement scores, duplicate rows) so cleaning is real work.

### Ground-truth isolation

The answer key is **committed on purpose** — hiding it would make the scoring unverifiable. The guarantee comes from enforced separation, not secrecy:

- only `run_analysis.py` reads it, and only in the final scoring step
- `test_ground_truth_not_read_by_analysis_modules` parses each analysis module's AST and fails the build on any live-code reference
- it must be excluded from the agent environment in the follow-on project that benchmarks against this data — see `data/ground_truth/README.md`

## Architecture

```
data/raw/customers.csv
        │
        ▼
   prepare.py ──── clean: dedupe, parse currency, normalise cases,
        │                impute missingness + indicator column
        ▼
  design matrix (12 covariates, standardised + one-hot)
        │
        ├──► diagnostics.py ──┬── logistic propensity ──► balance, overlap  [IPW, PSM]
        │                     └── crossfit GBM propensity ─► balance, overlap [AIPW]
        │                                                 └── stress test
        ▼
   estimators.py
        │  naive → OLS → PSM(ATT) → IPW → AIPW(cross-fitted)
        │  each with bootstrap CIs and a declared target estimand
        ▼
   economics.py ──► margin × revenue − shipping subsidy = contribution by segment
        │
        ▼
   results/summary.json
        │
        ▼
    explain.py ──► LLM stakeholder layer (3 guardrails)
```

**Two propensity models, diagnosed separately.** IPW and PSM use a logistic model; AIPW uses cross-fitted gradient boosting. They are reported side by side in `results/propensity.json` so a clean balance table from one model cannot vouch for an estimate produced by the other. Balance is computed on the *trimmed* population, using the same weights the estimator applies — reporting balance on the full sample while estimating on a trimmed one describes two different populations.

## Key engineering decisions

**Decision:** Cross-fit the AIPW nuisance models (2-fold) rather than fitting on the full sample.
**Alternatives:** Single-fit AIPW; parametric outcome models.
**Evidence:** Fitting flexible ML nuisance models and evaluating them on the same rows induces overfitting bias in the influence function. Cross-fitting is the Chernozhukov double/debiased ML correction.
**Cost:** 2× the nuisance fits.
**Conclusion:** Worth it — and switching from `GradientBoostingRegressor` to `HistGradientBoostingRegressor` cut runtime from 7.3s to 1.1s, which made 200-replicate bootstrapping feasible at all.

**Decision:** Label each estimator with its target estimand and score against the matching ground truth.
**Alternatives:** Report all five against the ATE; drop PSM entirely.
**Evidence:** PSM matches controls to treated units, reweighting to the treated covariate distribution — it targets the ATT. Scoring it against the ATE would have inflated its apparent error and, worse, implied the two are interchangeable.
**Cost:** An extra column and a second ground-truth value.
**Conclusion:** Keep both estimators and make the distinction explicit. The comparison is only meaningful if it is apples-to-apples.

**Decision:** Trim units with propensity outside [0.02, 0.98].
**Alternatives:** Keep everything; weight truncation at a percentile.
**Evidence:** Max IPW weight before trimming was 63×. Units with no counterfactual counterpart inflate variance without adding identification.
**Cost:** Drops ~0.24% of the sample and changes the estimand to the trimmed population.
**Conclusion:** Deploy, and state the estimand change rather than hiding it.

**Decision:** The LLM never sees raw data, only `results/summary.json`.
**Alternatives:** Give the model table access and let it analyse.
**Evidence:** Handed the raw table, an LLM will reproduce the naive $23 difference — the exact error this project exists to prevent. Statistical reasoning belongs in tested Python.
**Cost:** The assistant cannot answer questions outside the precomputed summary.
**Conclusion:** That constraint is a feature; the model is instructed to refuse those questions.

**Decision:** Carry the causal estimate through to contribution margin rather than stopping at incremental revenue.
**Alternatives:** Report the effect size and let stakeholders do the economics.
**Evidence:** The revenue ranking (low > mid > high) and the profit ranking are *not* the same decision — mid-spend is revenue-positive and contribution-negative, because subsidy scales with order frequency rather than basket size.
**Cost:** Requires margin and shipping-cost assumptions, which are declared and reported with break-even thresholds so they can be challenged.
**Conclusion:** Ship it. "The effect is $6.42" is a finding; "target low-spend only" is a decision.

## Error taxonomy

```
Estimation error
├── selection bias, unadjusted            +$17.17  (naive estimator)
├── residual bias after adjustment         +$0.51 to +$1.22
│   └── driven by unmodelled effect heterogeneity in the propensity model
├── CI under-coverage                       3 of 5 intervals exclude their target
│   └── incl. AIPW: 9% point error, yet its CI misses
│   └── bootstrap CIs capture sampling noise, not model bias  ← see "the lesson"
└── estimand drift from trimming            0.24% of sample

Decision-layer error
├── revenue-optimal ≠ profit-optimal        mid-spend flips from #2 to unresolvable
├── contribution CI straddles zero          mid-spend: [-0.32, +0.28], no decision possible
├── margin/shipping assumptions unverified  break-even columns bound the risk
└── static analysis ignores LTV             acquisition/retention effects unmodelled

Explanation-layer failure
├── causal overclaim about a confounder     caught by validate_response()
│   └── missed when phrased as a category LEVEL ("paid search") rather than the
│       column name ("channel") — found by tests, fixed by expanding the watchlist
├── silent numeric drift ($6.42 → $6.40)    caught by validate_numeric_fidelity()
│   └── first implementation compared formatted strings, so "6.4" matched both;
│       fixed by comparing floats to the cent
└── regex validators cannot parse negation or hedging — known limitation
```

## Unmeasured-confounding stress test

| γ | Estimate |
|---|---:|
| 0.0 | $6.42 |
| 0.2 | $6.53 |
| 0.3 | $7.92 |
| 0.5 | $11.63 |

**This is a stress test, not a formal bound.** It is not a Rosenbaum bound or an E-value, and γ is not calibrated to any interpretable real-world confounding strength — it is a simulation knob that perturbs both treatment assignment and outcome. Across the scenarios tested the estimate remained positive, which says the result is not *obviously* fragile and nothing stronger than that. A formal E-value (VanderWeele & Ding) is the right next step.

## Limitations

- Synthetic data. The confounding structure is one I planted; real self-selection may be non-linear in ways these covariates cannot capture.
- Bootstrap CIs understate total uncertainty (see the lesson above).
- 2-fold cross-fitting is the minimum; 5-fold would be lower-variance at higher cost.
- Guardrail validators are regex/tolerance based and will miss creatively-phrased overclaims. The principled version is structured output — have the model emit a Pydantic schema whose `causal_claims` list is validated field-by-field before the prose renders.
- Contribution analysis is static: no acquisition, retention, or LTV effects, which would need a longitudinal design.
- No temporal component — difference-in-differences on pre/post data would identify the effect under weaker assumptions.

## Setup

```bash
git clone <repo> && cd ecommerce-promo-causal
make setup          # venv + dependencies
make data           # generate the synthetic dataset
make analysis       # full causal pipeline (~4 min with 200 bootstraps)
make test           # 23 fast tests
make test-all       # + 2 full-data integration tests
make app            # launch the Streamlit demo
```

Set `ANTHROPIC_API_KEY` to enable the live LLM explanation layer; without it, `explain.py` runs in offline template mode so the pipeline stays testable in CI.

## Tests

25 tests across two tiers — fast unit/property tests on an 8k fixture, plus full-data integration tests (`-m slow`) with a tighter 15% tolerance.

Coverage includes: deterministic generation; confounding is genuinely strong (max |SMD| > 0.3); the naive estimator *is* badly biased; IPW/AIPW recover the planted effect; AIPW is reproducible at a fixed seed; propensity scores contain no NaN/degenerate values; post-trim weights stay bounded; all SMDs pass after weighting; both propensity models are diagnosed; cleaning handles currency/duplicates/missingness without mutating the caller's frame; the economics layer flags loss-making segments and reports break-evens; both guardrails fire correctly and don't false-positive; the system prompt retains its refusal and CI rules; and no analysis module reads the answer key.
