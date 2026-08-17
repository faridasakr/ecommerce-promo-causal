# Did Free Shipping Actually Work?

**[▶ Live demo](https://ecommerce-promo-causal-k8unq4zcnx7cspfo3ru8f4.streamlit.app)**

**A causal study of an e-commerce promotion — where the obvious analysis overstates the effect by 3.6×, and where the profitable answer is the opposite of what revenue alone suggests.**

---

## The decision

| Segment | Incremental revenue | Gross profit @45% | Shipping subsidy | **Net contribution** | 95% CI | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Low spend | $7.98 | $3.59 | −$2.30 | **+$1.29** | [+1.05, +1.43] | ✅ evidence supports targeting |
| Mid spend | $7.12 | $3.21 | −$2.95 | **+$0.25** | [+0.06, +0.58] | ✅ evidence supports targeting |
| High spend | $4.01 | $1.81 | −$3.89 | **−$2.09** | [−2.37, −1.39] | ❌ evidence suggests this segment destroys contribution |

**Recommendation: target the promotion at low-spend and mid-spend customers, and withhold it from high-spend. Mid-spend clears break-even by only six cents at the lower bound, and its verdict is hostage to the cost assumptions rather than to the statistics — see below before acting on it.**

**The verdict column is a rule, not a judgement call.** It reads the contribution interval, not the point estimate:

| Interval | Verdict |
|---|---|
| Lower bound > 0 | evidence supports targeting |
| Upper bound < 0 | evidence suggests this segment destroys contribution |
| Spans 0 | economically uncertain; recommend a controlled test |

A positive point estimate whose interval spans zero is not evidence that a segment pays — it is evidence that the data cannot tell. Keying the decision off `net_contribution > 0` would quietly promote such a segment to "target it", which is the overclaim this whole project is a warning about. Earlier in this analysis mid-spend sat at exactly that point: a **+$0.25** point estimate with an interval of [−0.02, +0.57]. The rule would have called it *economically uncertain* and asked for a controlled test; a point-estimate rule would have shipped it. The same three strings are emitted verbatim by the pipeline, the Streamlit app, and the LLM stakeholder layer, so no surface can soften the wording independently.

The subsidy scales with how many people *order*, not how much they spend. High-spend customers order more often, so you pay shipping on many purchases that would have happened anyway — which is how a segment can generate real incremental revenue and still lose money.

Two things a revenue-only analysis would have gotten wrong: it ranks mid-spend ($7.12) as nearly as good as low-spend ($7.98), when in contribution terms one clears break-even five times as wide as the other; and it makes high-spend look worth subsidising when it destroys $2.09 per customer.

**What actually decides mid-spend is not the statistics.** Its interval excludes zero, and stably so — re-running the bootstrap across three seeds at 50 and 150 replicates puts the lower bound between +$0.03 and +$0.11, never touching zero. But the segment breaks even at a shipping cost of **$7.06** against the **$6.50** assumed here. An 8.6% error in the carrier rate flips the recommendation; so does a blended margin below 41.4% against the 45% assumed. Both numbers are illustrative placeholders, not finance-sourced. Mid-spend is therefore a decision to make with finance's real figures in hand, not one this analysis settles on its own — which is what the break-even columns in `results/economics.csv` are there to make checkable.

**The subsidy multiplier has to be causal too.** How many orders you subsidise is the purchase rate you would face *if the segment were treated* — P(purchase | do(T=1)) — not the purchase rate observed among customers who chose the promo. Those customers self-selected, and the traits that drove uptake also drive purchasing, so the observed rate runs high:

| Segment | Observed rate (treated) | Causal rate, do(T=1) | Gap |
|---|---:|---:|---:|
| Low spend | 0.402 | 0.354 | −0.048 |
| Mid spend | 0.500 | 0.454 | −0.046 |
| High spend | 0.633 | 0.599 | −0.034 |

Using the observed rate would have quietly put the selection bias back into the decision — on the *cost* side, after removing it from the revenue side. It inflates the subsidy by 3–5pp of orders, which is $0.22–$0.32 per customer: enough to push mid-spend from marginally positive to marginally negative. The causal rate is estimated by running the same cross-fitted AIPW on a binary purchase indicator, so the cost side gets the same treatment as the revenue side. Both rates are reported in `results/economics.csv` so the correction is auditable rather than asserted.

**The interval comes from one bootstrap of the whole chain, not two composed ones.** The contribution depends on two estimated quantities — the revenue effect and the purchase probability — so each replicate resamples a segment once and re-runs the entire chain on it: revenue ATE → P(purchase | do(T=1)) → net contribution. Percentiles are taken over the contribution draws directly.

That matters because the two estimates are strongly *positively* correlated across replicates. Since contribution is a **difference** — margin × revenue − shipping × rate — positive correlation makes the two terms move together and partially cancel, so the correct interval is **narrower** than either shortcut, and the more correlated the segment, the bigger the gap:

| Segment | corr(revenue, rate) | Revenue CI only | Composed as independent | Joint (correct) |
|---|---:|---:|---:|---:|
| Low spend | +0.88 | 0.148 | 0.164 | 0.092 |
| Mid spend | +0.58 | 0.158 | 0.167 | 0.134 |
| High spend | +0.34 | 0.295 | 0.298 | 0.284 |

Standard deviation of the net contribution across bootstrap replicates, from `results/contribution_bootstrap.csv`. Holding the rate fixed overstates the spread by **60% / 18% / 4%**; composing two separately-bootstrapped quantities as if independent overstates it by **77% / 25% / 5%**. Both errors scale with the correlation, which is exactly the quantity a composed interval throws away.

Note the direction is not something to reason out in advance. Adding a second source of uncertainty *sounds* like it should widen the interval, and for a sum it would. For a difference with positively correlated terms it narrows. That is why the resampling has to be joint rather than assembled from parts — the covariance is doing real work and only a single resampling of the whole chain captures it.

## The estimate behind it

| Estimator | Target estimand | Target population | Estimate | 95% CI | True value | Error |
|---|---|---|---:|---:|---:|---:|
| Naive difference in means | ATE | full sample | $23.03 | [22.40, 23.57] | — | — |
| OLS regression adjustment | ATE | full sample | $7.09 | [6.60, 7.73] | $5.87 | +21% |
| Propensity score matching | **ATT** | **matched treated** | $7.08 | — | **$5.98** | +18% |
| IPW (stabilised) | ATE | trimmed (logistic) | $6.38 | [5.68, 7.27] | $5.87 | +9% |
| **AIPW (cross-fitted, doubly robust)** | ATE | trimmed (cross-fit) | **$6.42** | [5.98, 7.16] | $5.87 | +9% |

**PSM has no confidence interval on purpose.** The ordinary nonparametric bootstrap is *invalid* for nearest-neighbour matching with a fixed number of matches — Abadie & Imbens (2008), "On the Failure of the Bootstrap for Matching Estimators". The estimator is not smooth enough for the bootstrap to be consistent, and the failure is asymptotic: more replicates do not fix it. Reporting an interval anyway would look like quantified uncertainty while resting on nothing, which is worse than reporting none. The point estimate stays — matching is still a useful classical baseline — and the interval is simply withheld with the reason attached.

The **estimated causal effect** is **$6.42 per customer** (95% CI $5.98–$7.16), under conditional exchangeability, positivity, and SUTVA. Because this is synthetic data, the planted truth is known to be **$5.87** — on real data that column would not exist, which is precisely why the diagnostics and stated assumptions carry the weight.

**On the naive row:** it has no true value because it is not estimating one. A difference between two self-selected groups is a description of who opted in, not an attempt at a causal quantity. Scoring it against the ATE would imply it was aiming at the ATE and missing, when the honest statement is that it targets nothing causal at all. Its $23.03 is still the number most analyses would report, which is the point of showing it.

**On the two right-hand columns:** trimming and matching change *what is being estimated*, not just how well. Each estimator is therefore scored against the mean individual treatment effect over the units it actually used — the full sample for OLS, the trimmed sample for IPW and AIPW (which trim different units, because they use different propensity models), and the matched treated units for PSM. Those truths are computed from the per-customer effects at scoring time rather than hard-coded, so the target follows the estimator instead of the estimator being graded against someone else's population.

The correction turns out to be small here: the four targets are $5.867, $5.980, $5.869, and $5.868, so they round to the same cents and the Error column barely moves. That is a fact about this DGP — the trimmed units happen to have near-average effects — not a general result, and it is only visible *because* the targets are computed separately. PSM's is the substantive gap: its ATT refers to **18,664 matched treated customers**, not all 50,000 and not even all 18,671 treated.

A policy question about extending the promo to everyone needs the ATE; a question about whether it paid off for the people who took it needs the ATT. They are close under this DGP, but that is a property of this data, not a licence to conflate them.

---

## The lesson worth stealing

**The doubly robust estimator was only 9% off — and its 95% confidence interval still excluded the true value.**

That is not a bug. Bootstrap intervals quantify *sampling variability*: how much the estimate would move if you redrew the sample. They say nothing about *model misspecification* — systematic bias from a propensity model that doesn't capture the true assignment mechanism. So a tight interval around a slightly biased estimate is exactly what you should expect, and exactly what a stakeholder will misread as precision.

In this run, 1 of the 3 intervals contained its target. That is what happened in this one sample — not a measured coverage rate. Calling it "under-coverage" would claim something about how the procedure behaves in repeated sampling, which a single realization cannot establish; that would take a simulation study over many replications. The mechanism above is the transferable part, and it holds regardless of what this particular draw did.

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

In this realization, 2 of the 3 nominal 95% intervals did not contain their target. That is non-coverage in one sample, **not** an estimate of the procedure's coverage rate — establishing a rate would require a simulation study over many replications.

```
Estimation error
├── selection bias, unadjusted            +$17.17  (naive estimator)
├── residual bias after adjustment         +$0.51 to +$1.22
│   └── driven by unmodelled effect heterogeneity in the propensity model
├── intervals missing their target          2 of 3 in this realization
│   └── incl. AIPW: 9% point error, yet its interval misses
│   └── bootstrap CIs capture sampling noise, not model bias  ← see "the lesson"
└── estimand drift from trimming            0.24% of sample

Decision-layer error
├── revenue-optimal ≠ profit-optimal        high-spend is revenue-positive, contribution-negative
├── decision hinges on cost assumptions     mid-spend break-even at $7.06 vs $6.50 assumed
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
make test           # 25 fast tests
make test-all       # + 2 full-data integration tests
make app            # launch the Streamlit demo
```

Set `ANTHROPIC_API_KEY` to enable the live LLM explanation layer; without it, `explain.py` runs in offline template mode so the pipeline stays testable in CI.

## Tests

27 tests across two tiers — fast unit/property tests on an 8k fixture, plus full-data integration tests (`-m slow`) with a tighter 15% tolerance.

Coverage includes: deterministic generation; confounding is genuinely strong (max |SMD| > 0.3); the naive estimator *is* badly biased; IPW/AIPW recover the planted effect; AIPW is reproducible at a fixed seed; propensity scores contain no NaN/degenerate values; post-trim weights stay bounded; all SMDs pass after weighting; both propensity models are diagnosed; cleaning handles currency/duplicates/missingness without mutating the caller's frame; the economics layer flags loss-making segments and reports break-evens; both guardrails fire correctly and don't false-positive; the system prompt retains its refusal and CI rules; and no analysis module reads the answer key.
