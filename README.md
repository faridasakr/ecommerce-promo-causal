# Did Free Shipping Actually Work?

**[▶ Live demo](https://ecommerce-promo-causal-k8unq4zcnx7cspfo3ru8f4.streamlit.app)**

**A causal study of an e-commerce promotion — where the obvious analysis overstates the effect by 3.6×, and where the profitable answer is the opposite of what revenue alone suggests.**

---

## The decision

| Segment | Incremental revenue | Gross profit @45% | Shipping subsidy | **Net contribution** | 95% CI | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Low spend | $7.98 | $3.59 | −$2.30 | **+$1.29** | [+1.05, +1.51] | ✅ evidence supports targeting |
| Mid spend | $7.12 | $3.21 | −$2.95 | **+$0.25** | [+0.03, +0.51] | ✅ evidence supports targeting |
| High spend | $4.01 | $1.81 | −$3.89 | **−$2.09** | [−2.49, −1.40] | ❌ evidence suggests this segment destroys contribution |

**Low- and mid-spend customers show positive incremental economics under the stated assumptions, making them the strongest candidates for a targeted-offer experiment; high-spend shows negative economics. Mid-spend clears break-even by only three cents at the lower bound, and its verdict is hostage to the cost assumptions rather than to the statistics — see below before acting on it.**

**These are effects of promo *uptake*, not of *offering* the promo.** The treatment here is whether a customer used free shipping, so every figure describes customers who took it up. The policy question — whom to offer it to — is an intention-to-treat quantity this observational design does not identify, and which is generally smaller. Read the segment results as evidence about where an offer experiment is most likely to pay, not as a directly identified offer effect.

**What the policy is worth, in dollars.** Per-customer contribution is not the decision-relevant number on its own — segments differ in size, so the policy total is what matters:

| Policy | Segments offered | n offered | **Total contribution** | 95% CI | Per eligible customer | Per targeted customer |
|---|---|---:|---:|---:|---:|---:|
| Blanket | all three | 50,000 | **−$9,175** | [−17,357, +4,672] | −$0.18 | −$0.18 |
| Targeted | low + mid spend | 33,332 | **+$25,662** | [+20,400, +31,316] | +$0.51 | +$0.77 |

**The blanket offer is net negative, and its interval spans zero.** High-spend holds a third of the customers and destroys $2.09 each, which is enough to swamp the gains elsewhere — so offering to everyone is not merely worse than targeting, it is indistinguishable from value-destroying. Targeting is clearly positive. That comparison is only visible on a weighted basis; per-customer averages across segments hide it.

Both rows use the identical formula — Σ over segments of *N·C* — and the same eligible-customer denominator, so they are comparable. An earlier version compared an unweighted mean across all segments against an unweighted mean of the profitable ones; those are different bases, and the mismatch flattered targeting twice over, ignoring segment sizes and silently redefining "per customer" between the two policies. Intervals come from the joint contribution draws summed **per replicate**, with the policy segment set held fixed at the final rule — re-selecting segments inside each draw would fold policy-selection uncertainty into the answer, which is a different question.

**The verdict column is a rule, not a judgement call.** It reads the contribution interval, not the point estimate:

| Interval | Verdict |
|---|---|
| Lower bound > 0 | evidence supports targeting |
| Upper bound < 0 | evidence suggests this segment destroys contribution |
| Spans 0 | economically uncertain; recommend a controlled test |

A positive point estimate whose interval spans zero is not evidence that a segment pays — it is evidence that the data cannot tell. Keying the decision off `net_contribution > 0` would quietly promote such a segment to "target it", which is the overclaim this whole project is a warning about. Earlier in this analysis mid-spend sat at exactly that point: a **+$0.25** point estimate with an interval of [−0.02, +0.57]. The rule would have called it *economically uncertain* and asked for a controlled test; a point-estimate rule would have shipped it. The same three strings are emitted verbatim by the pipeline, the Streamlit app, and the LLM stakeholder layer, so no surface can soften the wording independently.

The subsidy scales with how many people *order*, not how much they spend. High-spend customers order more often, so you pay shipping on many purchases that would have happened anyway — which is how a segment can generate real incremental revenue and still lose money.

Two things a revenue-only analysis would have gotten wrong: it ranks mid-spend ($7.12) as nearly as good as low-spend ($7.98), when in contribution terms one clears break-even five times as wide as the other; and it makes high-spend look worth subsidising when it destroys $2.09 per customer.

**What actually decides mid-spend is not the statistics.** Its interval excludes zero, and stably so — re-running the bootstrap across three seeds at 50 and 150 replicates puts the lower bound between +$0.03 and +$0.11, never touching zero, and the final 200-replicate run lands at +$0.03. Note the direction: estimating the tail better moved the bound *closer* to zero, not further from it. But the segment breaks even at a shipping cost of **$7.06** against the **$6.50** assumed here. An 8.6% error in the carrier rate flips the recommendation; so does a blended margin below 41.4% against the 45% assumed. Both numbers are illustrative placeholders, not finance-sourced. Mid-spend is therefore a decision to make with finance's real figures in hand, not one this analysis settles on its own — which is what the break-even columns in `results/economics.csv` are there to make checkable.

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
| Low spend | +0.87 | 0.176 | 0.191 | 0.117 |
| Mid spend | +0.66 | 0.159 | 0.168 | 0.130 |
| High spend | +0.43 | 0.303 | 0.306 | 0.287 |

Standard deviation of the net contribution across bootstrap replicates, from `results/contribution_bootstrap.csv`. Holding the rate fixed overstates the spread by **50% / 23% / 6%**; composing two separately-bootstrapped quantities as if independent overstates it by **63% / 29% / 7%**. Both errors scale with the correlation, which is exactly the quantity a composed interval throws away.

Note the direction is not something to reason out in advance. Adding a second source of uncertainty *sounds* like it should widen the interval, and for a sum it would. For a difference with positively correlated terms it narrows. That is why the resampling has to be joint rather than assembled from parts — the covariance is doing real work and only a single resampling of the whole chain captures it.

## Segment effects, and a negative control

**The spend result is a pipeline test, not a discovery.** The data-generating process plants effect modification in prior spend on purpose: the promo is written to help low-spend customers more. Recovering that pattern shows the segment machinery works — it is not a finding about customers, and presenting it as one would be circular. On real data the same analysis would be genuinely exploratory and would need pre-registration or a holdout to carry weight.

So the honest question is not "did we find heterogeneity?" but "does this pipeline find heterogeneity that isn't there?" **Region answers that.** It is drawn independently in the DGP, entering neither the treatment model nor the outcome model, so there is no effect modification to find. The identical segment analysis runs on it as a negative control:

| Segment | Prespecified modifier — ATE | 95% CI |
|---|---:|---:|
| Low spend | $7.98 | [7.12, 8.62] |
| Mid spend | $7.12 | [6.54, 7.86] |
| High spend | $4.01 | [2.93, 5.40] |

| Region | Negative control — ATE | 95% CI |
|---|---:|---:|
| midwest | $6.31 | [5.39, 7.51] |
| northeast | $5.54 | [4.32, 6.99] |
| south | $6.96 | [6.00, 8.12] |
| west | $6.92 | [5.86, 8.21] |

**Read the intervals, not the spread.** The region point estimates fan out by $1.42 — from $5.54 to $6.96 — even though the DGP plants nothing there. That is estimation noise, and it is exactly why a spread comparison is the wrong test: a naive reading of those four numbers would invent a story about regional demand. The more informative comparison is that **every pair of region intervals overlaps**, so none of the regional differences separates at this sample size — whereas across spend terciles, low and high do not overlap at all ([7.12, 8.62] against [2.93, 5.40]).

Be precise about what that does and does not show. Overlapping intervals are **not** evidence that no regional heterogeneity exists; they show no separation on the scale of the planted spend effect. This is a descriptive sanity check on the segment machinery, not a formal test of equality across regions — a real test would need a pre-specified contrast and the power to detect the effect size in question, neither of which is set up here. Absence of a detected difference is not a demonstrated absence of difference.

So the pipeline separates segments where effect modification is planted, and does not manufacture separation where none is. Had the regions separated as sharply as the spend terciles, the spend result would have looked like an artefact of the method rather than a property of the data, and the targeting recommendation built on it would have been worth much less. Both tables regenerate every run, and a test asserts the overlap pattern in both directions, so this is a standing check rather than a one-off reassurance.

## The estimate behind it

| Estimator | Target estimand | Target population | Estimate | 95% CI | True value | Error |
|---|---|---|---:|---:|---:|---:|
| Naive difference in means | descriptive difference in means — non-causal | full sample | $23.03 | [22.40, 23.57] | — | — |
| OLS adjusted treatment coefficient | ATE | full sample | $7.09 | [6.60, 7.73] | $5.87 | +21% |
| Propensity score matching | **ATT** | **matched treated** | $7.08 | — | **$5.98** | +18% |
| IPW (stabilised) | ATE | trimmed (logistic) | $6.38 | [5.68, 7.27] | $5.87 | +9% |
| **AIPW (cross-fitted, doubly robust)** | ATE | trimmed (cross-fit) | **$6.42** | [5.98, 7.16] | $5.87 | +9% |

**PSM has no confidence interval on purpose.** The ordinary nonparametric bootstrap is *invalid* for nearest-neighbour matching with a fixed number of matches — Abadie & Imbens (2008), "On the Failure of the Bootstrap for Matching Estimators". The estimator is not smooth enough for the bootstrap to be consistent, and the failure is asymptotic: more replicates do not fix it. Reporting an interval anyway would look like quantified uncertainty while resting on nothing, which is worse than reporting none. The point estimate stays — matching is still a useful classical baseline — and the interval is simply withheld with the reason attached.

The **estimated causal effect** is **$6.42 per customer** (95% CI $5.98–$7.16), under conditional exchangeability, positivity, and SUTVA. Because this is synthetic data, the planted truth is known to be **$5.87** — on real data that column would not exist, which is precisely why the diagnostics and stated assumptions carry the weight.

**On the OLS row:** it is the adjusted treatment coefficient from `Y ~ T + X` with no treatment-covariate interactions. Because the effect is heterogeneous under this DGP, that coefficient is a weighted average of segment effects and is not guaranteed to equal the population ATE. The model is deliberately left unexpanded — it is the plain-vanilla adjustment most analyses actually run, and its gap from the ATE is part of what the comparison shows.

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
        │  each with a declared target estimand; bootstrap CIs where the
        │  bootstrap is valid
        ▼
   diagnostics.py ─┬─ heterogeneity by spend (prespecified modifier)
        │           ├─ heterogeneity by region (negative control)
        │           ├─ causal incidence P(purchase | do(T=1)) per segment
        │           └─ joint bootstrap: ATE → incidence → contribution
        ▼
   economics.py ──► margin × revenue − shipping subsidy = contribution
        │           interval-based verdict → policy economics (total dollars)
        ▼
   results/stakeholder_summary.json     results/evaluation_summary.json
        │            (no ground truth)          (truth + scoring)
        ▼                                             │
    explain.py ──► LLM layer (prompt → schema → regex)│
                                                      ▼
                                        estimator-comparison tab only
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

**Decision:** The LLM never sees raw data or ground truth — only `results/stakeholder_summary.json`, and it must fill a validated schema.
**Alternatives:** Give the model table access and let it analyse.
**Evidence:** Handed the raw table, an LLM will reproduce the naive $23 difference — the exact error this project exists to prevent. Handed the answer key, it quotes a planted number that on real data does not exist. And prose alone cannot be checked, so the load-bearing claims move into schema fields that are compared to the analysis before anything renders.
**Cost:** The assistant cannot answer questions outside the precomputed summary.
**Conclusion:** That constraint is a feature; the model is instructed to refuse those questions.

**Decision:** Carry the causal estimate through to contribution margin rather than stopping at incremental revenue.
**Alternatives:** Report the effect size and let stakeholders do the economics.
**Evidence:** The revenue ranking (low > mid > high) and the profit ranking are *not* the same decision — mid-spend is revenue-positive and contribution-negative, because subsidy scales with order frequency rather than basket size.
**Cost:** Requires margin and shipping-cost assumptions, which are declared and reported with break-even thresholds so they can be challenged.
**Conclusion:** Ship it. "The uptake effect is $6.42" is a finding; "these segments are the strongest candidates for a targeted-offer experiment" is a decision.

## Error taxonomy

In this realization, 2 of the 3 nominal 95% intervals did not contain their target. That is non-coverage in one sample, **not** an estimate of the procedure's coverage rate — establishing a rate would require a simulation study over many replications.

```
Estimation error
├── selection bias, unadjusted            +$17.17  (naive estimator)
├── residual bias after adjustment         +$0.51 to +$1.22
│   └── may reflect finite-sample variation and/or nuisance-model
│       misspecification; their contributions are not separately
│       identified from this realization
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
└── regex validators cannot parse negation or hedging — now a backstop behind
    schema validation, not the primary defence

Deployment error
├── stale cache after redeploy    @st.cache_data keys on the decorated
│                                 function's source, not the files it reads.
│                                 A redeploy served the previous artefacts'
│                                 cached parse against new tab code, throwing
│                                 KeyError on a key that existed in the
│                                 committed data. Fixed by fingerprinting
│                                 artefact mtime+size into the cache key.
└── data contract untested        three artefact changes broke the app while
                                  every test passed; src/artifacts.py now
                                  declares the contract and tests enforce it.
```

The cache entry is worth dwelling on, because the symptom pointed away from the cause. The key was present and top-level in the committed artefact, and the app rendered correctly on a fresh local process — so the obvious hypotheses (wrong file, wrong nesting, renamed key) were all false, and "just run it locally" would not have reproduced it. What made the failure possible was that the *loader's source had not changed*, only the data it reads and the code that consumes the result. Any cache keyed on code rather than data has this hole.

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
- The regex backstop still cannot parse negation or hedging. It now sits behind schema validation rather than carrying the load, but prose claims outside the structured fields remain only inspectable, not checkable.
- Contribution analysis is static: no acquisition, retention, or LTV effects, which would need a longitudinal design.
- No temporal component — difference-in-differences on pre/post data would identify the effect under weaker assumptions.

**What this estimates, and what it doesn't.** The treatment is promo *uptake*, so the estimand is the effect of using free shipping among customers who could use it. The policy question — should we offer free shipping, and to whom — is an intention-to-treat question about being *offered* the promo, which is not the same quantity. With one-sided non-compliance the uptake effect generally exceeds the offer effect. Identifying the offer effect would need randomised encouragement or an instrument for uptake; neither exists in this observational design. Read the targeting recommendation as 'among customers who would take the offer'.

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

Coverage includes: deterministic generation; confounding is genuinely strong (max |SMD| > 0.3); the naive estimator *is* badly biased; IPW/AIPW recover the planted effect; AIPW is reproducible at a fixed seed; propensity scores contain no NaN/degenerate values; post-trim weights stay bounded; all SMDs pass after weighting; both propensity models are diagnosed; cleaning handles currency/duplicates/missingness without mutating the caller's frame; the economics layer assigns interval-based verdicts and reports break-evens; both guardrails fire correctly and don't false-positive; the system prompt retains its refusal and CI rules; and no analysis module reads the answer key.
