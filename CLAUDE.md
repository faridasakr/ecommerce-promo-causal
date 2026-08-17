# CLAUDE.md

Context for Claude Code working in this repository.

## What this is

A causal study of an e-commerce free-shipping promotion, built on synthetic data with a **planted causal effect** so every estimator can be scored against a known truth rather than argued about.

The headline result: a naive difference in means says the promo drove **$23.03** per customer; cross-fitted AIPW says **$6.42** (95% CI $5.98–$7.16); the planted truth is **$5.87**. A 3.6× overstatement, driven entirely by self-selection into the promo.

The project then carries that estimate through to a business decision — netting revenue against gross margin and shipping subsidy — which **flips the ranking**: clearly profitable for low-spend, marginally profitable for mid-spend, loss-making for high-spend.

## Commands

```bash
make data        # regenerate synthetic dataset (data/raw/ is gitignored)
make analysis    # full pipeline, ~4 min at --boot 200
make test        # 25 fast tests
make test-all    # + 2 slow full-data integration tests
make app         # streamlit run app/main.py
make explain     # smoke-test the LLM stakeholder layer
```

Run `make test` before and after any change. If `data/raw/customers.csv` is missing, run `make data` first.

## File map

```
src/generate_data.py   potential-outcomes DGP; writes the answer key
src/prepare.py         cleaning + design matrix
src/estimators.py      naive, OLS, PSM(ATT), IPW, AIPW(cross-fitted) + bootstrap
src/diagnostics.py     balance, overlap, dual propensity report, stress test
src/economics.py       contribution margin -> targeting decision
src/run_analysis.py    orchestrates everything, writes results/, scores vs truth
src/explain.py         LLM stakeholder layer + 3 guardrails
app/main.py            Streamlit demo (5 tabs)
tests/test_pipeline.py 27 tests
```

## Invariants — do not break these

These are the load-bearing design decisions. Each has a test guarding it; if a change makes one fail, the change is wrong, not the test.

1. **Only `run_analysis.py` reads the ground-truth answer key**, and only in the final scoring step. No estimator, diagnostic, cleaning, or economics module may reference it. Guarded by `test_ground_truth_not_read_by_analysis_modules`, which parses each module's AST.

2. **PSM targets the ATT, everything else targets the ATE**, and each is scored against its own true value. Never collapse these into one column. Guarded by `test_psm_targets_att_not_ate`.

   Scoring goes further: each estimator is compared to the mean individual effect over *the units it actually used*, derived from its own mask (`ipw_trim_mask`, `aipw_trim_mask`, `psm_matched_treated`). Trimming and matching change the estimand, so a shared full-population target would grade estimators against a population they never estimated. `tau_individual.npy` must be joined on `customer_id` — the raw CSV is shuffled, and positional alignment fails silently.

3. **The LLM never sees raw data** — only `results/summary.json`. Handed the raw table, a model reproduces the naive $23 figure, which is the exact error the project exists to prevent. Statistical reasoning stays in tested Python.

4. **Two propensity models, diagnosed separately.** IPW/PSM use logistic; AIPW uses cross-fitted gradient boosting. Balance and overlap are reported per-model so a clean table from one cannot vouch for an estimate from the other. Balance is computed on the *trimmed* population, matching what the estimators actually use.

5. **The stress test is not a formal bound.** It is not a Rosenbaum bound or E-value, and gamma is not calibrated to real-world confounding strength. Language anywhere in the repo must say "stress test," never "sensitivity analysis" in the formal sense.

6. **Causal language is hedged.** "The estimated causal effect is $X under conditional exchangeability, positivity, and SUTVA" — not "the promo caused $X." The synthetic truth is reported separately and labelled as such.

**Schema stability:** the dataset and answer key produced by `src/generate_data.py` are consumed by a separate downstream project that benchmarks against these planted effects. Treat the output schema and the generator as a published interface — changing either breaks that consumer. The isolation rules for `data/ground_truth/` are specified in `data/ground_truth/README.md`.

## Known gotchas

- **pandas dtype check:** currency-string parsing uses `pd.api.types.is_numeric_dtype`, not `== object`. pandas 3.x reads mixed text columns as the new `str` dtype and an object-identity check silently misses them.
- **`clean()` copies defensively** (`drop_duplicates().copy()`) — without it, pandas 2.x raises `SettingWithCopyWarning` on downstream assignment.
- **Nuisance learner choice matters for runtime:** `HistGradientBoostingRegressor` is ~7× faster than `GradientBoostingRegressor` here (1.1s vs 7.3s per AIPW fit) with slightly better accuracy. Bootstrapping is infeasible with the latter. Do not swap it back.
- **`ast.get_docstring(node, clean=False)`** in the ground-truth test — the default dedents, so the returned text no longer equals the raw `ast.Constant` and the docstring filter misses.
- **Numeric fidelity validator compares floats, not strings.** String comparison let `$6.40` match a true `6.42` (both render as "6.4" at 1dp). Match to the cent.
- **Bootstrap is slow.** Use `--boot 40` while iterating; `--boot 200` only for final artefacts.

## Expected results (regression reference)

If a change moves these materially, investigate before accepting:

| Estimator | Estimand | Target population | n | Estimate | True |
|---|---|---|---:|---:|---:|
| Naive | ATE | full sample | 50,000 | 23.03 | — |
| OLS | ATE | full sample | 50,000 | 7.09 | 5.867 |
| PSM | ATT | matched treated | 18,664 | 7.08 (no CI) | 5.980 |
| IPW | ATE | trimmed (logistic) | 49,879 | 6.38 | 5.869 |
| AIPW | ATE | trimmed (cross-fit) | 49,972 | 6.42 | 5.868 |

Each truth is the mean of `tau_individual.npy` over the units that estimator used, computed at scoring time. Naive has no causal target and is reported with a dash. The four targets differ only in the third decimal under this DGP — that is a property of the planted effects, not a reason to collapse them back into one column.

Balance: worst \|SMD\| 0.660 → 0.007 (logistic), 0.042 (cross-fitted GBM).

Intervals containing their target: 1 of 3 in this realization (naive has no causal target; PSM has no valid bootstrap interval). Report this as a count for this sample, never as a coverage *rate* — a rate is a property of the procedure under repeated sampling and would need a simulation study. "Under-coverage" is the wrong word and must not reappear.

Segment contribution: low +$1.29 [+1.05, +1.43], mid +$0.25 [+0.06, +0.58], high −$2.09 [−2.37, −1.39].

The subsidy multiplier is the CAUSAL incidence P(purchase | do(T=1)) per segment (0.354 / 0.454 / 0.599), not the observed treated rate (0.402 / 0.500 / 0.633). The observed rate is confounded and runs 3-5pp high; using it reintroduces selection bias on the cost side. Both are reported in economics.csv.

Contribution intervals come from a JOINT bootstrap: each replicate resamples a segment once and re-runs revenue ATE -> P(purchase | do(T=1)) -> net contribution, then percentiles are taken over the contribution draws. Never compose two separately-bootstrapped quantities.

The two estimates correlate +0.88 / +0.58 / +0.34 (low/mid/high). Because contribution is a *difference* (margin x revenue - shipping x rate), that POSITIVE correlation cancels variance rather than adding it — so the correct interval is NARROWER, and the error from a shortcut scales with the correlation:

| Segment | corr | sd revenue-only | sd if independent | sd joint |
|---|---:|---:|---:|---:|
| low | +0.88 | 0.148 (+60%) | 0.164 (+77%) | 0.092 |
| mid | +0.58 | 0.158 (+18%) | 0.167 (+25%) | 0.134 |
| high | +0.34 | 0.295 (+4%) | 0.298 (+5%) | 0.284 |

All four columns are in contribution_bootstrap.csv, so the correction is checkable rather than asserted. Do not assume adding a second uncertainty source widens an interval — for a difference with correlated terms it narrows.

All three segments now have sign_is_certain=True. Mid-spend's crossing is stable, not bootstrap noise — across seeds 0/1/2 at 50 and 150 replicates the lower bound stays in [+0.03, +0.11] and never reaches zero. But it clears break-even by cents: mid flips negative if shipping exceeds $7.06 (vs $6.50 assumed) or margin falls below 41.4% (vs 45%). Those are illustrative placeholders, so mid-spend is assumption-limited, not data-limited. Do not describe it as resolved without that caveat.
