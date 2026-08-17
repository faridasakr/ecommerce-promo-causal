# CLAUDE.md

Context for Claude Code working in this repository.

## What this is

A portfolio project for a senior Data Scientist job search: a causal study of an e-commerce free-shipping promotion, built on synthetic data with a **planted causal effect** so every estimator can be scored against a known truth rather than argued about.

The headline result: a naive difference in means says the promo drove **$23.03** per customer; cross-fitted AIPW says **$6.42** (95% CI $5.98–$7.16); the planted truth is **$5.87**. A 3.6× overstatement, driven entirely by self-selection into the promo.

The project then carries that estimate through to a business decision — netting revenue against gross margin and shipping subsidy — which **flips the ranking**: profitable for low-spend customers, loss-making for high-spend, statistically unresolvable for mid-spend.

Audience is a hiring manager reading the README for 60 seconds and a technical interviewer reading the code for 30 minutes. Optimise for both.

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

| Estimator | Estimand | Estimate | True |
|---|---|---:|---:|
| Naive | ATE | 23.03 | 5.87 |
| OLS | ATE | 7.09 | 5.87 |
| PSM | ATT | 7.08 | 5.98 |
| IPW | ATE | 6.38 | 5.87 |
| AIPW | ATE | 6.42 | 5.87 |

Balance: worst \|SMD\| 0.660 → 0.007 (logistic), 0.042 (cross-fitted GBM). CI coverage: 2 of 5.

Segment contribution: low +$0.98 [+0.60, +1.14], mid −$0.04 [−0.32, +0.28], high −$2.31 [−2.89, −1.72].

Low-spend net sits ~6ppm from a rounding boundary (true value $0.9750058); a ±0.01 drift across library versions is expected, not a regression.
