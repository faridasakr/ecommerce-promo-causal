# Ground truth — isolation contract

`answer_key.json` contains the true ATE, ATT, and per-segment effects planted by
`src/generate_data.py`. It exists so every estimator can be **scored** rather
than argued about.

`tau_individual.npy` holds the per-customer effect `y1 - y0` for all 50,000
customers. It is what lets the scoring step compute an estimand-specific truth
for whatever sub-population an estimator actually used — trimmed, matched, or
full — instead of comparing everything to one hard-coded number.

**Alignment:** row `i` corresponds to `customer_id` `f"C{i:07d}"`.
`add_realistic_mess()` shuffles and duplicates rows before the CSV is written,
so consumers MUST join on `customer_id`. Aligning by row position produces a
number that looks plausible — the overall mean is unchanged — while every
subgroup figure is silently wrong.

Both files are covered by the rules below.

## Rules

1. **Only `run_analysis.py` may read these files**, and only in the final scoring
   step. No estimator, diagnostic, or cleaning module may reference it.
   Enforced by `tests/test_pipeline.py::test_ground_truth_not_read_by_analysis_modules`,
   which parses each module's AST and fails on any live-code reference.

2. **They are committed to the repo on purpose.** Hiding it would make the scoring
   unverifiable to a reviewer. The guarantee comes from the enforced separation
   above, not from secrecy.

3. **They must be excluded from any agent environment.** The analytics agent built
   in the next portfolio project queries this same database and is benchmarked
   against these effects. If the agent can read these files, the benchmark is
   worthless. When mounting the dataset for that project:

   ```
   # expose:  data/raw/
   # exclude: data/ground_truth/
   ```

   The scoring harness reads the answer key out-of-band, in a process the agent
   cannot reach.
