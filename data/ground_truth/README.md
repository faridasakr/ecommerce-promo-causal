# Ground truth — isolation contract

`answer_key.json` contains the true ATE, ATT, and per-segment effects planted by
`src/generate_data.py`. It exists so every estimator can be **scored** rather
than argued about.

## Rules

1. **Only `run_analysis.py` may read this file**, and only in the final scoring
   step. No estimator, diagnostic, or cleaning module may reference it.
   Enforced by `tests/test_pipeline.py::test_ground_truth_not_read_by_analysis_modules`,
   which parses each module's AST and fails on any live-code reference.

2. **It is committed to the repo on purpose.** Hiding it would make the scoring
   unverifiable to a reviewer. The guarantee comes from the enforced separation
   above, not from secrecy.

3. **It must be excluded from any agent environment.** The analytics agent built
   in the next portfolio project queries this same database and is benchmarked
   against these effects. If the agent can read this file, the benchmark is
   worthless. When mounting the dataset for that project:

   ```
   # expose:  data/raw/
   # exclude: data/ground_truth/
   ```

   The scoring harness reads the answer key out-of-band, in a process the agent
   cannot reach.
