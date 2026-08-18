"""End-to-end analysis pipeline. Writes every artefact the write-up needs.

    python src/run_analysis.py [--boot 200]

Outputs (all to results/):
    estimates.csv        point estimates + CIs + target estimand + bias vs truth
    balance.csv          covariate balance, on the trimmed population
    propensity.json      diagnostics for BOTH propensity models
    heterogeneity.csv    effect by prior-spend tercile, with CIs
    economics.csv        contribution analysis -> the targeting decision
    stress_test.csv      estimate under simulated unmeasured confounding
    stakeholder_summary.json  everything the LLM layer is allowed to see -- NO ground truth
    evaluation_summary.json   ground truth + per-estimator bias; never shown to the LLM

The ground-truth answer key is read ONLY in step 7. Nothing upstream touches it,
and `tests/test_pipeline.py::test_ground_truth_not_read_by_estimator_modules`
enforces that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import diagnostics
import economics
import estimators
import prepare

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEGMENTS = ["low spend", "mid spend", "high spend"]

# Below this, per-segment bootstraps run at a quarter count so iteration stays
# fast. At or above it, they run at the full count.
FULL_SEGMENT_BOOT_THRESHOLD = 100


def segment_boot(n_boot: int) -> int:
    """Replicate count for the per-segment bootstraps.

    Segment intervals carry the targeting recommendation, and the verdict rule
    reads their *lower tail* specifically -- the 2.5th percentile is where a
    segment flips between "evidence supports targeting" and "economically
    uncertain". A tail estimated from 50 draws is essentially the first or
    second order statistic, which is far noisier than the pooled estimate it
    sits beside. For final artefacts they get the full count.

    Below the threshold the quarter count is kept, because during development
    the point estimates matter and the tails do not.
    """
    return n_boot if n_boot >= FULL_SEGMENT_BOOT_THRESHOLD else max(20, n_boot // 4)


def main(n_boot: int = 200, seed: int = 0) -> None:
    RESULTS.mkdir(exist_ok=True)
    print("=" * 72)
    print("FREE-SHIPPING PROMOTION — CAUSAL ANALYSIS")
    print("=" * 72)

    # ---------------------------------------------------------------- 1. data
    print("\n[1/7] Loading and cleaning")
    df = prepare.load()
    X, t, y, names = prepare.build_design_matrix(df)
    print(f"  {len(df):,} customers | {t.mean():.1%} used the promo")
    print(f"  mean revenue: treated ${y[t==1].mean():.2f} vs control ${y[t==0].mean():.2f}")

    # -------------------------------------------------------- 2. diagnostics
    print("\n[2/7] Propensity diagnostics (both models)")
    prop = diagnostics.dual_propensity_report(X, t, names, seed=seed)
    with open(RESULTS / "propensity.json", "w") as f:
        json.dump(prop, f, indent=2)

    for model, info in prop.items():
        ov = info["overlap"]
        print(f"  {model:<14} worst |SMD| after weighting: "
              f"{info['worst_abs_smd_after_weighting']:.4f} "
              f"({'PASS' if info['worst_abs_smd_after_weighting'] < 0.10 else 'FAIL'})")
        print(f"  {'':<14} trimmed {ov['n_trimmed']} ({ov['pct_trimmed']:.2f}%) | "
              f"max weight {ov['max_ipw_weight_untrimmed']:.1f} -> "
              f"{ov['max_ipw_weight_trimmed']:.1f} after trim")

    # Balance table for the logistic model, on the trimmed population.
    bal = diagnostics.balance_table(X, t, names, trim=True, seed=seed)
    bal.to_csv(RESULTS / "balance.csv", index=False)

    # ---------------------------------------------------------- 3. estimation
    print(f"\n[3/7] Estimating causal effect (bootstrap n={n_boot})")
    results = estimators.run_all(X, t, y, n_boot=n_boot, seed=seed)
    for r in results:
        print(f"  {r}")

    # ------------------------------------------------------- 4. heterogeneity
    print("\n[4/7] Heterogeneous effects by prior-spend tercile")
    spend = df["prior_12w_spend"].to_numpy()
    cuts = np.quantile(spend, [1 / 3, 2 / 3])
    segment = np.digitize(spend, cuts)
    hte = diagnostics.heterogeneous_effects(
        X, t, y, segment, SEGMENTS, seed=seed, n_boot=segment_boot(n_boot)
    )
    hte.to_csv(RESULTS / "heterogeneity.csv", index=False)
    for _, row in hte.iterrows():
        ci = (f" [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
              if "ci_low" in row and np.isfinite(row["ci_low"]) else "")
        print(f"  {row['segment']:<12} n={row['n']:>6,}  ATE=${row['ate']:.2f}{ci}")
    print("  (prior spend is a PRESPECIFIED effect modifier — recovering it")
    print("   tests the pipeline; it is not an organic discovery)")

    # ------------------------------------------- 4b. negative control: region
    # Region is drawn independently in the DGP: it enters neither the treatment
    # model nor the outcome model, so there is no effect modification to find.
    # Running the identical segment analysis on it asks whether the pipeline
    # MANUFACTURES segment differences. A spread here comparable to the spend
    # terciles would mean the spend result cannot be trusted either.
    print("\n[4b/7] Negative control — same analysis by region")
    region_labels = sorted(df["region"].unique())
    region_idx = {r: k for k, r in enumerate(region_labels)}
    region_seg = df["region"].map(region_idx).to_numpy()
    hte_region = diagnostics.heterogeneous_effects(
        X, t, y, region_seg, region_labels, seed=seed, n_boot=segment_boot(n_boot)
    )
    hte_region.to_csv(RESULTS / "heterogeneity_region.csv", index=False)
    for _, row in hte_region.iterrows():
        ci = (f" [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
              if "ci_low" in row and np.isfinite(row["ci_low"]) else "")
        print(f"  {row['segment']:<12} n={row['n']:>6,}  ATE=${row['ate']:.2f}{ci}")
    spend_spread = float(hte["ate"].max() - hte["ate"].min())
    region_spread = float(hte_region["ate"].max() - hte_region["ate"].min())
    print(f"  spread: ${region_spread:.2f} across regions vs "
          f"${spend_spread:.2f} across spend terciles")

    # --------------------------------------------------------- 5. economics
    print("\n[5/7] Contribution analysis — turning the estimate into a decision")
    assumptions = economics.CostAssumptions()

    # The subsidy multiplier must be causal. The observed purchase rate among
    # treated customers is confounded by the same self-selection the revenue
    # estimate corrects for, so using it would put the bias back on the cost
    # side of the decision.
    purchased = (y > 0).astype(int)
    incidence = diagnostics.causal_incidence_by_segment(
        X, t, purchased, segment, SEGMENTS, seed=seed
    )
    incidence.to_csv(RESULTS / "incidence.csv", index=False)

    causal_rates = dict(zip(incidence["segment"], incidence["causal_rate_treated"]))
    observed_rates = economics.purchase_rates_by_segment(
        df, segment, SEGMENTS, prepare.TREATMENT, prepare.OUTCOME
    )
    for _, r in incidence.iterrows():
        print(f"  {r['segment']:<12} purchase rate: observed(treated) "
              f"{r['observed_treated_rate']:.3f} vs causal do(T=1) "
              f"{r['causal_rate_treated']:.3f} "
              f"(lift {r['incidence_lift']:+.3f})")

    # Joint bootstrap: each replicate re-runs the whole chain (revenue ATE ->
    # causal purchase probability -> net contribution) on one resample, and the
    # interval is the percentiles of the contribution draws. Two separately
    # bootstrapped quantities cannot be composed: the contribution depends on
    # both and they are correlated. Scales with --boot; this is the slow step.
    n_boot_contrib = segment_boot(n_boot)
    print(f"  joint bootstrap of the contribution chain (n={n_boot_contrib}) …")
    contrib_boot, net_draws = diagnostics.segment_contribution_bootstrap(
        X, t, y, purchased, segment, SEGMENTS,
        gross_margin=assumptions.gross_margin,
        shipping_cost_per_order=assumptions.shipping_cost_per_order,
        n_boot=n_boot_contrib, seed=seed,
    )
    contrib_boot.to_csv(RESULTS / "contribution_bootstrap.csv", index=False)
    contribution_ci = {
        r["segment"]: (r["net_contribution_low"], r["net_contribution_high"])
        for _, r in contrib_boot.iterrows()
    }

    econ = economics.segment_economics(
        hte, causal_rates, assumptions,
        observed_rates=observed_rates,
        contribution_ci=contribution_ci,
    )
    econ.to_csv(RESULTS / "economics.csv", index=False)
    print(f"  assuming {assumptions.gross_margin:.0%} gross margin, "
          f"${assumptions.shipping_cost_per_order:.2f}/order shipping cost")
    for _, r in econ.iterrows():
        ci = (f"[{r['net_contribution_low']:>6.2f}, {r['net_contribution_high']:>6.2f}]"
              if np.isfinite(r.get("net_contribution_low", np.nan)) else "")
        print(f"  {r['segment']:<12} rev ${r['incremental_revenue']:>5.2f} "
              f"-> profit ${r['gross_profit']:>5.2f} "
              f"- subsidy ${abs(r['shipping_subsidy']):>5.2f} "
              f"= ${r['net_contribution']:>6.2f} {ci}")
        # The verdict is emitted verbatim -- never paraphrased per surface.
        print(f"  {'':<12} -> {r['verdict']}")
    rec = economics.recommendation(econ, assumptions)

    # Policy value on identical denominators for both policies, with the
    # segment set held fixed at the final rule.
    policy = economics.policy_economics(
        econ,
        segment_counts=dict(zip(hte["segment"], hte["n"])),
        policy_segments=rec["segments_positive_economics"],
        net_draws=net_draws,
    )
    b, tg = policy["blanket_offer"], policy["targeted_offer"]
    print(f"\n  Policy value over {policy['n_eligible_customers']:,} eligible customers:")
    for name, blk in (("blanket", b), ("targeted", tg)):
        ci = blk.get("total_contribution_ci")
        ci_s = f" [{ci[0]:,.0f}, {ci[1]:,.0f}]" if ci else ""
        print(f"    {name:<9} total ${blk['total_contribution']:>12,.0f}{ci_s}"
              f"   per eligible ${blk['per_eligible_customer']:>6.3f}"
              f"   per targeted ${blk['per_targeted_customer']:>6.3f}"
              f"   (n={blk['n_targeted']:,})")
    print(f"\n  >>> {rec['decision']}")

    # --------------------------------------------------------- 6. stress test
    print("\n[6/7] Unmeasured-confounding stress test (NOT a formal bound)")
    stress = diagnostics.unmeasured_confounding_stress_test(X, t, y, seed=seed)
    stress.to_csv(RESULTS / "stress_test.csv", index=False)
    for _, row in stress.iterrows():
        print(f"  gamma={row['gamma']:.1f}  estimate=${row['ate_under_confounding']:.2f}")

    # -------------------------------------------- 7. score against ground truth
    print("\n[7/7] Scoring against held-out ground truth")
    truth_dir = ROOT / "data" / "ground_truth"
    truth = json.load(open(truth_dir / "answer_key.json"))

    # Individual effects, aligned by customer_id. add_realistic_mess() shuffles
    # and duplicates rows before the CSV is written, so positional alignment
    # would silently score every estimator against the wrong customers.
    tau_by_generation_index = np.load(truth_dir / "tau_individual.npy")
    tau = tau_by_generation_index[df["customer_id"].str[1:].astype(int).to_numpy()]

    # The population each estimator actually used. Trimming and matching change
    # the estimand, so each is scored over its own units rather than against one
    # full-population number.
    full = np.ones(len(t), dtype=bool)
    masks = {
        "OLS adjusted treatment coefficient": full,
        "IPW (stabilised)": estimators.ipw_trim_mask(X, t, seed=seed),
        "AIPW (cross-fitted, doubly robust)": estimators.aipw_trim_mask(X, t, seed=seed),
        "Propensity score matching": estimators.psm_matched_treated(X, t, y, seed=seed),
    }
    populations = {
        "Naive difference in means": "full sample",
        "OLS adjusted treatment coefficient": "full sample",
        "Propensity score matching": "matched treated",
        "IPW (stabilised)": "trimmed (logistic)",
        "AIPW (cross-fitted, doubly robust)": "trimmed (cross-fit)",
    }

    rows = []
    for r in results:
        # The naive difference in means is a descriptive contrast between two
        # self-selected groups. It does not target a causal quantity, so there
        # is no true value to score it against -- reporting one would imply it
        # was trying and failing to estimate the ATE, rather than estimating
        # something else entirely.
        if r.name not in masks:
            target = bias = pct = None
            covers = None
        else:
            m = masks[r.name]
            target = float(tau[m].mean())
            bias = r.ate - target
            pct = abs(bias) / abs(target) * 100
            covers = (
                bool(r.ci_low <= target <= r.ci_high)
                if np.isfinite(r.ci_low)
                else None
            )
        rows.append(
            {
                "estimator": r.name,
                "target_estimand": r.estimand,
                "target_population": populations[r.name],
                "n_target": int(masks[r.name].sum()) if r.name in masks else len(t),
                "estimate": round(r.ate, 3),
                "ci_low": round(r.ci_low, 3),
                "ci_high": round(r.ci_high, 3),
                "ci_note": r.ci_note,
                "true_value": None if target is None else round(target, 3),
                "bias": None if bias is None else round(bias, 3),
                "abs_pct_error": None if pct is None else round(pct, 1),
                "true_value_role": (
                    None if target is None
                    else "formal target" if r.name in estimators.FORMAL_TARGET
                    else "reference benchmark (not a guaranteed target)"
                ),
                "ci_covers_truth": covers,
            }
        )
    est_df = pd.DataFrame(rows)
    est_df.to_csv(RESULTS / "estimates.csv", index=False)

    display = est_df.drop(columns=["ci_note"]).copy()
    for col in ("ci_low", "ci_high", "true_value", "bias", "abs_pct_error",
                "ci_covers_truth"):
        # pd.isna, not `is None`: pandas coerces None to NaN in numeric columns.
        display[col] = display[col].map(lambda v: "—" if pd.isna(v) else v)
    print(display.to_string(index=False))
    for note in est_df.loc[est_df["ci_note"] != "", "ci_note"].unique():
        print(f"  note: {note}")

    headline = est_df[est_df["estimator"].str.startswith("AIPW")].iloc[0]
    naive = est_df.iloc[0]

    summary = {
        "question": (
            "Did the 4-week free-shipping promotion cause incremental revenue, "
            "or would those customers have purchased anyway?"
        ),
        "n_customers": int(len(df)),
        "treated_share": float(t.mean()),
        "naive_estimate": float(naive["estimate"]),
        "headline_estimate": float(headline["estimate"]),
        "headline_ci": [float(headline["ci_low"]), float(headline["ci_high"])],
        "headline_method": "Cross-fitted AIPW (doubly robust)",
        "headline_estimand": "ATE",
        "selection_bias_dollars": float(naive["estimate"] - headline["estimate"]),
        "overstatement_factor": round(float(naive["estimate"] / headline["estimate"]), 2),
        "identifying_assumptions": [
            "Conditional exchangeability (no unmeasured confounding) given the "
            "observed covariates.",
            "Positivity: every customer had a non-zero probability of both using "
            "and not using the promo (checked via overlap; trimmed where violated).",
            "SUTVA: one customer's promo use does not affect another's revenue.",
            "The treatment is promo UPTAKE, not promo OFFER, so this estimates "
            "the effect among customers who would take the offer -- not the "
            "intention-to-treat effect of offering it, which is the actual "
            "policy question and is generally smaller.",
        ],
        "limitations": [
            "Bootstrap intervals quantify sampling variability only, not model "
            "misspecification.",
            "The unmeasured-confounding check is a simulation stress test, not "
            "a formal bound such as a Rosenbaum bound or E-value.",
            "Margin and shipping cost are illustrative placeholders, not "
            "finance-sourced figures; break-even columns show how far they can "
            "move before the recommendation flips.",
            "Static contribution analysis: acquisition, retention and long-run "
            "LTV effects are not modelled and would need a longitudinal design.",
            "The estimand is the effect of promo uptake, not of being offered "
            "the promo; identifying the offer effect would need randomised "
            "encouragement or an instrument for uptake.",
        ],
        "estimand_note": (
            "The five estimators do not share an estimand. Propensity score "
            "matching targets the ATT -- the effect among customers who "
            "actually used the promo. IPW and AIPW target the ATE, the "
            "effect if the promo were extended to everyone. The naive "
            "difference in means is a descriptive contrast between "
            "self-selected groups and targets no causal quantity. Trimming "
            "and matching also change which customers an "
            "estimate refers to, which is recorded per estimator in "
            "target_population. These are different questions and their "
            "answers should not be compared as if they were the same number. "
            "The OLS figure is the adjusted treatment coefficient from a "
            "model with no treatment-covariate interactions; because the "
            "effect is heterogeneous, that coefficient is not guaranteed to "
            "equal the population ATE."
        ),
        "propensity_models": {
            "logistic": {
                "used_by": prop["logistic"]["used_by"],
                "worst_abs_smd_after_weighting": prop["logistic"]["worst_abs_smd_after_weighting"],
                "pct_trimmed": prop["logistic"]["overlap"]["pct_trimmed"],
            },
            "crossfit_gbm": {
                "used_by": prop["crossfit_gbm"]["used_by"],
                "worst_abs_smd_after_weighting": prop["crossfit_gbm"]["worst_abs_smd_after_weighting"],
                "pct_trimmed": prop["crossfit_gbm"]["overlap"]["pct_trimmed"],
            },
        },
        "heterogeneity": hte.replace({np.nan: None}).to_dict(orient="records"),
        "heterogeneity_note": (
            "Prior spend is a PRESPECIFIED effect modifier: the data-generating "
            "process plants stronger effects for lower-spend customers, so "
            "recovering that pattern validates the pipeline rather than "
            "discovering something about customers."
        ),
        "negative_control_region": hte_region.replace({np.nan: None}).to_dict(
            orient="records"
        ),
        "negative_control_note": (
            "Region is a negative control: it enters neither the treatment nor "
            "the outcome model, so no effect modification exists to find. The "
            "identical segment analysis is run on it to check the pipeline does "
            "not manufacture segment differences."
        ),
        "economics": econ.replace({np.nan: None}).to_dict(orient="records"),
        "policy_economics": policy,
        "recommendation": rec,
        "stress_test": stress.to_dict(orient="records"),
        "stress_test_caveat": (
            "Simulation-based stress test, not a Rosenbaum bound or E-value. "
            "gamma is not calibrated to a real-world confounding strength."
        ),
        "ci_interpretation_note": (
            "Bootstrap confidence intervals quantify sampling variability only "
            "-- how much the estimate would move if the sample were redrawn. "
            "They do NOT bound error from model misspecification, so a tight "
            "interval is not by itself evidence that an estimate is close to "
            "correct. Report intervals, never a point estimate alone."
        ),
    }

    # Only assessable where an interval exists AND has a target to contain:
    # naive has no causal target, PSM has no valid bootstrap interval. This is a
    # count for this one realization, not an estimate of a coverage RATE -- that
    # is a property of the procedure and needs many replications to measure.
    scored = est_df[
        est_df["ci_low"].notna()
        & (est_df["true_value_role"] == "formal target")
    ]
    n_covering = int(scored["ci_covers_truth"].sum())

    # ------------------------------------------------------------------
    # Two artefacts, deliberately separated.
    #
    # The ground truth must not reach the LLM. run_analysis.py is allowed to
    # read the answer key, but writing it into the file explain.py consumes
    # would leak it transitively and defeat invariant 3 -- the model could then
    # quote the planted effect, which on real data would not exist and which
    # the whole project exists to say you cannot know.
    # ------------------------------------------------------------------
    with open(RESULTS / "stakeholder_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    evaluation = {
        "what_this_is": (
            "Scoring artefact. Contains the planted ground truth and is "
            "therefore NOT visible to the LLM stakeholder layer -- see "
            "results/stakeholder_summary.json for that. Read only by the "
            "scoring step and the app's estimator-comparison tab."
        ),
        "true_ate_revealed": float(truth["true_ate"]),
        "true_att_revealed": float(truth["true_att"]),
        "estimates": est_df.replace({np.nan: None}).to_dict(orient="records"),
        "scoring_note": (
            "Each estimator is scored against the mean individual treatment "
            "effect over the units it actually used: the full sample for OLS, "
            "the trimmed sample for IPW and AIPW (which trim under different "
            "propensity models), and the matched treated units for PSM, which "
            "targets the ATT. The naive difference in means is a descriptive "
            "contrast between self-selected groups and has no causal target, "
            "so it is reported without a true value or bias."
        ),
        "intervals_containing_target": {
            "n_containing": n_covering,
            "n_assessable": int(len(scored)),
            "excluded": {
                "Naive difference in means": "no causal target",
                "Propensity score matching": "no valid bootstrap interval",
            },
            "note": (
                "A count for THIS realization, not an estimate of the "
                "procedure's coverage rate. A rate would require a simulation "
                "study over many replications. Do not call this "
                "'under-coverage' or quote it as a percentage."
            ),
        },
    }
    with open(RESULTS / "evaluation_summary.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    print("\n" + "=" * 72)
    print(f"ESTIMATED CAUSAL EFFECT: ${headline['estimate']:.2f} per customer "
          f"(95% CI ${headline['ci_low']:.2f}-${headline['ci_high']:.2f})")
    print(f"  under conditional exchangeability, positivity, and SUTVA")
    print(f"  naive analysis said ${naive['estimate']:.2f} "
          f"({summary['overstatement_factor']}x overstatement)")
    print(f"  planted truth (synthetic data): ${truth['true_ate']:.2f}")
    print(f"  In this realization: {n_covering}/{len(scored)} intervals contained "
          f"their target (not a coverage rate)")
    print(f"    excluded: naive (no causal target), OLS (reference benchmark, not a "
          f"formal target), PSM (no valid bootstrap interval)")
    print(f"\nDECISION: {rec['decision']}")
    print("=" * 72)
    print(f"\nArtefacts written to {RESULTS}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(n_boot=args.boot, seed=args.seed)
