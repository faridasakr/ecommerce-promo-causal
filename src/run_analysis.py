"""End-to-end analysis pipeline. Writes every artefact the write-up needs.

    python src/run_analysis.py [--boot 200]

Outputs (all to results/):
    estimates.csv        point estimates + CIs + target estimand + bias vs truth
    balance.csv          covariate balance, on the trimmed population
    propensity.json      diagnostics for BOTH propensity models
    heterogeneity.csv    effect by prior-spend tercile, with CIs
    economics.csv        contribution analysis -> the targeting decision
    stress_test.csv      estimate under simulated unmeasured confounding
    summary.json         everything the LLM explanation layer is allowed to see

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
        X, t, y, segment, SEGMENTS, seed=seed, n_boot=max(20, n_boot // 4)
    )
    hte.to_csv(RESULTS / "heterogeneity.csv", index=False)
    for _, row in hte.iterrows():
        ci = (f" [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
              if "ci_low" in row and np.isfinite(row["ci_low"]) else "")
        print(f"  {row['segment']:<12} n={row['n']:>6,}  ATE=${row['ate']:.2f}{ci}")

    # --------------------------------------------------------- 5. economics
    print("\n[5/7] Contribution analysis — turning the estimate into a decision")
    assumptions = economics.CostAssumptions()
    rates = economics.purchase_rates_by_segment(
        df, segment, SEGMENTS, prepare.TREATMENT, prepare.OUTCOME
    )
    econ = economics.segment_economics(hte, rates, assumptions)
    econ.to_csv(RESULTS / "economics.csv", index=False)
    print(f"  assuming {assumptions.gross_margin:.0%} gross margin, "
          f"${assumptions.shipping_cost_per_order:.2f}/order shipping cost")
    for _, r in econ.iterrows():
        flag = "PROFITABLE" if r["profitable"] else "LOSS-MAKING"
        print(f"  {r['segment']:<12} rev ${r['incremental_revenue']:>5.2f} "
              f"-> profit ${r['gross_profit']:>5.2f} "
              f"- subsidy ${abs(r['shipping_subsidy']):>5.2f} "
              f"= ${r['net_contribution']:>6.2f}  {flag}")
    rec = economics.recommendation(econ, assumptions)
    print(f"\n  >>> {rec['decision']}")

    # --------------------------------------------------------- 6. stress test
    print("\n[6/7] Unmeasured-confounding stress test (NOT a formal bound)")
    stress = diagnostics.unmeasured_confounding_stress_test(X, t, y, seed=seed)
    stress.to_csv(RESULTS / "stress_test.csv", index=False)
    for _, row in stress.iterrows():
        print(f"  gamma={row['gamma']:.1f}  estimate=${row['ate_under_confounding']:.2f}")

    # -------------------------------------------- 7. score against ground truth
    print("\n[7/7] Scoring against held-out ground truth")
    truth = json.load(open(ROOT / "data" / "ground_truth" / "answer_key.json"))
    targets = {"ATE": truth["true_ate"], "ATT": truth["true_att"]}

    rows = []
    for r in results:
        target = targets[r.estimand]
        covers = (
            bool(r.ci_low <= target <= r.ci_high) if np.isfinite(r.ci_low) else None
        )
        rows.append(
            {
                "estimator": r.name,
                "target_estimand": r.estimand,
                "estimate": round(r.ate, 3),
                "ci_low": round(r.ci_low, 3),
                "ci_high": round(r.ci_high, 3),
                "true_value": round(target, 3),
                "bias": round(r.ate - target, 3),
                "abs_pct_error": round(abs(r.ate - target) / target * 100, 1),
                "ci_covers_truth": covers,
            }
        )
    est_df = pd.DataFrame(rows)
    est_df.to_csv(RESULTS / "estimates.csv", index=False)
    print(est_df.to_string(index=False))

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
        ],
        "estimand_note": (
            "Propensity score matching targets the ATT and is scored against the "
            "true ATT; all other estimators target the ATE."
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
        "economics": econ.replace({np.nan: None}).to_dict(orient="records"),
        "recommendation": rec,
        "stress_test": stress.to_dict(orient="records"),
        "stress_test_caveat": (
            "Simulation-based stress test, not a Rosenbaum bound or E-value. "
            "gamma is not calibrated to a real-world confounding strength."
        ),
        "true_ate_revealed": float(truth["true_ate"]),
        "true_att_revealed": float(truth["true_att"]),
        "ci_coverage_note": (
            "Bootstrap CIs capture sampling variability only, not model "
            "misspecification. Some intervals exclude the true value despite "
            "small point-estimate error."
        ),
    }
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    n_covering = est_df["ci_covers_truth"].sum()
    print("\n" + "=" * 72)
    print(f"ESTIMATED CAUSAL EFFECT: ${headline['estimate']:.2f} per customer "
          f"(95% CI ${headline['ci_low']:.2f}-${headline['ci_high']:.2f})")
    print(f"  under conditional exchangeability, positivity, and SUTVA")
    print(f"  naive analysis said ${naive['estimate']:.2f} "
          f"({summary['overstatement_factor']}x overstatement)")
    print(f"  planted truth (synthetic data): ${truth['true_ate']:.2f}")
    print(f"  CI coverage: {n_covering}/{len(est_df)} intervals contain their target")
    print(f"\nDECISION: {rec['decision']}")
    print("=" * 72)
    print(f"\nArtefacts written to {RESULTS}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(n_boot=args.boot, seed=args.seed)
