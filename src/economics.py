"""Turn the causal estimate into a targeting decision.

An incremental-revenue number is not a recommendation. Free shipping costs money
to provide, and revenue is not profit. The chain that actually answers "should we
run this again, and for whom?" is:

    incremental revenue  (the causal estimate, per segment)
      x gross margin
      - incremental shipping subsidy
    = incremental contribution per customer

The subsidy term is what makes this interesting. It scales with how many people
ORDER, not with how much they spend -- so a segment can generate real
incremental revenue and still lose money, because you paid shipping on every
order including the ones that would have happened anyway.

That last clause is the whole point. The naive analysis says the promo is
enormously profitable. The causal analysis says it is marginal overall and
negative for the customers you least want to subsidise.

All cost parameters are ASSUMPTIONS, declared here and surfaced in the output so
a stakeholder can challenge them. Break-even thresholds are reported so the
recommendation degrades gracefully if the assumptions are wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The targeting rule is INTERVAL-based, not point-estimate based.
#
# A positive point estimate whose interval spans zero is not evidence that a
# segment pays: it is evidence that the data cannot tell. Keying the decision
# off `net_contribution > 0` silently promotes such a segment to "target it",
# which is precisely the overclaim this project exists to avoid. These three
# verdicts are emitted verbatim wherever the decision is reported, so the
# README, the app and the stakeholder summary cannot drift from each other or
# soften the wording independently.
# ---------------------------------------------------------------------------
VERDICT_TARGET = "evidence supports targeting"
VERDICT_DESTROYS = "evidence suggests this segment destroys contribution"
VERDICT_UNCERTAIN = "economically uncertain; recommend a controlled test"


def verdict_from_interval(ci_low: float, ci_high: float) -> str:
    """Three-way verdict from a contribution interval.

    A missing interval yields UNCERTAIN: with no interval there is no evidence
    of a sign, which is the same practical position as an interval spanning
    zero -- do not act, run a controlled test.
    """
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return VERDICT_UNCERTAIN
    if ci_low > 0:
        return VERDICT_TARGET
    if ci_high < 0:
        return VERDICT_DESTROYS
    return VERDICT_UNCERTAIN


@dataclass
class CostAssumptions:
    """Declared, challengeable, and reported alongside every conclusion."""

    gross_margin: float = 0.45  # contribution margin on incremental revenue
    shipping_cost_per_order: float = 6.50  # what the retailer pays the carrier
    source: str = (
        "Illustrative. Replace with finance's actual blended margin and "
        "carrier rate before acting on this analysis."
    )


def segment_economics(
    hte: pd.DataFrame,
    purchase_rates: dict[str, float],
    assumptions: CostAssumptions | None = None,
    observed_rates: dict[str, float] | None = None,
    contribution_ci: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Per-segment contribution analysis.

    Parameters
    ----------
    hte : DataFrame with columns [segment, ate, ci_low, ci_high]
    purchase_rates : CAUSAL purchase incidence under treatment, P(purchase |
        do(T=1)), per segment. The subsidy is paid per order, so the multiplier
        must be the rate you would actually face if the segment were treated.

        This must NOT be the observed purchase rate among treated customers.
        That rate is confounded: treated customers self-selected, and the traits
        that drove uptake also drive purchasing, so it runs high and inflates
        the subsidy. Using it would reintroduce exactly the selection bias the
        rest of the pipeline exists to remove -- on the cost side of a decision
        the revenue side was carefully de-confounded for.
    observed_rates : the confounded rate, carried through for comparison only.
        Reported next to the causal figure so the gap is visible rather than
        asserted. Never used in the arithmetic.
    contribution_ci : {segment: (low, high)} from a JOINT bootstrap in which
        each replicate re-estimated the revenue ATE and the purchase
        probability on the same resample. Preferred, because the contribution
        depends on both and they are correlated.

        Without it, the interval falls back to propagating the revenue CI alone
        while holding the purchase rate fixed at its point estimate. That
        understates uncertainty -- the rate is estimated too. The fallback is
        kept for callers that have no bootstrap, and the `ci_source` column
        records which path produced the interval so a reader can tell.
    """
    a = assumptions or CostAssumptions()
    rows = []

    for _, r in hte.iterrows():
        seg = r["segment"]
        ate = float(r["ate"])
        rate = purchase_rates[seg]

        gross_profit = ate * a.gross_margin
        subsidy = a.shipping_cost_per_order * rate
        net = gross_profit - subsidy

        # Break-even: the shipping cost at which this segment turns negative.
        breakeven_ship = gross_profit / rate if rate > 0 else np.nan
        # Break-even: the margin needed to cover the subsidy at current cost.
        breakeven_margin = subsidy / ate if ate > 0 else np.nan

        row = {
            "segment": seg,
            "incremental_revenue": round(ate, 2),
            "gross_profit": round(gross_profit, 2),
            "shipping_subsidy": round(-subsidy, 2),
            "net_contribution": round(net, 2),
            "profitable": bool(net > 0),
            "breakeven_shipping_cost": round(breakeven_ship, 2),
            "breakeven_gross_margin": round(breakeven_margin, 3),
            # The rate actually used in the subsidy above.
            "causal_purchase_rate": round(rate, 3),
        }

        # Both rates are reported so the correction is auditable: if they are
        # close that is itself worth showing, and if they diverge the gap is the
        # finding.
        if observed_rates is not None and seg in observed_rates:
            obs = observed_rates[seg]
            row["observed_treated_rate"] = round(obs, 3)
            row["rate_confounding_gap"] = round(obs - rate, 3)
            row["subsidy_if_observed_rate"] = round(
                -a.shipping_cost_per_order * obs, 2
            )

        # The decision must carry the estimate's uncertainty rather than
        # laundering it away. Prefer the joint bootstrap: it is the only version
        # that reflects uncertainty in BOTH the revenue effect and the purchase
        # probability, and the covariance between them.
        lo = hi = None
        if contribution_ci is not None and seg in contribution_ci:
            lo, hi = contribution_ci[seg]
            source = "joint bootstrap (revenue + incidence)"
        elif "ci_low" in r and np.isfinite(r.get("ci_low", np.nan)):
            lo = float(r["ci_low"]) * a.gross_margin - subsidy
            hi = float(r["ci_high"]) * a.gross_margin - subsidy
            source = "revenue CI only (purchase rate held fixed — understates uncertainty)"

        if lo is not None and np.isfinite(lo) and np.isfinite(hi):
            row["net_contribution_low"] = round(lo, 2)
            row["net_contribution_high"] = round(hi, 2)
            row["ci_source"] = source
            row["sign_is_certain"] = bool(
                np.sign(row["net_contribution_low"]) == np.sign(row["net_contribution_high"])
            )

        # Derived from the ROUNDED bounds so the verdict always agrees with the
        # interval as published. A bound of -0.004 printing as "0.00" must not
        # be read as strictly negative by the rule but as zero by the reader.
        row["verdict"] = verdict_from_interval(
            row.get("net_contribution_low", np.nan),
            row.get("net_contribution_high", np.nan),
        )

        rows.append(row)

    return pd.DataFrame(rows)


def purchase_rates_by_segment(
    df: pd.DataFrame, segment: np.ndarray, labels: list[str], treatment_col: str, outcome_col: str
) -> dict[str, float]:
    """Share of treated customers in each segment who placed any order."""
    rates = {}
    for k, label in enumerate(labels):
        m = (segment == k) & (df[treatment_col].to_numpy() == 1)
        rates[label] = float((df.loc[m, outcome_col].to_numpy() > 0).mean())
    return rates


def recommendation(econ: pd.DataFrame, assumptions: CostAssumptions | None = None) -> dict:
    """Convert the contribution table into a stated decision with caveats.

    The rule is INTERVAL-based. A segment is recommended only when its whole
    contribution interval sits above zero -- not when its point estimate is
    positive. A positive point estimate with an interval spanning zero means
    the data cannot resolve the sign, and the honest output is a controlled
    test, not a targeting instruction.
    """
    a = assumptions or CostAssumptions()

    def by_verdict(v: str) -> list[str]:
        return econ[econ["verdict"] == v]["segment"].tolist()

    target = by_verdict(VERDICT_TARGET)
    destroys = by_verdict(VERDICT_DESTROYS)
    uncertain = by_verdict(VERDICT_UNCERTAIN)

    # `profitable` remains a point-estimate fact and is still reported, but it
    # no longer drives the decision.
    profitable = econ[econ["profitable"]]["segment"].tolist()
    unprofitable = econ[~econ["profitable"]]["segment"].tolist()

    blanket_net = float(econ["net_contribution"].mean())
    targeted_net = (
        float(econ[econ["verdict"] == VERDICT_TARGET]["net_contribution"].mean())
        if target
        else 0.0
    )

    parts = []
    if target:
        parts.append(f"Target the promotion at: {', '.join(target)}.")
    if destroys:
        parts.append(f"Withhold it from: {', '.join(destroys)}.")
    if uncertain:
        parts.append(
            f"Run a controlled test before deciding on: {', '.join(uncertain)}."
        )
    if not parts:
        parts.append("Do not re-run the promotion as structured.")

    return {
        "decision": " ".join(parts),
        "verdicts": dict(zip(econ["segment"], econ["verdict"])),
        "segments_evidence_supports_targeting": target,
        "segments_evidence_says_destroy_contribution": destroys,
        "segments_economically_uncertain": uncertain,
        "decision_rule": (
            "Interval-based: a segment is recommended only if the whole 95% "
            "contribution interval lies above zero, withheld only if it lies "
            "entirely below zero, and otherwise reported as economically "
            "uncertain. A positive point estimate whose interval spans zero is "
            "not evidence that the segment pays."
        ),
        "profitable_segments": profitable,
        "unprofitable_segments": unprofitable,
        "net_contribution_blanket_offer": round(blanket_net, 2),
        "net_contribution_targeted_offer": round(targeted_net, 2),
        "segments_where_sign_is_uncertain": uncertain,
        "assumptions": asdict(a),
        "caveats": [
            "Contribution figures inherit the causal estimate's assumptions "
            "(conditional exchangeability, positivity, SUTVA).",
            "Margin and shipping cost are illustrative placeholders, not "
            "finance-sourced figures.",
            "Break-even columns show how far the assumptions can move before "
            "the recommendation flips.",
            "This is a static contribution analysis: it ignores promo-driven "
            "acquisition, retention, and any long-run LTV effects, all of "
            "which would require a longitudinal design to estimate.",
        ],
    }
