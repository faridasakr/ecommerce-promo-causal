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

        # Propagate the causal CI through to contribution, so the decision
        # carries the estimate's uncertainty rather than laundering it away.
        if "ci_low" in r and np.isfinite(r.get("ci_low", np.nan)):
            row["net_contribution_low"] = round(
                float(r["ci_low"]) * a.gross_margin - subsidy, 2
            )
            row["net_contribution_high"] = round(
                float(r["ci_high"]) * a.gross_margin - subsidy, 2
            )
            row["sign_is_certain"] = bool(
                np.sign(row["net_contribution_low"]) == np.sign(row["net_contribution_high"])
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
    """Convert the contribution table into a stated decision with caveats."""
    a = assumptions or CostAssumptions()
    profitable = econ[econ["profitable"]]["segment"].tolist()
    unprofitable = econ[~econ["profitable"]]["segment"].tolist()

    blanket_net = float(econ["net_contribution"].mean())
    targeted_net = float(econ[econ["profitable"]]["net_contribution"].mean()) if profitable else 0.0

    uncertain = (
        econ[econ.get("sign_is_certain", True) == False]["segment"].tolist()  # noqa: E712
        if "sign_is_certain" in econ.columns
        else []
    )

    return {
        "decision": (
            f"Target the promotion at: {', '.join(profitable)}."
            if profitable
            else "Do not re-run the promotion as structured."
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
