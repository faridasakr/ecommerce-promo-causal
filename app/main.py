"""Streamlit demo — the live URL for the portfolio.

    streamlit run app/main.py

Deliberately shows the *comparison*, not just the headline. The point a visitor
should leave with is how far the naive number is from the causal one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Did Free Shipping Work?", layout="wide")


@st.cache_data
def load_results():
    with open(ROOT / "results" / "summary.json") as f:
        summary = json.load(f)
    estimates = pd.read_csv(ROOT / "results" / "estimates.csv")
    hte = pd.read_csv(ROOT / "results" / "heterogeneity.csv")
    balance = pd.read_csv(ROOT / "results" / "balance.csv")
    sens = pd.read_csv(ROOT / "results" / "stress_test.csv")
    return summary, estimates, hte, balance, sens


summary, estimates, hte, balance, sens = load_results()

st.title("Did free shipping actually work?")
st.caption(
    "A causal study of an e-commerce promotion — where the obvious analysis "
    "overstates the effect by 3.6×."
)

c1, c2, c3 = st.columns(3)
c1.metric(
    "Naive estimate",
    f"${summary['naive_estimate']:.2f}",
    help="Difference in mean revenue between promo users and non-users. Confounded.",
)
c2.metric(
    "Estimated causal effect",
    f"${summary['headline_estimate']:.2f}",
    delta=f"-${summary['selection_bias_dollars']:.2f} vs naive",
    delta_color="inverse",
    help=summary["headline_method"],
)
c3.metric("True effect (held out)", f"${summary['true_ate_revealed']:.2f}")

st.markdown(
    f"> The **estimated causal effect** of the promotion is "
    f"**\\${summary['headline_estimate']:.2f}** of incremental revenue per customer "
    f"(95% CI \\${summary['headline_ci'][0]:.2f}–\\${summary['headline_ci'][1]:.2f}), "
    f"under conditional exchangeability, positivity, and SUTVA. "
    f"The naive comparison says \\${summary['naive_estimate']:.2f} — a "
    f"**{summary['overstatement_factor']}× overstatement** driven by selection bias."
)
st.caption(
    f"Because this is synthetic data, the planted causal truth is known: "
    f"${summary['true_ate_revealed']:.2f}. Every estimator below is scored against it. "
    f"On real data that column would not exist — which is exactly why the "
    f"diagnostics and assumptions carry the weight."
)

tab1, tab2, tab5, tab3, tab4 = st.tabs(
    ["Estimator comparison", "Who does it work for?", "Should we run it again?",
     "Diagnostics", "Ask a question"]
)

with tab1:
    st.subheader("Every estimator, scored against the known truth")
    st.dataframe(estimates, width="stretch", hide_index=True)
    st.bar_chart(estimates.set_index("estimator")["estimate"])
    n_cover = int(estimates["ci_covers_truth"].sum())
    st.info(
        f"**Note the `target_estimand` column.** Propensity score matching targets "
        f"the ATT (effect among those who used the promo); everything else targets "
        f"the ATE. Each is scored against its own true value — comparing them as if "
        f"they estimated the same quantity would be a category error.\n\n"
        f"Also: only {n_cover} of {len(estimates)} confidence intervals contain their "
        f"target. Bootstrap CIs capture sampling variability, not model "
        f"misspecification — a tight interval around a slightly biased estimate is "
        f"exactly what you should expect, and exactly what a stakeholder would "
        f"misread as precision."
    )

with tab2:
    st.subheader("Effect by prior-spend tercile")
    st.dataframe(hte, width="stretch", hide_index=True)
    st.bar_chart(hte.set_index("segment")["ate"])
    st.info(
        "Effect size alone does not justify a targeting decision — free shipping "
        "costs money to provide. See the **Should we run it again?** tab, which "
        "nets these figures against margin and shipping subsidy."
    )

with tab5:
    st.subheader("Contribution analysis — from estimate to decision")
    econ = pd.DataFrame(summary["economics"])
    rec = summary["recommendation"]
    a = rec["assumptions"]

    st.caption(
        f"Assuming a {a['gross_margin']:.0%} gross margin and "
        f"${a['shipping_cost_per_order']:.2f} shipping cost per order. "
        f"{a['source']}"
    )
    st.dataframe(
        econ[["segment", "incremental_revenue", "gross_profit", "shipping_subsidy",
              "net_contribution", "profitable", "breakeven_shipping_cost"]],
        width="stretch", hide_index=True,
    )
    st.bar_chart(econ.set_index("segment")["net_contribution"])

    st.success(f"**Decision:** {rec['decision']}")
    st.markdown(
        f"Blanket offer nets **\\${rec['net_contribution_blanket_offer']:.2f}** "
        f"per customer; targeting only the profitable segment nets "
        f"**\\${rec['net_contribution_targeted_offer']:.2f}**."
    )
    st.warning(
        "**Why revenue alone would have misled you:** the subsidy scales with "
        "how many people *order*, not how much they spend. High-spend customers "
        "order more often, so you pay shipping on many purchases that would have "
        "happened anyway — which is how a segment can show real incremental "
        "revenue and still lose money."
    )
    with st.expander("Caveats"):
        for c in rec["caveats"]:
            st.markdown(f"- {c}")

with tab3:
    st.subheader("Propensity models — each estimate diagnosed against its own")
    pm = summary["propensity_models"]
    st.dataframe(pd.DataFrame([
        {"model": k, "used_by": ", ".join(v["used_by"]),
         "worst |SMD| after weighting": round(v["worst_abs_smd_after_weighting"], 4),
         "% trimmed": round(v["pct_trimmed"], 2)}
        for k, v in pm.items()
    ]), width="stretch", hide_index=True)
    st.caption(
        "IPW/PSM use a logistic propensity model; AIPW uses cross-fitted gradient "
        "boosting. They are diagnosed separately so a clean balance table from one "
        "model cannot vouch for an estimate produced by the other."
    )

    st.subheader("Covariate balance (standardised mean differences)")
    st.dataframe(balance, width="stretch", hide_index=True)
    st.caption("|SMD| < 0.10 after weighting is the conventional pass threshold.")

    st.subheader("Unmeasured-confounding stress test")
    st.line_chart(sens.set_index("gamma")["ate_under_confounding"])
    st.caption(
        "**This is a stress test, not a formal bound.** It is not a Rosenbaum "
        "bound or an E-value, and gamma is not calibrated to any real-world "
        "confounding strength. Across the scenarios tested the estimate stayed "
        "positive — which says the result is not obviously fragile, and nothing "
        "stronger than that."
    )

with tab4:
    st.subheader("Stakeholder assistant")
    st.caption(
        "The model sees only the computed summary — never the raw data. It is "
        "instructed to refuse questions the study cannot answer."
    )
    q = st.text_input("Question", "Should we run this promotion again?")
    if st.button("Ask"):
        from explain import ask

        out = ask(q, summary=summary)
        st.write(out["answer"])
        if out["warnings"]:
            st.warning("Guardrail flags:\n\n" + "\n\n".join(f"- {w}" for w in out["warnings"]))
        else:
            st.caption("✓ No causal overclaims detected.")
