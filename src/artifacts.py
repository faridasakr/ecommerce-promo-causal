"""The data contract between `results/` and the Streamlit app.

Kept separate from `app/main.py` on purpose: importing that module executes the
whole Streamlit script, so the contract could not otherwise be tested without
rendering. Everything the app reads BY NAME is declared here, which is what
lets a test fail when the pipeline stops producing it.

This exists because three artefact changes in a row broke the app while every
test stayed green. The tests covered the pipeline's outputs and the README's
numbers; nothing checked that the app could still find what it reads.

CACHING NOTE. `@st.cache_data` keys on the decorated function's source hash and
its arguments -- NOT on the files the function happens to open. A pipeline run
that rewrites `results/` therefore does not invalidate the cache, and a
long-lived deployment will keep serving a stale parse of the old artefacts
against newly deployed code. That is exactly how `policy_economics` came to
raise KeyError in production while working locally. `artefact_fingerprint()`
exists to be passed as a cache argument so the cache turns over when the files
do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Everything the app opens or reads by name.
# ---------------------------------------------------------------------------
JSON_FILES = ["stakeholder_summary.json", "evaluation_summary.json"]
CSV_FILES = [
    "estimates.csv",
    "heterogeneity.csv",
    "balance.csv",
    "stress_test.csv",
]

# Top-level keys the app indexes on the stakeholder artefact.
SUMMARY_KEYS = [
    "naive_estimate",
    "headline_estimate",
    "headline_ci",
    "headline_method",
    "selection_bias_dollars",
    "overstatement_factor",
    "economics",
    "policy_economics",
    "recommendation",
    "propensity_models",
    "stress_test_caveat",
]
EVALUATION_KEYS = ["true_ate_revealed"]

# Nested keys, by the path the app walks to reach them.
RECOMMENDATION_KEYS = [
    "decision",
    "decision_rule",
    "assumptions",
    "caveats",
    "segments_insufficient_evidence",
]
POLICY_KEYS = ["n_eligible_customers", "blanket_offer", "targeted_offer", "basis_note"]
POLICY_OFFER_KEYS = [
    "segments",
    "n_targeted",
    "total_contribution",
    "per_eligible_customer",
    "per_targeted_customer",
]
ASSUMPTION_KEYS = ["gross_margin", "shipping_cost_per_order", "source"]

# Columns the app selects by name.
CSV_COLUMNS = {
    "estimates.csv": ["estimator", "estimate", "true_value", "ci_low", "ci_covers_truth"],
    "heterogeneity.csv": ["segment", "ate"],
    "balance.csv": ["covariate", "smd_unadjusted", "smd_weighted"],
    "stress_test.csv": ["gamma", "ate_under_confounding"],
}
ECONOMICS_COLUMNS = [
    "segment",
    "incremental_revenue",
    "gross_profit",
    "shipping_subsidy",
    "net_contribution",
    "net_contribution_low",
    "net_contribution_high",
    "verdict",
    "breakeven_shipping_cost",
]

# The keys `explain.ask()` returns, which the assistant tab destructures.
ASK_KEYS = ["answer", "structured", "errors", "warnings", "rendered", "mode"]


def results_dir(root: Path) -> Path:
    return Path(root) / "results"


def artefact_fingerprint(root: Path) -> str:
    """Cheap content key for cache invalidation.

    Pass this into the Streamlit-cached loader. Without it the cache survives a
    pipeline run, because the cache key is the loader's source, not the data.
    """
    d = results_dir(root)
    parts = []
    for name in sorted(JSON_FILES + CSV_FILES):
        p = d / name
        parts.append(f"{name}:{p.stat().st_mtime_ns}:{p.stat().st_size}" if p.exists() else f"{name}:missing")
    return "|".join(parts)


def load_artifacts(root: Path) -> dict:
    """Load every artefact the app needs. Raises if one is missing."""
    d = results_dir(root)
    out: dict = {}
    for name in JSON_FILES:
        with open(d / name) as f:
            out[name] = json.load(f)
    for name in CSV_FILES:
        out[name] = pd.read_csv(d / name)
    return out


def check_contract(root: Path) -> list[str]:
    """Return every way the artefacts fail what the app expects.

    A list rather than an exception so a test can report all breakages at once
    instead of one per run.
    """
    d = results_dir(root)
    problems: list[str] = []

    for name in JSON_FILES + CSV_FILES:
        if not (d / name).exists():
            problems.append(f"missing artefact: results/{name}")
    if problems:
        return problems

    art = load_artifacts(root)
    summary = art["stakeholder_summary.json"]
    evaluation = art["evaluation_summary.json"]

    for k in SUMMARY_KEYS:
        if k not in summary:
            problems.append(f"stakeholder_summary.json is missing key {k!r}")
    for k in EVALUATION_KEYS:
        if k not in evaluation:
            problems.append(f"evaluation_summary.json is missing key {k!r}")

    rec = summary.get("recommendation", {})
    for k in RECOMMENDATION_KEYS:
        if k not in rec:
            problems.append(f"recommendation is missing key {k!r}")
    for k in ASSUMPTION_KEYS:
        if k not in rec.get("assumptions", {}):
            problems.append(f"recommendation.assumptions is missing key {k!r}")

    pol = summary.get("policy_economics", {})
    for k in POLICY_KEYS:
        if k not in pol:
            problems.append(f"policy_economics is missing key {k!r}")
    for offer in ("blanket_offer", "targeted_offer"):
        for k in POLICY_OFFER_KEYS:
            if k not in pol.get(offer, {}):
                problems.append(f"policy_economics.{offer} is missing key {k!r}")

    econ_rows = summary.get("economics", [])
    if not econ_rows:
        problems.append("stakeholder_summary.json economics is empty")
    else:
        for c in ECONOMICS_COLUMNS:
            if c not in econ_rows[0]:
                problems.append(f"economics rows are missing column {c!r}")

    for name, cols in CSV_COLUMNS.items():
        for c in cols:
            if c not in art[name].columns:
                problems.append(f"results/{name} is missing column {c!r}")

    return problems
