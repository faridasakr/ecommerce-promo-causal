"""Tests that would catch a silently broken causal pipeline.

Two tiers:
  - Fast unit/property tests on an 8k-row fixture (default).
  - A full-data integration test on all 50k rows with a tighter tolerance,
    marked `slow`. Run with `pytest -m slow` or `make test-all`.

The tests that matter most are test_naive_estimator_is_biased and
test_estimators_recover_planted_effect: together they assert that the
confounding is real and that adjustment removes most of it.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import diagnostics  # noqa: E402
import economics  # noqa: E402
import estimators  # noqa: E402
import generate_data  # noqa: E402
import pandas as pd  # noqa: E402
import prepare  # noqa: E402
from explain import (  # noqa: E402
    SYSTEM_PROMPT,
    check_answer,
    validate_numeric_fidelity,
    validate_response,
)


@pytest.fixture(scope="module")
def data():
    df, truth = generate_data.generate(n=8000, seed=42)
    df = prepare.clean(generate_data.add_realistic_mess(df, seed=42), verbose=False)
    X, t, y, names = prepare.build_design_matrix(df)
    return X, t, y, names, truth


@pytest.fixture(scope="module")
def full_data():
    df, truth = generate_data.generate(seed=generate_data.SEED)
    df = prepare.clean(generate_data.add_realistic_mess(df, seed=generate_data.SEED), verbose=False)
    X, t, y, names = prepare.build_design_matrix(df)
    return X, t, y, names, truth


# ---------------------------------------------------------------- generation
def test_generation_is_deterministic():
    a, ta = generate_data.generate(n=2000, seed=1)
    b, tb = generate_data.generate(n=2000, seed=1)
    assert ta["true_ate"] == tb["true_ate"]
    assert a["revenue_promo_window"].sum() == b["revenue_promo_window"].sum()


def test_treatment_is_confounded(data):
    """Promo users must genuinely differ from non-users, or there is no problem."""
    X, t, y, names, truth = data
    smds = [abs(diagnostics.standardised_mean_diff(X[:, j], t)) for j in range(X.shape[1])]
    assert max(smds) > 0.3, "confounding too weak -- the study would be trivial"


# ---------------------------------------------------------------- estimation
def test_naive_estimator_is_biased(data):
    X, t, y, names, truth = data
    naive = estimators.naive_difference(X, t, y)
    assert naive > truth["true_ate"] * 2, "naive estimate should be badly inflated"


@pytest.mark.parametrize("name", ["IPW (stabilised)", "AIPW (cross-fitted, doubly robust)"])
def test_estimators_recover_planted_effect(data, name):
    """Loose tolerance: 8k-row fixture is noisy. Tight check lives in the slow test."""
    X, t, y, names, truth = data
    est = estimators.ESTIMATORS[name](X, t, y, seed=0)
    rel_err = abs(est - truth["true_ate"]) / truth["true_ate"]
    assert rel_err < 0.35, f"{name} off by {rel_err:.0%}"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["IPW (stabilised)", "AIPW (cross-fitted, doubly robust)"])
def test_full_data_recovery_is_tight(full_data, name):
    """On the full 50k sample, adjusted estimators should be within 15%."""
    X, t, y, names, truth = full_data
    est = estimators.ESTIMATORS[name](X, t, y, seed=0)
    rel_err = abs(est - truth["true_ate"]) / truth["true_ate"]
    assert rel_err < 0.15, f"{name} off by {rel_err:.1%} on full data (got {est:.2f})"


def test_aipw_is_deterministic_at_fixed_seed(data):
    X, t, y, names, truth = data
    a = estimators.aipw(X, t, y, seed=7)
    b = estimators.aipw(X, t, y, seed=7)
    assert a == pytest.approx(b, rel=1e-12), "AIPW must be reproducible at a fixed seed"


def test_psm_targets_att_not_ate(data):
    """PSM is scored against the ATT. Guard the labelling that makes that true."""
    assert estimators.ESTIMANDS["Propensity score matching"] == "ATT"
    assert all(
        estimators.ESTIMANDS[n] == "ATE"
        for n in estimators.ESTIMATORS
        if n != "Propensity score matching"
    )
    X, t, y, names, truth = data
    assert "true_att" in truth and truth["true_att"] != truth["true_ate"]


# --------------------------------------------------------------- diagnostics
def test_propensity_scores_are_well_formed(data):
    X, t, y, names, truth = data
    for ps in (estimators.fit_propensity(X, t), estimators.crossfit_propensity(X, t)):
        assert not np.isnan(ps).any(), "NaN propensity scores"
        assert np.isfinite(ps).all(), "non-finite propensity scores"
        assert ((ps > 0) & (ps < 1)).all(), "propensity scores at or beyond {0,1}"


def test_trimming_bounds_extreme_weights(data):
    X, t, y, names, truth = data
    ps = estimators.fit_propensity(X, t)
    keep = estimators.trim_common_support(ps, t)
    w = np.where(t[keep] == 1, 1 / ps[keep], 1 / (1 - ps[keep]))
    assert w.max() < 60, f"max post-trim weight {w.max():.1f} -- overlap is too poor"


def test_all_smds_pass_after_weighting(data):
    X, t, y, names, truth = data
    bal = diagnostics.balance_table(X, t, names, trim=True)
    failing = bal[bal["smd_weighted"].abs() >= 0.10]["covariate"].tolist()
    assert not failing, f"covariates still imbalanced after weighting: {failing}"


def test_weighting_improves_balance(data):
    X, t, y, names, truth = data
    bal = diagnostics.balance_table(X, t, names)
    assert bal["smd_weighted"].abs().max() < bal["smd_unadjusted"].abs().max()


def test_dual_report_covers_both_propensity_models(data):
    """Each estimator must be diagnosed against the model it actually uses."""
    X, t, y, names, truth = data
    rep = diagnostics.dual_propensity_report(X, t, names)
    assert set(rep) == {"logistic", "crossfit_gbm"}
    assert "AIPW (cross-fitted, doubly robust)" in rep["crossfit_gbm"]["used_by"]
    assert "IPW (stabilised)" in rep["logistic"]["used_by"]
    for info in rep.values():
        assert not info["overlap"]["has_nan"]


# ------------------------------------------------------------------ cleaning
def test_cleaning_handles_currency_strings_and_duplicates():
    df, _ = generate_data.generate(n=1500, seed=3)
    messy = generate_data.add_realistic_mess(df, seed=3)
    cleaned = prepare.clean(messy, verbose=False)
    assert cleaned["revenue_promo_window"].dtype.kind == "f"
    assert not cleaned.duplicated().any()
    assert cleaned["email_engagement"].isna().sum() == 0
    assert set(cleaned["region"].unique()) <= set(generate_data.REGIONS)


def test_clean_does_not_mutate_caller_frame():
    df, _ = generate_data.generate(n=500, seed=5)
    messy = generate_data.add_realistic_mess(df, seed=5)
    before = messy.copy()
    prepare.clean(messy, verbose=False)
    pd.testing.assert_frame_equal(messy, before)


# ----------------------------------------------------------------- economics
def test_economics_flags_loss_making_segments():
    hte = pd.DataFrame(
        {"segment": ["low spend", "high spend"], "ate": [8.0, 4.0],
         "ci_low": [7.2, 2.5], "ci_high": [8.2, 5.0]}
    )
    rates = {"low spend": 0.40, "high spend": 0.63}
    econ = economics.segment_economics(hte, rates)
    assert econ.loc[econ.segment == "low spend", "profitable"].item()
    assert not econ.loc[econ.segment == "high spend", "profitable"].item()


def test_economics_reports_breakeven_and_caveats():
    hte = pd.DataFrame({"segment": ["low spend"], "ate": [8.0], "ci_low": [7.2], "ci_high": [8.2]})
    econ = economics.segment_economics(hte, {"low spend": 0.40})
    assert econ["breakeven_shipping_cost"].iloc[0] > 0
    rec = economics.recommendation(econ)
    assert rec["assumptions"]["gross_margin"] == 0.45
    assert any("LTV" in c or "illustrative" in c.lower() for c in rec["caveats"])


# ---------------------------------------------------------------- guardrails
def test_guardrail_flags_causal_overclaim():
    bad = "The promo caused $6.42 in revenue. Paid search also drove higher spend."
    assert validate_response(bad), "should flag the paid-search causal claim"


def test_guardrail_allows_clean_response():
    good = (
        "The estimated causal effect of the promotion is $6.42 per customer "
        "(95% CI $5.98-$7.17). Acquisition channel was adjusted for as a confounder."
    )
    assert validate_response(good) == []


def test_numeric_fidelity_catches_silent_rounding():
    summary = {"headline_estimate": 6.42, "headline_ci": [5.98, 7.17]}
    assert validate_numeric_fidelity("The effect is $6.42.", summary) == []
    drifted = validate_numeric_fidelity("The effect is $6.40.", summary)
    assert drifted, "a quietly rounded figure must be flagged"


def test_numeric_fidelity_catches_invented_figures():
    summary = {"headline_estimate": 6.42}
    assert validate_numeric_fidelity("This will generate $1,200,000 next quarter.", summary)


def test_check_answer_runs_both_validators():
    summary = {"headline_estimate": 6.42}
    warns = check_answer("Paid search drove $99.99 of revenue.", summary)
    assert len(warns) >= 2, "expected both an overclaim and a fidelity warning"


def test_system_prompt_requires_refusal_and_intervals():
    """The refusal and CI rules are load-bearing; assert they survive edits."""
    low = SYSTEM_PROMPT.lower()
    assert "cannot answer" in low or "refuse" in low
    assert "confidence interval" in low
    assert "never give" in low and "point estimate" in low


# ------------------------------------------------------- ground-truth hygiene
def test_ground_truth_not_read_by_analysis_modules():
    """Only run_analysis.py may touch the answer key, and only to score.

    This is the guarantee the whole project rests on. It also matters for
    Project 2: the analytics agent must never be able to reach this file.
    """
    forbidden = ("answer_key", "ground_truth", "true_ate", "true_att")
    for module in ("estimators.py", "prepare.py", "diagnostics.py", "economics.py"):
        source = (SRC / module).read_text()
        tree = ast.parse(source)
        # Ignore docstrings/comments -- only real code references count.
        code_strings = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                # clean=False: the default dedents/strips, so the returned text
                # no longer equals the raw ast.Constant and the filter misses.
                d = ast.get_docstring(node, clean=False)
                if d:
                    docstrings.add(d)
        live = [s for s in code_strings if s not in docstrings]
        for term in forbidden:
            assert not any(term in s for s in live), (
                f"{module} references '{term}' in live code -- the answer key must "
                f"only be read by run_analysis.py at the scoring step"
            )


# ------------------------------------------------- README / results agreement
# The README quotes numbers that are regenerated by `make analysis`. Nothing
# stops the two drifting apart, and a portfolio README with stale figures is
# worse than no README. These tests parse the markdown tables and check them
# against results/ numerically -- not by string equality, because several
# cells sit near a 2dp rounding boundary (low-spend net contribution is ~6ppm
# from one) and exact matching would fail on library-version noise.

RESULTS = Path(__file__).resolve().parents[1] / "results"
README = Path(__file__).resolve().parents[1] / "README.md"

TOL = 0.015          # dollar figures: absorbs 2dp rounding, catches real drift
PCT_TOL = 1.0        # the Error column is displayed as a whole percent


def _num(cell: str) -> float:
    """Parse a README table cell into a float, stripping display decoration."""
    s = (
        cell.replace("−", "-")   # unicode minus
        .replace("**", "")
        .replace("$", "")
        .replace("%", "")
        .replace("×", "")
        .strip()
    )
    return float(s)


def _pair(cell: str) -> tuple[float, float]:
    """Parse a '[low, high]' interval cell."""
    lo, hi = cell.strip().strip("[]").split(",")
    return _num(lo), _num(hi)


def _table_rows() -> list[list[str]]:
    """Every markdown table row in the README, as stripped cell lists."""
    rows = []
    for line in README.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue  # separator row
        rows.append(cells)
    return rows


@pytest.fixture(scope="module")
def results_present():
    needed = ["economics.csv", "estimates.csv"]
    missing = [f for f in needed if not (RESULTS / f).exists()]
    if missing:
        pytest.skip(f"run `make analysis` first -- missing {missing}")


def test_readme_decision_table_matches_economics(results_present):
    """The segment contribution table must agree with results/economics.csv."""
    econ = pd.read_csv(RESULTS / "economics.csv").set_index("segment")
    seen = set()

    for cells in _table_rows():
        seg = cells[0].lower()
        if seg not in econ.index:
            continue
        seen.add(seg)
        row = econ.loc[seg]
        rev, profit, subsidy, net = (_num(c) for c in cells[1:5])
        ci_low, ci_high = _pair(cells[5])

        assert abs(rev - row["incremental_revenue"]) <= TOL, f"{seg} incremental revenue"
        assert abs(profit - row["gross_profit"]) <= TOL, f"{seg} gross profit"
        assert abs(subsidy - row["shipping_subsidy"]) <= TOL, f"{seg} shipping subsidy"
        assert abs(net - row["net_contribution"]) <= TOL, f"{seg} net contribution"
        assert abs(ci_low - row["net_contribution_low"]) <= TOL, f"{seg} CI low"
        assert abs(ci_high - row["net_contribution_high"]) <= TOL, f"{seg} CI high"

    assert seen == set(econ.index), f"README is missing segments: {set(econ.index) - seen}"


def test_readme_estimator_table_matches_estimates(results_present):
    """The estimator table must agree with results/estimates.csv."""
    est = pd.read_csv(RESULTS / "estimates.csv").set_index("estimator")
    seen = set()

    for cells in _table_rows():
        name = cells[0].replace("**", "").strip()
        if name not in est.index:
            continue
        seen.add(name)
        row = est.loc[name]
        assert cells[1].replace("**", "").strip() == row["target_estimand"], (
            f"{name}: README estimand disagrees with the declared target -- "
            f"PSM targets the ATT and must not be collapsed into the ATE"
        )
        assert abs(_num(cells[2]) - row["estimate"]) <= TOL, f"{name} estimate"
        ci_low, ci_high = _pair(cells[3])
        assert abs(ci_low - row["ci_low"]) <= TOL, f"{name} CI low"
        assert abs(ci_high - row["ci_high"]) <= TOL, f"{name} CI high"
        assert abs(_num(cells[4]) - row["true_value"]) <= TOL, f"{name} true value"
        assert abs(_num(cells[5]) - row["abs_pct_error"]) <= PCT_TOL, f"{name} error pct"

    assert seen == set(est.index), f"README is missing estimators: {set(est.index) - seen}"
