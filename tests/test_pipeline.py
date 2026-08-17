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


def test_psm_reports_no_bootstrap_ci(data):
    """PSM must ship without an interval.

    Abadie & Imbens (2008) show the ordinary nonparametric bootstrap is
    inconsistent for nearest-neighbour matching with a fixed number of matches:
    the estimator is not smooth enough, and the failure does not go away with
    more replicates. An interval here would look like quantified uncertainty
    while being unfounded -- worse than reporting none.
    """
    X, t, y, names, truth = data
    results = {r.name: r for r in estimators.run_all(X, t, y, n_boot=25, seed=0)}

    psm = results["Propensity score matching"]
    assert not np.isfinite(psm.ci_low) and not np.isfinite(psm.ci_high), (
        "PSM must not report a bootstrap CI -- the ordinary bootstrap is "
        "invalid for fixed-NN matching (Abadie & Imbens 2008)"
    )
    assert "Abadie" in psm.ci_note, "the reason for the missing CI must travel with it"
    assert np.isfinite(psm.ate), "the point estimate should still be reported"

    # The other side of the guard: this must fail if bootstrapping breaks
    # everywhere, not just for PSM.
    for name, r in results.items():
        if name == "Propensity score matching":
            continue
        assert np.isfinite(r.ci_low) and np.isfinite(r.ci_high), (
            f"{name} lost its bootstrap CI -- only PSM should be missing one"
        )
        assert r.ci_note == "", f"{name} should carry no CI caveat"


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


def test_subsidy_uses_causal_incidence_not_observed_rate():
    """The subsidy multiplier must be P(purchase | do(T=1)), not the raw rate.

    The observed purchase rate among treated customers is confounded -- treated
    customers self-selected, and the traits driving uptake also drive
    purchasing. Using it would reintroduce, on the cost side, exactly the
    selection bias the revenue side is corrected for. This test pins the
    arithmetic to the causal rate by making the two differ sharply.
    """
    hte = pd.DataFrame({"segment": ["low spend"], "ate": [8.0],
                        "ci_low": [7.2], "ci_high": [8.2]})
    causal = {"low spend": 0.30}
    observed = {"low spend": 0.60}  # confounded, deliberately far from causal

    econ = economics.segment_economics(
        hte, causal, economics.CostAssumptions(), observed_rates=observed
    )
    row = econ.iloc[0]

    ship = economics.CostAssumptions().shipping_cost_per_order
    assert row["shipping_subsidy"] == pytest.approx(-ship * 0.30, abs=0.01), (
        "subsidy must be computed from the causal incidence (0.30), not the "
        "observed treated rate (0.60)"
    )
    assert row["shipping_subsidy"] != pytest.approx(-ship * 0.60, abs=0.01)

    # net = margin * ate - causal subsidy
    assert row["net_contribution"] == pytest.approx(
        0.45 * 8.0 - ship * 0.30, abs=0.01
    )

    # Both rates must be visible so the correction is auditable.
    assert row["causal_purchase_rate"] == pytest.approx(0.30)
    assert row["observed_treated_rate"] == pytest.approx(0.60)
    assert row["rate_confounding_gap"] == pytest.approx(0.30, abs=0.001)


def test_causal_incidence_recovers_planted_lift(data):
    """AIPW on the purchase indicator should recover the planted incidence lift.

    INCIDENCE_LIFT is planted at 5.5pp with heterogeneity by prior spend, so the
    pooled causal lift should land near it -- and well below the naive
    treated-vs-control gap, which is inflated by self-selection.
    """
    X, t, y, names, truth = data
    purchased = (y > 0).astype(int)

    p0, p1 = estimators.aipw_arm_means(X, t, purchased, seed=0)
    causal_lift = p1 - p0
    naive_lift = purchased[t == 1].mean() - purchased[t == 0].mean()

    assert 0.0 < causal_lift < 0.15, f"implausible causal incidence lift {causal_lift}"
    assert causal_lift < naive_lift, (
        "the causal incidence lift must be smaller than the naive gap -- "
        "self-selection inflates the observed difference"
    )
    assert 0.0 < p1 < 1.0 and 0.0 < p0 < 1.0, "probabilities must stay in (0, 1)"


def test_contribution_ci_comes_from_joint_bootstrap_when_supplied():
    """A supplied joint-bootstrap interval must be used verbatim.

    The fallback path propagates only the revenue CI while holding the purchase
    rate fixed, which understates uncertainty now that the rate is estimated
    too. When a joint interval is available it must win, and `ci_source` must
    say which path produced the number.
    """
    hte = pd.DataFrame({"segment": ["low spend"], "ate": [8.0],
                        "ci_low": [7.2], "ci_high": [8.2]})
    rates = {"low spend": 0.30}

    fallback = economics.segment_economics(hte, rates)
    assert "revenue CI only" in fallback["ci_source"].iloc[0]
    # Revenue-only propagation: margin * ate_ci - fixed subsidy.
    ship = economics.CostAssumptions().shipping_cost_per_order
    assert fallback["net_contribution_low"].iloc[0] == pytest.approx(
        7.2 * 0.45 - ship * 0.30, abs=0.01
    )

    joint = economics.segment_economics(
        hte, rates, contribution_ci={"low spend": (-0.5, 2.5)}
    )
    assert joint["net_contribution_low"].iloc[0] == pytest.approx(-0.5)
    assert joint["net_contribution_high"].iloc[0] == pytest.approx(2.5)
    assert "joint bootstrap" in joint["ci_source"].iloc[0]
    # Straddles zero, so the sign must be reported as unresolved.
    assert not bool(joint["sign_is_certain"].iloc[0])


def test_joint_bootstrap_reflects_both_sources_of_uncertainty(data):
    """The contribution interval must respond to BOTH estimated quantities.

    Note the direction is deliberately not asserted. Because
    net = margin*ate - ship*rate, a positive correlation between the two
    estimates CANCELS variance rather than adding it, so the correct joint
    interval can be narrower than the revenue-only one. (It is here: the two
    correlate around +0.5 across replicates.) What must hold is that the rate
    genuinely varies per replicate -- if it were pinned at its point estimate,
    the joint width would exactly equal the revenue-only width, since a
    constant subsidy shifts an interval without resizing it.
    """
    X, t, y, names, truth = data
    purchased = (y > 0).astype(int)
    segment = np.zeros(len(y), dtype=int)  # single segment: whole fixture
    a = economics.CostAssumptions()

    boot = diagnostics.segment_contribution_bootstrap(
        X, t, y, purchased, segment, ["all"],
        gross_margin=a.gross_margin,
        shipping_cost_per_order=a.shipping_cost_per_order,
        n_boot=25, seed=0,
    )
    row = boot.iloc[0]
    assert row["n_draws"] >= 20, "too few usable draws to form an interval"

    joint_width = row["net_contribution_high"] - row["net_contribution_low"]
    # Revenue-only width is just the ATE interval scaled by margin: a fixed
    # subsidy shifts the interval without changing its width.
    revenue_only_width = (row["ate_ci_high"] - row["ate_ci_low"]) * a.gross_margin

    assert joint_width > 0
    assert abs(joint_width - revenue_only_width) > 1e-6, (
        "joint width equals the revenue-only width, which means the purchase "
        "rate was held fixed across replicates instead of re-estimated"
    )
    # The rate must be a genuine interval inside (0, 1), not a pinned point.
    assert 0.0 < row["causal_rate_ci_low"] < row["causal_rate_ci_high"] < 1.0

    # The covariance that makes a joint bootstrap necessary must be recorded.
    assert np.isfinite(row["corr_ate_rate"])
    assert -1.0 <= row["corr_ate_rate"] <= 1.0


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
RATE_TOL = 0.002     # purchase rates are quoted to 3dp; keep this tight


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


def _is_dash(cell: str) -> bool:
    """True if the cell is an em/en dash or hyphen placeholder, not a number."""
    return cell.replace("**", "").strip() in {"—", "–", "-", "--", "n/a"}


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
        row = econ.loc[seg]

        # Two tables key off segment names: the 7-column decision table and the
        # 4-column observed-vs-causal rate table. Dispatch on shape so adding a
        # segment-keyed table never silently unpacks into the wrong columns.
        if len(cells) == 4:
            obs, causal, gap = (_num(c) for c in cells[1:4])
            assert abs(obs - row["observed_treated_rate"]) <= RATE_TOL, f"{seg} observed rate"
            assert abs(causal - row["causal_purchase_rate"]) <= RATE_TOL, f"{seg} causal rate"
            # README states the gap as causal - observed; the CSV as observed -
            # causal. Check the README is self-consistent AND matches in size.
            assert abs(gap - (causal - obs)) <= RATE_TOL, f"{seg} gap sign/self-consistency"
            assert abs(abs(gap) - abs(row["rate_confounding_gap"])) <= RATE_TOL, f"{seg} gap size"
            continue

        assert len(cells) == 7, (
            f"unexpected segment-keyed table row with {len(cells)} cells: {cells}"
        )
        seen.add(seg)
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
        assert cells[2].replace("**", "").strip() == row["target_population"], (
            f"{name}: README target population disagrees with the population the "
            f"estimator actually used -- trimming and matching change the estimand"
        )
        assert abs(_num(cells[3]) - row["estimate"]) <= TOL, f"{name} estimate"

        # An estimator with no valid interval must show a dash, not a range.
        if pd.isna(row["ci_low"]):
            assert _is_dash(cells[4]), (
                f"{name} has no CI in estimates.csv, so the README must show a "
                f"dash, not {cells[4]!r}"
            )
        else:
            assert not _is_dash(cells[4]), (
                f"{name} has a CI in estimates.csv but the README shows a dash"
            )
            ci_low, ci_high = _pair(cells[4])
            assert abs(ci_low - row["ci_low"]) <= TOL, f"{name} CI low"
            assert abs(ci_high - row["ci_high"]) <= TOL, f"{name} CI high"

        # An estimator with no causal target must be shown with a dash, not a
        # number. Printing a "true value" for the naive contrast would imply it
        # was aiming at the ATE and missing.
        if pd.isna(row["true_value"]):
            assert _is_dash(cells[5]), (
                f"{name} has no causal target in estimates.csv, so the README "
                f"must show a dash for its true value, not {cells[5]!r}"
            )
            assert _is_dash(cells[6]), (
                f"{name} has no causal target, so its error cell must be a dash"
            )
        else:
            assert not _is_dash(cells[5]), (
                f"{name} has a true value of {row['true_value']} in estimates.csv "
                f"but the README shows a dash"
            )
            assert abs(_num(cells[5]) - row["true_value"]) <= TOL, f"{name} true value"
            assert abs(_num(cells[6]) - row["abs_pct_error"]) <= PCT_TOL, f"{name} error pct"

    assert seen == set(est.index), f"README is missing estimators: {set(est.index) - seen}"

    # A withheld interval must be explained where a reader will see it. The
    # citation is the load-bearing part -- "no CI" without a reason reads as an
    # omission rather than a deliberate methodological choice.
    prose = README.read_text()
    if (est["ci_note"].fillna("") != "").any():
        assert "Abadie" in prose, (
            "estimates.csv withholds a CI with a stated reason, but the README "
            "never explains it -- a missing interval needs its justification "
            "next to the table, not only in the CSV"
        )
