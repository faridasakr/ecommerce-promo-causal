"""Diagnostics that decide whether the causal estimates are trustworthy.

Three checks, in the order a reviewer will want them:

  1. Covariate balance  -- standardised mean differences before/after weighting.
                           |SMD| < 0.1 is the conventional threshold.
  2. Overlap / positivity -- propensity distributions must actually intersect.
  3. Stress test          -- inject a simulated unmeasured confounder and watch
                             whether the estimate is obviously fragile.

Check 3 is deliberately NOT called a sensitivity analysis: it is a simulation,
not a formal bound (see the function docstring). Every observational estimate
rests on an untestable assumption, and the honest move is to probe it while
being precise about what the probe does and does not establish.

Note on propensity models: this project fits two. IPW and PSM use a logistic
model; AIPW uses cross-fitted gradient boosting. `balance_table` and
`overlap_summary` therefore take an explicit `ps` argument so each estimate is
diagnosed against its OWN model -- see `dual_propensity_report`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from estimators import aipw, fit_propensity, trim_common_support


def standardised_mean_diff(x: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> float:
    """SMD between treated and control, optionally weighted."""
    if w is None:
        w = np.ones_like(x, dtype=float)
    w1, w0 = w[t == 1], w[t == 0]
    x1, x0 = x[t == 1], x[t == 0]

    m1 = np.average(x1, weights=w1)
    m0 = np.average(x0, weights=w0)
    v1 = np.average((x1 - m1) ** 2, weights=w1)
    v0 = np.average((x0 - m0) ** 2, weights=w0)
    pooled = np.sqrt((v1 + v0) / 2)
    return float((m1 - m0) / pooled) if pooled > 0 else 0.0


def balance_table(
    X: np.ndarray,
    t: np.ndarray,
    names: list[str],
    ps: np.ndarray | None = None,
    trim: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Covariate balance, computed on the SAME population the estimator uses.

    `ps` defaults to the logistic propensity model (what IPW uses). Pass the
    cross-fitted scores from `estimators.crossfit_propensity` to diagnose the
    AIPW nuisance model instead.

    `trim=True` restricts to the common-support region before computing SMDs,
    matching the trimming the estimators apply. Reporting balance on the full
    sample while estimating on a trimmed one describes two different
    populations -- a subtle mismatch that is easy to miss and hard to defend.
    """
    if ps is None:
        ps = fit_propensity(X, t, seed=seed)

    if trim:
        keep = trim_common_support(ps, t)
        X, t, ps = X[keep], t[keep], ps[keep]

    p_treat = t.mean()
    w = np.where(t == 1, p_treat / ps, (1 - p_treat) / (1 - ps))

    rows = []
    for j, name in enumerate(names):
        rows.append(
            {
                "covariate": name,
                "smd_unadjusted": standardised_mean_diff(X[:, j], t),
                "smd_weighted": standardised_mean_diff(X[:, j], t, w),
            }
        )
    df = pd.DataFrame(rows)
    df["balanced_after"] = df["smd_weighted"].abs() < 0.10
    return df.sort_values("smd_unadjusted", key=np.abs, ascending=False).reset_index(drop=True)


def overlap_summary(
    X: np.ndarray, t: np.ndarray, ps: np.ndarray | None = None, seed: int = 0
) -> dict:
    """Positivity diagnostics for a given propensity model.

    Pass `ps` explicitly to characterise a specific model; otherwise the
    logistic model is used.
    """
    if ps is None:
        ps = fit_propensity(X, t, seed=seed)
    keep = trim_common_support(ps, t)
    return {
        "ps_treated_min": float(ps[t == 1].min()),
        "ps_treated_max": float(ps[t == 1].max()),
        "ps_control_min": float(ps[t == 0].min()),
        "ps_control_max": float(ps[t == 0].max()),
        "n_trimmed": int((~keep).sum()),
        "pct_trimmed": float((~keep).mean() * 100),
        "max_ipw_weight_untrimmed": float(np.max(np.where(t == 1, 1 / ps, 1 / (1 - ps)))),
        "max_ipw_weight_trimmed": float(
            np.max(np.where(t[keep] == 1, 1 / ps[keep], 1 / (1 - ps[keep])))
        ),
        "has_nan": bool(np.isnan(ps).any()),
        "has_extreme": bool(((ps <= 0) | (ps >= 1)).any()),
    }


def dual_propensity_report(
    X: np.ndarray, t: np.ndarray, names: list[str], seed: int = 0
) -> dict:
    """Diagnose BOTH propensity models side by side.

    The project uses two, and conflating them would let a clean balance table
    from one model vouch for an estimate produced by the other:

        IPW / PSM  -> logistic propensity
        AIPW       -> cross-fitted HistGradientBoosting propensity

    Both are reported so each estimate is backed by diagnostics of its own model.
    """
    from estimators import crossfit_propensity

    logistic_ps = fit_propensity(X, t, seed=seed)
    crossfit_ps = crossfit_propensity(X, t, seed=seed)

    return {
        "logistic": {
            "used_by": ["IPW (stabilised)", "Propensity score matching"],
            "overlap": overlap_summary(X, t, ps=logistic_ps),
            "worst_abs_smd_after_weighting": float(
                balance_table(X, t, names, ps=logistic_ps)["smd_weighted"].abs().max()
            ),
        },
        "crossfit_gbm": {
            "used_by": ["AIPW (cross-fitted, doubly robust)"],
            "overlap": overlap_summary(X, t, ps=crossfit_ps),
            "worst_abs_smd_after_weighting": float(
                balance_table(X, t, names, ps=crossfit_ps)["smd_weighted"].abs().max()
            ),
        },
    }


def heterogeneous_effects(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    segment: np.ndarray,
    labels: list[str],
    seed: int = 0,
    n_boot: int = 0,
) -> pd.DataFrame:
    """Re-estimate the effect within each segment using the doubly robust estimator.

    With `n_boot > 0`, attaches bootstrap CIs per segment. These matter: the
    downstream contribution analysis multiplies these estimates by a margin and
    subtracts a fixed cost, so a segment whose CI straddles the break-even point
    cannot support a targeting decision, however tidy its point estimate looks.
    """
    from estimators import bootstrap_ci

    rows = []
    for k, label in enumerate(labels):
        m = segment == k
        if m.sum() < 500 or t[m].sum() < 50:
            rows.append({"segment": label, "n": int(m.sum()), "ate": np.nan})
            continue

        row = {
            "segment": label,
            "n": int(m.sum()),
            "treated_share": float(t[m].mean()),
            "ate": aipw(X[m], t[m], y[m], seed=seed),
        }
        if n_boot > 0:
            lo, hi = bootstrap_ci(aipw, X[m], t[m], y[m], n_boot=n_boot, seed=seed)
            row["ci_low"], row["ci_high"] = lo, hi
        rows.append(row)

    return pd.DataFrame(rows)


def causal_incidence_by_segment(
    X: np.ndarray,
    t: np.ndarray,
    purchased: np.ndarray,
    segment: np.ndarray,
    labels: list[str],
    seed: int = 0,
) -> pd.DataFrame:
    """Causal purchase incidence per segment, on a binary purchase indicator.

    The economics layer needs P(purchase | do(T=1)) -- how many orders you would
    actually be subsidising if the segment were treated. The OBSERVED purchase
    rate among treated customers is not that quantity: treated customers
    self-selected, and the traits that drove uptake also drive purchasing, so
    the observed rate is confounded upward.

    Reuses the same cross-fitted AIPW machinery as the revenue estimate, applied
    to `purchased` instead of revenue. Returns E[purchase(0)], E[purchase(1)],
    the causal incidence lift, and the observed treated rate for comparison.
    """
    from estimators import aipw_arm_means

    rows = []
    for k, label in enumerate(labels):
        m = segment == k
        observed = float(purchased[m & (t == 1)].mean()) if (m & (t == 1)).any() else np.nan

        if m.sum() < 500 or t[m].sum() < 50:
            rows.append({
                "segment": label,
                "observed_treated_rate": observed,
                "causal_rate_untreated": np.nan,
                "causal_rate_treated": np.nan,
                "incidence_lift": np.nan,
            })
            continue

        p0, p1 = aipw_arm_means(X[m], t[m], purchased[m], seed=seed)
        rows.append({
            "segment": label,
            "observed_treated_rate": observed,
            "causal_rate_untreated": p0,
            "causal_rate_treated": p1,
            "incidence_lift": p1 - p0,
        })

    return pd.DataFrame(rows)


def segment_contribution_bootstrap(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    purchased: np.ndarray,
    segment: np.ndarray,
    labels: list[str],
    gross_margin: float,
    shipping_cost_per_order: float,
    n_boot: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Joint bootstrap of the whole contribution chain, per segment.

    The contribution depends on TWO estimated quantities -- the revenue ATE and
    the causal purchase probability -- and both carry sampling uncertainty.
    Composing an interval from two separately-bootstrapped quantities would be
    wrong twice over: it ignores their covariance, and it treats whichever one
    was held fixed as if it were known exactly.

    So each replicate resamples the segment ONCE and re-runs the entire chain on
    that resample: revenue ATE, P(purchase | do(T=1)), and the net contribution
    they imply. Percentiles are taken over the contribution draws directly. The
    two estimates move together across replicates exactly as they do in the
    data, because they are computed from the same resampled customers.

    Returns per-segment percentile intervals for all three quantities. The ATE
    interval comes from the same draws as the contribution interval, so the two
    are mutually consistent rather than two different resamplings of the same
    segment.
    """
    from estimators import aipw, aipw_arm_means

    rng = np.random.default_rng(seed)
    rows = []

    for k, label in enumerate(labels):
        m = segment == k
        Xs, ts, ys, ps = X[m], t[m], y[m], purchased[m]
        n = len(ys)

        ate_draws: list[float] = []
        rate_draws: list[float] = []
        net_draws: list[float] = []

        if n >= 500 and ts.sum() >= 50:
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                try:
                    # Same resample feeds both estimates -- that is the point.
                    a = aipw(Xs[idx], ts[idx], ys[idx], seed=seed)
                    _, p1 = aipw_arm_means(Xs[idx], ts[idx], ps[idx], seed=seed)
                except Exception:
                    continue
                if not (np.isfinite(a) and np.isfinite(p1)):
                    continue
                ate_draws.append(a)
                rate_draws.append(p1)
                net_draws.append(a * gross_margin - shipping_cost_per_order * p1)

        def pct(draws: list[float]) -> tuple[float, float]:
            if len(draws) < 20:
                return (np.nan, np.nan)
            return (float(np.percentile(draws, 2.5)),
                    float(np.percentile(draws, 97.5)))

        ate_lo, ate_hi = pct(ate_draws)
        rate_lo, rate_hi = pct(rate_draws)
        net_lo, net_hi = pct(net_draws)

        # The covariance is the whole reason this has to be a joint bootstrap.
        # net = margin*ate - ship*rate, so a POSITIVE correlation between the
        # two cancels variance rather than adding it: composing independent
        # bootstraps would overstate the interval. Recorded so the direction of
        # the correction is auditable rather than asserted.
        if len(net_draws) >= 20:
            a_arr = np.asarray(ate_draws) * gross_margin
            r_arr = np.asarray(rate_draws) * shipping_cost_per_order
            corr = float(np.corrcoef(ate_draws, rate_draws)[0, 1])
            sd_joint = float(np.std(net_draws))
            # What the two discarded shortcuts would have produced, so the size
            # AND direction of the correction are checkable from the artefact.
            sd_revenue_only = float(np.std(a_arr))
            sd_if_independent = float(np.sqrt(np.var(a_arr) + np.var(r_arr)))
        else:
            corr = sd_joint = sd_revenue_only = sd_if_independent = np.nan

        rows.append({
            "segment": label,
            "ate_ci_low": ate_lo,
            "ate_ci_high": ate_hi,
            "causal_rate_ci_low": rate_lo,
            "causal_rate_ci_high": rate_hi,
            "net_contribution_low": net_lo,
            "net_contribution_high": net_hi,
            "corr_ate_rate": corr,
            "sd_net_joint": sd_joint,
            "sd_net_revenue_only": sd_revenue_only,
            "sd_net_if_independent": sd_if_independent,
            "n_draws": len(net_draws),
        })

    return pd.DataFrame(rows)


def unmeasured_confounding_stress_test(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    gammas: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    seed: int = 0,
) -> pd.DataFrame:
    """Stress-test the estimate against a SIMULATED unmeasured confounder.

    WHAT THIS IS NOT: this is not a Rosenbaum bound, an E-value, or any formal
    sensitivity analysis. It does not bound the bias from unmeasured
    confounding, and `gamma` is not calibrated to an interpretable real-world
    confounding strength -- it is an arbitrary simulation knob that perturbs
    both treatment assignment and the outcome.

    WHAT IT IS: a stress test. We inject a synthetic confounder of increasing
    strength and watch how the estimate moves. It answers "is this estimate
    obviously fragile?" -- not "how wrong could I be?".

    Report it as a stress test. Claiming more than that from this procedure is
    exactly the kind of overreach the project is otherwise arguing against.

    For a formal treatment, an E-value (VanderWeele & Ding) would be the right
    next step and is listed in the README's future work.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    rows = []

    for gamma in gammas:
        u = rng.normal(size=n)
        # Confounder pushes treated units up on both treatment and outcome.
        t_shift = (u > 0).astype(int)
        # Reassign a fraction of treatment consistent with U, and shift outcome.
        flip = rng.random(n) < gamma * 0.5
        t_conf = np.where(flip, t_shift, t)
        y_conf = y + gamma * u * y.std()

        est = aipw(X, t_conf, y_conf, seed=seed)
        rows.append({"gamma": gamma, "ate_under_confounding": est})

    return pd.DataFrame(rows)
