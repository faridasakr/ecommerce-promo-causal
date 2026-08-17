"""Causal estimators for the free-shipping promotion study.

Five estimators, deliberately ordered from naive to doubly robust, so the
write-up can show what each layer of adjustment buys:

  1. naive_difference   -- difference in group means. Badly confounded.
  2. ols_adjustment     -- regression control for observed confounders.
  3. psm_matching       -- 1-NN propensity score matching with caliper.
  4. ipw                -- stabilised inverse probability weighting.
  5. aipw               -- augmented IPW (doubly robust). The headline estimate.

All estimators return an Estimate with a bootstrap confidence interval so the
comparison table carries uncertainty, not just point estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors


@dataclass
class Estimate:
    name: str
    ate: float
    ci_low: float = np.nan
    ci_high: float = np.nan
    estimand: str = "ATE"  # "ATE" or "ATT" -- these are NOT the same quantity
    diagnostics: dict = field(default_factory=dict)

    def __str__(self) -> str:
        ci = (
            f"[{self.ci_low:6.2f}, {self.ci_high:6.2f}]"
            if np.isfinite(self.ci_low)
            else "        --        "
        )
        return f"{self.name:<34} {self.estimand:<4} {self.ate:7.2f}  {ci}"


# ---------------------------------------------------------------------------
# Propensity score
# ---------------------------------------------------------------------------
def fit_propensity(X: np.ndarray, t: np.ndarray, seed: int = 0) -> np.ndarray:
    model = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    model.fit(X, t)
    return model.predict_proba(X)[:, 1]


def trim_common_support(
    ps: np.ndarray, t: np.ndarray, lo: float = 0.02, hi: float = 0.98
) -> np.ndarray:
    """Boolean mask keeping units in the region of common support.

    Units with propensity near 0 or 1 have no counterfactual counterpart; keeping
    them makes IPW weights explode and silently inflates variance.
    """
    return (ps > lo) & (ps < hi)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def naive_difference(X, t, y, **kw) -> float:
    return y[t == 1].mean() - y[t == 0].mean()


def ols_adjustment(X, t, y, **kw) -> float:
    """Regress y on [t, X]; the coefficient on t is the adjusted effect."""
    design = np.column_stack([t, X])
    model = LinearRegression().fit(design, y)
    return float(model.coef_[0])


def psm_matching(X, t, y, caliper: float = 0.02, seed: int = 0, **kw) -> float:
    """1-nearest-neighbour matching on the propensity score, with a caliper.

    ESTIMAND: this returns the ATT -- the average effect among customers who
    ACTUALLY used the promo -- not the ATE. Matching controls to treated units
    reweights the sample to the treated covariate distribution, so the target
    quantity changes.

    It is scored against `true_att` from the answer key, not `true_ate`. Under
    this DGP they happen to be close ($5.98 vs $5.87), but "close" is a property
    of this data-generating process, not a licence to conflate them. A policy
    question about extending the promo to everyone needs the ATE; a question
    about whether the promo paid off for the people who used it needs the ATT.
    """
    ps = fit_propensity(X, t, seed=seed)
    keep = trim_common_support(ps, t)
    ps, t_, y_ = ps[keep], t[keep], y[keep]

    treated_ps = ps[t_ == 1].reshape(-1, 1)
    control_ps = ps[t_ == 0].reshape(-1, 1)
    control_y = y_[t_ == 0]

    nn = NearestNeighbors(n_neighbors=1).fit(control_ps)
    dist, idx = nn.kneighbors(treated_ps)

    within = dist.ravel() <= caliper
    if within.sum() == 0:
        return np.nan
    matched_control = control_y[idx.ravel()[within]]
    treated_y = y_[t_ == 1][within]
    return float(treated_y.mean() - matched_control.mean())


def ipw(X, t, y, seed: int = 0, **kw) -> float:
    """Stabilised inverse probability weighting."""
    ps = fit_propensity(X, t, seed=seed)
    keep = trim_common_support(ps, t)
    ps, t_, y_ = ps[keep], t[keep], y[keep]

    p_treat = t_.mean()
    w = np.where(t_ == 1, p_treat / ps, (1 - p_treat) / (1 - ps))

    y1 = np.sum(w * t_ * y_) / np.sum(w * t_)
    y0 = np.sum(w * (1 - t_) * y_) / np.sum(w * (1 - t_))
    return float(y1 - y0)


def aipw(X, t, y, seed: int = 0, n_folds: int = 2, **kw) -> float:
    """Cross-fitted augmented IPW (doubly robust / DML).

    Consistent if EITHER the propensity model OR the outcome model is correct.
    Both nuisance models are gradient-boosted, so we are not assuming linearity.

    Cross-fitting matters: fitting flexible ML nuisance models and evaluating them
    on the SAME rows induces overfitting bias in the influence function. We fit on
    K-1 folds and predict on the held-out fold, which removes it. This is the
    Chernozhukov et al. double/debiased ML recipe.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % n_folds

    ps = np.zeros(n)
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)

    reg = dict(max_iter=100, max_depth=3, learning_rate=0.1, random_state=seed)

    for k in range(n_folds):
        train, test = folds != k, folds == k

        p_model = HistGradientBoostingClassifier(**reg).fit(X[train], t[train])
        ps[test] = p_model.predict_proba(X[test])[:, 1]

        tr1 = train & (t == 1)
        tr0 = train & (t == 0)
        mu1[test] = HistGradientBoostingRegressor(**reg).fit(X[tr1], y[tr1]).predict(X[test])
        mu0[test] = HistGradientBoostingRegressor(**reg).fit(X[tr0], y[tr0]).predict(X[test])

    keep = trim_common_support(ps, t)
    ps_, t_, y_, mu1_, mu0_ = ps[keep], t[keep], y[keep], mu1[keep], mu0[keep]

    score1 = mu1_ + t_ * (y_ - mu1_) / ps_
    score0 = mu0_ + (1 - t_) * (y_ - mu0_) / (1 - ps_)
    return float(np.mean(score1 - score0))


ESTIMATORS = {
    "Naive difference in means": naive_difference,
    "OLS regression adjustment": ols_adjustment,
    "Propensity score matching": psm_matching,
    "IPW (stabilised)": ipw,
    "AIPW (cross-fitted, doubly robust)": aipw,
}

# Which quantity each estimator actually targets. Scored against the matching
# ground-truth key, so the comparison table is apples-to-apples.
ESTIMANDS = {
    "Naive difference in means": "ATE",
    "OLS regression adjustment": "ATE",
    "Propensity score matching": "ATT",
    "IPW (stabilised)": "ATE",
    "AIPW (cross-fitted, doubly robust)": "ATE",
}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_ci(
    fn, X, t, y, n_boot: int = 200, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            val = fn(X[idx], t[idx], y[idx], seed=seed)
            if np.isfinite(val):
                draws.append(val)
        except Exception:
            continue
    if len(draws) < 20:
        return (np.nan, np.nan)
    return (
        float(np.percentile(draws, 100 * alpha / 2)),
        float(np.percentile(draws, 100 * (1 - alpha / 2))),
    )


def run_all(X, t, y, n_boot: int = 200, seed: int = 0) -> list[Estimate]:
    results = []
    for name, fn in ESTIMATORS.items():
        point = fn(X, t, y, seed=seed)
        lo, hi = bootstrap_ci(fn, X, t, y, n_boot=n_boot, seed=seed)
        results.append(
            Estimate(
                name=name,
                ate=float(point),
                ci_low=lo,
                ci_high=hi,
                estimand=ESTIMANDS[name],
            )
        )
    return results


def crossfit_propensity(X, t, seed: int = 0, n_folds: int = 2) -> np.ndarray:
    """Return the SAME cross-fitted propensity scores the AIPW estimator uses.

    Exposed so diagnostics can characterise the model that actually produces the
    headline estimate, rather than a different (logistic) model that happens to
    be easier to fit.
    """
    n = len(t)
    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % n_folds
    ps = np.zeros(n)
    reg = dict(max_iter=100, max_depth=3, learning_rate=0.1, random_state=seed)
    for k in range(n_folds):
        train, test = folds != k, folds == k
        ps[test] = (
            HistGradientBoostingClassifier(**reg)
            .fit(X[train], t[train])
            .predict_proba(X[test])[:, 1]
        )
    return ps
