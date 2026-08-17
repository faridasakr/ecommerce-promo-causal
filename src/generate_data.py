"""
Synthetic e-commerce dataset with a PLANTED causal effect.

Business setup
--------------
An online retailer ran a free-shipping promotion for 4 weeks. Some customers
used it, some didn't. Leadership wants to know: did the promo CAUSE incremental
revenue, or did it just subsidise purchases that would have happened anyway?

The catch (and the point of the project): promo uptake is NOT random. Engaged,
high-spending, long-tenured customers are far more likely to use it AND would
have spent more regardless. That is textbook confounding by self-selection.

We generate this using the potential-outcomes framework: every customer gets
both Y(0) (revenue if they had NOT used the promo) and Y(1) (revenue if they
HAD). We then reveal only the one matching their actual treatment. Because we
generated both, the true ATE is known exactly and written to a held-out file.

The analysis pipeline never reads that file. It is only used at the very end to
score how close each estimator got.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Ground-truth parameters. THESE ARE THE ANSWER KEY.
# --------------------------------------------------------------------------
SEED = 20260816
N_CUSTOMERS = 50_000

# The promo works through two planted channels:
#   INCIDENCE_LIFT -- percentage-point lift in probability of buying at all
#   BASKET_LIFT    -- dollar lift in basket size among buyers
INCIDENCE_LIFT = 0.055
BASKET_LIFT = 12.00

# Price-sensitive (low prior spend) shoppers are induced to buy at all far more
# readily by free shipping. Both channels therefore point the same direction,
# which is what makes the segment recommendation actionable.
INCIDENCE_HTE = 0.040

# Effect heterogeneity: the promo helps LOW prior spenders more than high ones.
# basket_lift_i = BASKET_LIFT - HTE_SLOPE * z_prior_spend
# This creates the business insight a naive segment analysis gets backwards.
HTE_SLOPE = 6.00

CHANNELS = ["organic", "paid_search", "social", "referral"]
CHANNEL_P = [0.38, 0.30, 0.22, 0.10]

# Channel shifts on treatment uptake and on baseline revenue.
CHANNEL_TREAT = {"organic": 0.00, "paid_search": 0.35, "social": -0.20, "referral": 0.45}
CHANNEL_REV = {"organic": 0.00, "paid_search": 0.18, "social": -0.12, "referral": 0.30}

REGIONS = ["northeast", "midwest", "south", "west"]


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def generate(n: int = N_CUSTOMERS, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- covariates
    tenure_days = rng.gamma(shape=2.2, scale=180, size=n).clip(14, 2000)
    prior_12w_spend = rng.lognormal(mean=3.9, sigma=0.85, size=n).clip(0, 4000)
    email_engagement = rng.beta(2.0, 4.5, size=n)  # open/click rate 0-1
    num_prior_orders = rng.poisson(lam=1.0 + tenure_days / 400).clip(0, 60)
    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_P)
    region = rng.choice(REGIONS, size=n, p=[0.22, 0.24, 0.31, 0.23])
    is_mobile = rng.binomial(1, 0.62, size=n)

    z_tenure = _zscore(np.log(tenure_days))
    z_spend = _zscore(np.log1p(prior_12w_spend))
    z_engage = _zscore(email_engagement)
    z_orders = _zscore(np.log1p(num_prior_orders))

    ch_treat = np.array([CHANNEL_TREAT[c] for c in channel])
    ch_rev = np.array([CHANNEL_REV[c] for c in channel])

    # ------------------------------------------------- treatment assignment (confounded)
    # Engaged, high-spend, long-tenure customers self-select into the promo.
    logit_p = (
        -0.85
        + 0.80 * z_engage
        + 0.65 * z_spend
        + 0.40 * z_tenure
        + 0.25 * z_orders
        + ch_treat
        + 0.15 * is_mobile
    )
    p_treat = 1 / (1 + np.exp(-logit_p))
    treated = rng.binomial(1, p_treat)

    # ------------------------------------------------------------ potential outcomes
    # Y(0): revenue in the 4-week window had they NOT used the promo.
    # Same drivers as treatment uptake -> confounding.
    log_mu0 = (
        3.05
        + 0.62 * z_spend
        + 0.38 * z_engage
        + 0.22 * z_tenure
        + 0.18 * z_orders
        + ch_rev
        - 0.08 * is_mobile
    )
    mu0 = np.exp(log_mu0)
    noise = rng.normal(0, 14.0, size=n)
    basket0 = np.clip(mu0 + noise, 0, None)

    # Free shipping works through TWO channels, both of which we plant:
    #   (a) incidence  -- it induces some non-buyers to buy at all
    #   (b) basket size -- it lifts spend among people who would have bought
    # Effect (b) is bigger for LOW prior spenders (price-sensitive shoppers).
    basket_lift = np.clip(BASKET_LIFT - HTE_SLOPE * z_spend, 0, None)
    basket1 = np.clip(basket0 + basket_lift, 0, None)

    # Coupled (monotone) incidence draw: one uniform, two thresholds. Guarantees
    # nobody is induced *out* of buying and keeps the counterfactual pair tight.
    p_buy0 = np.clip(0.40 + 0.16 * z_spend + 0.10 * z_engage, 0.02, 0.95)
    p_buy1 = np.clip(p_buy0 + INCIDENCE_LIFT - INCIDENCE_HTE * z_spend, 0.02, 0.98)
    u = rng.random(n)
    bought0 = (u < p_buy0).astype(float)
    bought1 = (u < p_buy1).astype(float)

    y0 = basket0 * bought0
    y1 = basket1 * bought1

    # ------------------------------------------------------------- observed outcome
    revenue = np.where(treated == 1, y1, y0)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:07d}" for i in range(n)],
            "tenure_days": tenure_days.round(0).astype(int),
            "prior_12w_spend": prior_12w_spend.round(2),
            "email_engagement": email_engagement.round(4),
            "num_prior_orders": num_prior_orders,
            "acquisition_channel": channel,
            "region": region,
            "is_mobile": is_mobile,
            "used_free_shipping": treated,
            "revenue_promo_window": revenue.round(2),
        }
    )

    # ------------------------------------------------------------------ ground truth
    # Individual treatment effects. Persisting these is what lets the scoring
    # step compute an estimand-specific truth for whatever sub-population an
    # estimator actually used (trimmed, matched, ...) instead of comparing
    # everything to a hard-coded full-population number.
    tau = y1 - y0

    truth = {
        "tau_individual": tau,
        "true_ate": float(np.mean(y1 - y0)),
        "true_att": float(np.mean((y1 - y0)[treated == 1])),
        "incidence_lift_pp": INCIDENCE_LIFT,
        "basket_lift_dollars": BASKET_LIFT,
        "hte_slope": HTE_SLOPE,
        "true_ate_low_spend_tercile": float(
            np.mean((y1 - y0)[z_spend <= np.quantile(z_spend, 1 / 3)])
        ),
        "true_ate_high_spend_tercile": float(
            np.mean((y1 - y0)[z_spend >= np.quantile(z_spend, 2 / 3)])
        ),
        "naive_diff_in_means": float(
            revenue[treated == 1].mean() - revenue[treated == 0].mean()
        ),
        "treated_share": float(treated.mean()),
        "n": int(n),
        "seed": seed,
        "tau_individual_file": "tau_individual.npy",
        "tau_alignment": (
            "Row i of tau_individual.npy is the individual treatment effect "
            "y1 - y0 for customer_id f'C{i:07d}'. add_realistic_mess() shuffles "
            "and duplicates rows, so consumers MUST align on customer_id rather "
            "than on row position."
        ),
        "note": (
            "true_ate is computed directly from the full potential-outcome pairs "
            "(Y1, Y0) and is the number to score estimators against. It combines the "
            "incidence lift and the basket lift, so it is not simply BASKET_LIFT. "
            "true_ate and true_att are full-population and all-treated summaries; "
            "for an estimator that trims or matches, take the mean of "
            "tau_individual over the units it actually used."
        ),
    }

    return df, truth


def add_realistic_mess(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Introduce the kind of dirt that exists in every real warehouse table.

    Cleaning this is real work and belongs in the project. Nothing here changes
    the causal structure -- it is cosmetic damage only.
    """
    rng = np.random.default_rng(seed + 1)
    df = df.copy()

    # 1. Missing engagement scores for ~4% of rows.
    miss = rng.random(len(df)) < 0.04
    df.loc[miss, "email_engagement"] = np.nan

    # 2. Inconsistent region capitalisation / whitespace.
    variants = {"northeast": ["northeast", "NorthEast", " northeast ", "NORTHEAST"]}
    mask = df["region"] == "northeast"
    idx = df.index[mask]
    df.loc[idx, "region"] = rng.choice(variants["northeast"], size=len(idx))

    # 3. Revenue stored as string with currency symbols on a subset.
    dirty = rng.random(len(df)) < 0.08
    df["revenue_promo_window"] = df["revenue_promo_window"].astype(object)
    df.loc[dirty, "revenue_promo_window"] = df.loc[dirty, "revenue_promo_window"].map(
        lambda v: f"${v:,.2f}"
    )

    # 4. A handful of exact duplicate rows.
    dupes = df.sample(n=180, random_state=7)
    df = pd.concat([df, dupes], ignore_index=True)

    # 5. Shuffle so duplicates aren't all at the bottom.
    return df.sample(frac=1.0, random_state=11).reset_index(drop=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    truth_dir = root / "data" / "ground_truth"
    raw_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    clean, truth = generate()
    messy = add_realistic_mess(clean)

    messy.to_csv(raw_dir / "customers.csv", index=False)

    # The per-customer effects go to .npy: 50k floats would bloat the JSON and
    # they are read as an array, never by eye.
    tau = truth.pop("tau_individual")
    np.save(truth_dir / "tau_individual.npy", tau)

    with open(truth_dir / "answer_key.json", "w") as f:
        json.dump(truth, f, indent=2)

    print(f"Wrote {len(messy):,} rows -> data/raw/customers.csv")
    print(f"Wrote {len(tau):,} individual effects -> data/ground_truth/tau_individual.npy")
    print(f"True ATE (held out): ${truth['true_ate']:.2f}")
    print(f"Naive difference in means: ${truth['naive_diff_in_means']:.2f}")
    print(f"Selection bias: ${truth['naive_diff_in_means'] - truth['true_ate']:.2f}")


if __name__ == "__main__":
    main()
