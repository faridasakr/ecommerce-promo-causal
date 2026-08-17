"""Load and clean the raw warehouse extract.

The raw file has the usual damage: string-formatted currency, inconsistent
categorical casing, missing engagement scores, and duplicate rows. Cleaning is
deterministic and logged so the pipeline is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONFOUNDERS = [
    "tenure_days",
    "prior_12w_spend",
    "email_engagement",
    "num_prior_orders",
    "is_mobile",
]
CATEGORICALS = ["acquisition_channel", "region"]
TREATMENT = "used_free_shipping"
OUTCOME = "revenue_promo_window"


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    log = []
    n0 = len(df)

    # 1. Exact duplicates -- these are warehouse artefacts, not real customers.
    #    .copy() is deliberate: without it, downstream column assignment can
    #    raise SettingWithCopyWarning on pandas 2.x (pandas 3.x's copy-on-write
    #    hides it). Never mutate a caller's frame in place either way.
    df = df.drop_duplicates().copy()
    log.append(f"dropped {n0 - len(df)} exact duplicate rows")

    # 2. Currency strings -> float. Check by dtype kind, not `== object`:
    #    pandas 3.x reads mixed text columns as the new `str` dtype, so an
    #    object-identity check silently misses them.
    if not pd.api.types.is_numeric_dtype(df[OUTCOME]):
        df[OUTCOME] = (
            df[OUTCOME]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .astype(float)
        )
        log.append("parsed currency-formatted revenue strings")

    # 3. Categorical hygiene.
    for col in CATEGORICALS:
        df[col] = df[col].astype(str).str.strip().str.lower()
    log.append("normalised categorical casing/whitespace")

    # 4. Missing engagement -> median impute + explicit indicator.
    #    The indicator matters: if missingness correlates with treatment, dropping
    #    the flag silently discards information the estimator needs.
    n_missing = int(df["email_engagement"].isna().sum())
    df["engagement_missing"] = df["email_engagement"].isna().astype(int)
    df["email_engagement"] = df["email_engagement"].fillna(df["email_engagement"].median())
    log.append(f"imputed {n_missing} missing engagement scores (+ indicator column)")

    if verbose:
        for line in log:
            print(f"  clean: {line}")

    return df.reset_index(drop=True)


def build_design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, treatment, outcome, feature_names).

    Continuous confounders are log-transformed where skewed, then standardised.
    Categoricals are one-hot encoded with the first level dropped.
    """
    X = pd.DataFrame(index=df.index)

    X["log_tenure"] = np.log(df["tenure_days"].clip(lower=1))
    X["log_prior_spend"] = np.log1p(df["prior_12w_spend"])
    X["engagement"] = df["email_engagement"]
    X["log_prior_orders"] = np.log1p(df["num_prior_orders"])
    X["is_mobile"] = df["is_mobile"]
    X["engagement_missing"] = df["engagement_missing"]

    for col in CATEGORICALS:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)

    # Standardise continuous columns only.
    for col in ["log_tenure", "log_prior_spend", "engagement", "log_prior_orders"]:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    return (
        X.to_numpy(dtype=float),
        df[TREATMENT].to_numpy(dtype=int),
        df[OUTCOME].to_numpy(dtype=float),
        list(X.columns),
    )


def load(path: str | Path | None = None, verbose: bool = True) -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    path = Path(path) if path else root / "data" / "raw" / "customers.csv"
    raw = pd.read_csv(path)
    if verbose:
        print(f"  loaded {len(raw):,} raw rows from {path.name}")
    return clean(raw, verbose=verbose)


if __name__ == "__main__":
    df = load()
    print(f"\nCleaned: {len(df):,} rows")
    print(df[[TREATMENT, OUTCOME]].groupby(TREATMENT).agg(["count", "mean"]).round(2))
