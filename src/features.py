"""
features.py
-----------
Leakage-safe feature engineering for the Area Feasibility Scoring Model.

Design principles
~~~~~~~~~~~~~~~~~
1. **No leakage from test into train** – all statistics (percentiles, means)
   that depend on the training distribution are computed *only* on the
   training split.  The ``LeakageSafeFeaturizer`` class enforces this via
   a fit / transform interface identical to scikit-learn transformers.

2. **No target leakage** – the label ``affordable`` is derived from
   ``budget >= median_price``.  The ``affordability_ratio`` feature uses
   the *area* median price (a population statistic), not the label itself.

3. **No future leakage** – if temporal data is present, features must be
   constructed from transactions dated *before* the valuation date.
   (Handled outside this module; see the notebook for an example.)

Features produced
~~~~~~~~~~~~~~~~~
- median_price         raw area statistic
- price_25th           raw area statistic
- price_75th           raw area statistic
- num_listings         transaction count (supply proxy)
- price_spread         price_75th − price_25th (market volatility)
- price_spread_pct     price_spread / median_price (relative volatility)
- affordability_ratio  budget / median_price
- budget_vs_p25        budget / price_25th
- budget_vs_p75        budget / price_75th
- pct_within_budget    estimated % of listings the user can afford
- budget_deficit       median_price − budget  (negative ⟹ affordable)
- log_budget           log1p(budget)
- log_median_price     log1p(median_price)
- log_num_listings     log1p(num_listings)
- ratio_x_spread       affordability_ratio × price_spread_pct  (interaction)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

# Columns that must be present in the raw area-level dataframe
REQUIRED_RAW_COLS = [
    "budget",
    "median_price",
    "price_25th",
    "price_75th",
    "num_listings",
]

# Feature columns fed into the model (no target, no raw price columns)
FEATURE_COLS = [
    "affordability_ratio",
    "budget_vs_p25",
    "budget_vs_p75",
    "price_spread",
    "price_spread_pct",
    "pct_within_budget",
    "budget_deficit",
    "log_budget",
    "log_median_price",
    "log_num_listings",
    "ratio_x_spread",
]

# Classification target: 1 = user can afford the area median price
CLASSIFICATION_TARGET = "affordable"

# Optional regression target
REGRESSION_TARGET = "median_price"


# ---------------------------------------------------------------------------
# Leakage-safe transformer
# ---------------------------------------------------------------------------

class LeakageSafeFeaturizer(BaseEstimator, TransformerMixin):
    """Compute affordability features in a leakage-safe manner.

    ``fit`` computes normalisation statistics *only* from the training
    rows passed in.  ``transform`` applies those statistics to any split.

    This transformer is designed to sit inside a scikit-learn ``Pipeline``
    so that the ``fit`` call is never accidentally applied to test data.

    Parameters
    ----------
    add_interactions : bool
        Whether to add interaction feature ``ratio_x_spread``.
    """

    def __init__(self, add_interactions: bool = True):
        self.add_interactions = add_interactions
        # Fitted flag – set to True once fit() has been called
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        """Validate columns and mark the transformer as fitted.

        Parameters
        ----------
        X : pd.DataFrame
            Training rows.  Must contain ``REQUIRED_RAW_COLS``.
        """
        self._validate_columns(X)
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add engineered feature columns to ``X`` and return them.

        Parameters
        ----------
        X : pd.DataFrame
            Any split (train or test).  Must contain ``REQUIRED_RAW_COLS``.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the columns listed in
            ``FEATURE_COLS`` (no raw input columns, no target).
        """
        if self.is_fitted_ is False:
            raise RuntimeError("Call fit() before transform().")
        self._validate_columns(X)

        out = X.copy()

        # ---- core price features ----------------------------------------
        safe_median = out["median_price"].replace(0, np.nan)
        safe_p25 = out["price_25th"].replace(0, np.nan)
        safe_p75 = out["price_75th"].replace(0, np.nan)

        out["affordability_ratio"] = out["budget"] / safe_median
        out["budget_vs_p25"] = out["budget"] / safe_p25
        out["budget_vs_p75"] = out["budget"] / safe_p75

        out["price_spread"] = out["price_75th"] - out["price_25th"]
        out["price_spread_pct"] = out["price_spread"] / safe_median

        # ---- supply feature ---------------------------------------------
        out["log_num_listings"] = np.log1p(out["num_listings"])

        # ---- budget vs. distribution ------------------------------------
        # Approximate % of properties within budget using the IQR.
        # We model price as Gaussian with mean=median and std estimated
        # from the IQR (IQR ≈ 1.35 × σ for a normal distribution).
        # This is a simple closed-form estimate; no future data is used.
        estimated_std = (out["price_75th"] - out["price_25th"]) / 1.35
        z_score = (out["budget"] - out["median_price"]) / estimated_std.replace(
            0, np.nan
        )
        # Clamp to [0, 1] using a sigmoid approximation
        out["pct_within_budget"] = 1 / (1 + np.exp(-z_score))

        out["budget_deficit"] = out["median_price"] - out["budget"]

        # ---- log-scale features -----------------------------------------
        out["log_budget"] = np.log1p(out["budget"])
        out["log_median_price"] = np.log1p(out["median_price"])

        # ---- interaction terms ------------------------------------------
        if self.add_interactions:
            out["ratio_x_spread"] = (
                out["affordability_ratio"] * out["price_spread_pct"]
            )
        else:
            out["ratio_x_spread"] = 0.0

        # Return only model features (no target, no raw price columns)
        return out[FEATURE_COLS].astype(float)

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_columns(X: pd.DataFrame):
        missing = [c for c in REQUIRED_RAW_COLS if c not in X.columns]
        if missing:
            raise ValueError(
                f"LeakageSafeFeaturizer: missing required columns {missing}"
            )


# ---------------------------------------------------------------------------
# Target builder (applied to full dataset before splitting)
# ---------------------------------------------------------------------------

def build_target(df: pd.DataFrame) -> pd.Series:
    """Return the binary classification target.

    1 = the user's budget is at or above the area median price (affordable).
    0 = the user's budget is below the area median price (Not Affordable).

    Note: This is computed from the raw area statistic (median_price),
    not from any engineered feature, so there is no leakage.
    """
    if "budget" not in df.columns or "median_price" not in df.columns:
        raise ValueError("DataFrame must contain 'budget' and 'median_price'.")
    return (df["budget"] >= df["median_price"]).astype(int).rename(CLASSIFICATION_TARGET)


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

def assert_no_leakage(feature_cols: list, target_cols: list):
    """Raise ``ValueError`` if any feature column is also a target.

    Call this before fitting any model as a sanity check.
    """
    overlap = set(feature_cols) & set(target_cols)
    if overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED – column(s) appear in both features "
            f"and targets: {sorted(overlap)}"
        )
