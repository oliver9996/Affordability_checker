"""
data_loader.py
--------------
Data loading and area-level aggregation for the Area Feasibility Scoring Model.

Supports:
  - UK Land Registry Price Paid data (CSV download)
  - Synthetic data generation for development / testing

Land Registry columns used:
  transaction_id, price, date_of_transfer, postcode, property_type,
  old_new, duration, paon, saon, street, locality, town_city,
  district, county, ppd_category_type, record_status

Area aggregation level: postcode district (e.g. "SW1A" from "SW1A 1AA")
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Land Registry Price Paid CSV column names (no header in the official file)
LR_COLUMNS = [
    "transaction_id",
    "price",
    "date_of_transfer",
    "postcode",
    "property_type",
    "old_new",
    "duration",
    "paon",
    "saon",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category_type",
    "record_status",
]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_land_registry(path: str) -> pd.DataFrame:
    """Load a UK Land Registry Price Paid CSV file.

    The official download has no header row, so column names are assigned
    from ``LR_COLUMNS``.

    Parameters
    ----------
    path : str
        Local path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with typed columns.
    """
    df = pd.read_csv(path, header=None, names=LR_COLUMNS, low_memory=False)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date_of_transfer"] = pd.to_datetime(df["date_of_transfer"], errors="coerce")

    # Remove rows with missing price or postcode
    df = df.dropna(subset=["price", "postcode"])
    df = df[df["price"] > 0]

    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load a generic CSV file that already has a header row.

    Expects at minimum: ``price``, ``postcode`` (or ``area``).
    """
    df = pd.read_csv(path, low_memory=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "date_of_transfer" in df.columns:
        df["date_of_transfer"] = pd.to_datetime(
            df["date_of_transfer"], errors="coerce"
        )
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    return df


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_transactions: int = 20_000,
    n_areas: int = 50,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a synthetic property transactions dataset.

    The synthetic data mimics UK property price distributions across
    multiple postcode districts with realistic variation.

    Parameters
    ----------
    n_transactions : int
        Number of individual property transactions to generate.
    n_areas : int
        Number of distinct postcode district areas.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Transaction-level dataframe with columns:
        ``transaction_id``, ``price``, ``date_of_transfer``,
        ``postcode``, ``area``.
    """
    rng = np.random.default_rng(random_state)

    # Each area has its own underlying price distribution
    area_base_prices = rng.uniform(150_000, 800_000, size=n_areas)
    area_std_prices = area_base_prices * rng.uniform(0.15, 0.40, size=n_areas)
    areas = [f"AREA{i:02d}" for i in range(n_areas)]

    rows = []
    for i in range(n_transactions):
        area_idx = rng.integers(0, n_areas)
        area = areas[area_idx]
        price = float(
            rng.normal(area_base_prices[area_idx], area_std_prices[area_idx])
        )
        price = max(price, 10_000)  # floor price
        date = pd.Timestamp("2018-01-01") + pd.Timedelta(
            days=int(rng.integers(0, 365 * 5))
        )
        rows.append(
            {
                "transaction_id": f"T{i:06d}",
                "price": round(price, -2),  # round to nearest £100
                "date_of_transfer": date,
                "postcode": f"{area} 1AA",
                "area": area,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Area extraction
# ---------------------------------------------------------------------------

def extract_area(postcode: str) -> str:
    """Return the postcode district (first 2–4 chars) from a full postcode.

    Examples
    --------
    >>> extract_area("SW1A 1AA")
    'SW1A'
    >>> extract_area("M1 1AE")
    'M1'
    """
    if not isinstance(postcode, str):
        return "UNKNOWN"
    parts = postcode.strip().split()
    if len(parts) >= 2:
        return parts[0].upper()
    # Fallback: everything except last 3 chars
    pc = postcode.strip().upper().replace(" ", "")
    return pc[:-3] if len(pc) > 3 else pc


# ---------------------------------------------------------------------------
# Area-level aggregation
# ---------------------------------------------------------------------------

def aggregate_to_area(
    df: pd.DataFrame,
    price_col: str = "price",
    area_col: str = "area",
) -> pd.DataFrame:
    """Aggregate individual transactions to area-level summary statistics.

    This produces the *raw* area features before any user-specific
    affordability features are computed.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction-level dataframe.
    price_col : str
        Name of the price column.
    area_col : str
        Name of the area identifier column. If absent, it is derived
        from the ``postcode`` column.

    Returns
    -------
    pd.DataFrame
        One row per area with columns:
        ``area``, ``median_price``, ``price_25th``, ``price_75th``,
        ``mean_price``, ``price_std``, ``num_listings``.
    """
    if area_col not in df.columns:
        if "postcode" not in df.columns:
            raise ValueError(
                "DataFrame must have an 'area' or 'postcode' column."
            )
        df = df.copy()
        df[area_col] = df["postcode"].apply(extract_area)

    agg = (
        df.groupby(area_col)[price_col]
        .agg(
            median_price="median",
            price_25th=lambda x: x.quantile(0.25),
            price_75th=lambda x: x.quantile(0.75),
            mean_price="mean",
            price_std="std",
            num_listings="count",
        )
        .reset_index()
        .rename(columns={area_col: "area"})
    )

    return agg


# ---------------------------------------------------------------------------
# Full data preparation entry point
# ---------------------------------------------------------------------------

def prepare_dataset(
    path: str = None,
    user_budget: float = 300_000,
    use_synthetic: bool = False,
    n_transactions: int = 20_000,
    n_areas: int = 50,
) -> pd.DataFrame:
    """Load (or generate), aggregate, and attach a user budget.

    Parameters
    ----------
    path : str, optional
        Path to data file. Required unless ``use_synthetic=True``.
    user_budget : float
        The user's property budget in £.
    use_synthetic : bool
        If True, generate synthetic data regardless of ``path``.
    n_transactions : int
        Number of synthetic transactions (only used when
        ``use_synthetic=True``).
    n_areas : int
        Number of synthetic areas (only used when ``use_synthetic=True``).

    Returns
    -------
    pd.DataFrame
        Area-level dataframe with a ``budget`` column attached, ready for
        feature engineering.
    """
    if use_synthetic or path is None:
        print("Generating synthetic transaction data …")
        tx_df = generate_synthetic_data(
            n_transactions=n_transactions, n_areas=n_areas
        )
    else:
        print(f"Loading data from: {path}")
        # Auto-detect Land Registry format (no header) vs generic CSV
        with open(path, "r") as f:
            first_line = f.readline()
        if first_line.startswith("{"):
            raise ValueError("Expected CSV, got JSON-like content.")
        try:
            pd.to_numeric(first_line.split(",")[1])  # price in column 2 → no header
            tx_df = load_land_registry(path)
        except (ValueError, IndexError):
            tx_df = load_csv(path)

        if "area" not in tx_df.columns and "postcode" in tx_df.columns:
            tx_df["area"] = tx_df["postcode"].apply(extract_area)

    area_df = aggregate_to_area(tx_df)
    area_df["budget"] = user_budget

    print(
        f"Dataset prepared: {len(area_df)} areas, "
        f"user budget £{user_budget:,.0f}"
    )
    return area_df
