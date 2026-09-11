"""
Shared read-only loader for the cleaned/processed MOJ sales dataset.

Used by both EDA.py and outlier_analysis.py so neither analysis script
repeats cleaning logic or depends on the other. This module only reads
data/processed -- it never writes back to it and never touches data/raw.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "moj_sales_2020_2025_merged.csv.gz"
)

# Kept in sync with REQUIRED_TRANSACTION_COLUMNS in data_cleaning.py -- these
# are the fields data_cleaning.py guarantees have zero missing values.
REQUIRED_TRANSACTION_COLUMNS = [
    "price", "area", "transaction_date", "region_en", "property_type",
]


def load_clean_data() -> pd.DataFrame:
    """Load the processed MOJ dataset.

    This does NOT perform any cleaning or row filtering -- data_cleaning.py
    is the only place that permanently removes rows. This loader only
    validates that the required transaction fields are already fully
    populated, and raises a clear error if they are not (which means the
    processed file is stale or was produced by an older version of
    data_cleaning.py and should be regenerated).
    """

    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{PROCESSED_PATH}\n"
            "Run src/data_cleaning.py first."
        )

    df = pd.read_csv(
        PROCESSED_PATH,
        compression="gzip",
        parse_dates=["transaction_date"],
        low_memory=False,
    )

    missing_counts = df[REQUIRED_TRANSACTION_COLUMNS].isna().sum()
    still_missing = missing_counts[missing_counts > 0]

    if len(still_missing) > 0:
        raise ValueError(
            "The processed dataset has missing values in required "
            f"transaction fields:\n{still_missing.to_string()}\n"
            "Rerun src/data_cleaning.py to regenerate a fully cleaned "
            "dataset before running this analysis."
        )

    df["region"] = df["region_en"]

    return df


# Shared with outlier_analysis.py so the "suspicious high-price group" rule
# lives in exactly one place: data_cleaning.py applies it (removes matching
# rows), outlier_analysis.py applies it again read-only (confirms none
# remain in the final processed file).
SUSPICIOUS_GROUP_COLS = ["price", "transaction_date", "region_en", "city"]


def find_suspicious_price_groups(
    df: pd.DataFrame,
    group_cols: list = None,
    area_ratio_threshold: float = 10.0,
) -> pd.DataFrame:
    """Identify suspicious high-price transaction groups.

    A group of rows (sharing identical price/transaction_date/region_en/
    city by default) is flagged when: its price is above the global P99,
    it contains more than one distinct reference_number, and
    max(area) / min(area) exceeds area_ratio_threshold -- consistent with
    a single bulk contract's total value having been duplicated onto
    multiple parcels' rows rather than priced per parcel.

    Read-only: never modifies or filters df. Returns one row per
    suspicious group with the group key columns, n_reference_numbers,
    min_area, max_area, and area_ratio.
    """

    if group_cols is None:
        group_cols = SUSPICIOUS_GROUP_COLS

    global_p99_price = df["price"].quantile(0.99)
    high = df[df["price"] > global_p99_price]

    grouped = (
        high.groupby(group_cols)
        .agg(
            n_reference_numbers=("reference_number", "nunique"),
            min_area=("area", "min"),
            max_area=("area", "max"),
        )
        .reset_index()
    )
    grouped["area_ratio"] = grouped["max_area"] / grouped["min_area"].replace(0, np.nan)

    suspicious = grouped[
        (grouped["n_reference_numbers"] > 1) & (grouped["area_ratio"] > area_ratio_threshold)
    ].reset_index(drop=True)

    return suspicious
