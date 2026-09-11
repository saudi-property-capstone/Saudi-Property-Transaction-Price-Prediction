"""
Outlier Investigation for Price Prediction
MOJ Sales, 2020-2025 (cleaned/processed dataset)

Investigates whether extreme price/area transactions are invalid data,
legitimate extreme transactions, or suspicious records. Nothing is
permanently removed from the cleaned dataset -- all inspection here is
read-only and diagnostic. Model training is out of scope for this file;
it belongs in the (separate, not-yet-written) model-training workflow.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from data_io import find_suspicious_price_groups, load_clean_data  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

TOP_N = 100


# ============================================================
# TASK 1 -- INVALID VALUE INSPECTION
# ============================================================

def inspect_invalid_values(df: pd.DataFrame) -> pd.Series:

    print("\n" + "=" * 60)
    print("TASK 1: INVALID VALUE INSPECTION")
    print("=" * 60)

    n = len(df)
    checks = {
        "price <= 0": (df["price"] <= 0).sum(),
        "area <= 0": (df["area"] <= 0).sum(),
        "price missing": df["price"].isna().sum(),
        "area missing": df["area"].isna().sum(),
        "price_per_m2_calculated infinite": np.isinf(df["price_per_m2_calculated"]).sum(),
        "price_per_m2_calculated missing": df["price_per_m2_calculated"].isna().sum(),
        "n_properties <= 0": (df["n_properties"] <= 0).sum(),
        "n_properties missing": df["n_properties"].isna().sum(),
        "n_properties > 1 (bulk)": (df["n_properties"] > 1).sum(),
    }

    report = pd.Series(checks, name="row_count")
    report_pct = (report / n * 100).round(4)

    summary = pd.DataFrame({"row_count": report, "pct_of_dataset": report_pct})
    print(summary.to_string())

    print(
        "\nNote: data_cleaning.py already removes rows with n_properties > 1 "
        "(bulk transactions), invalid/non-positive price or area, and rows "
        "missing any required transaction field -- these counts are expected "
        "to be 0 (or, for missing/non-positive n_properties, retained "
        "on purpose for later review, since that field is not a required "
        "transaction column)."
    )

    return report


# ============================================================
# TASK 2 & 3 -- TOP EXTREME PRICE / AREA TRANSACTIONS
# ============================================================

CONTEXT_COLS = [
    "price", "area", "n_properties", "property_type",
    "region", "city", "city_district", "year", "quarter", "price_per_m2_calculated",
]


def flag_extreme_rows(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print(f"TASK 2 & 3: TOP {n} EXTREME PRICE / AREA TRANSACTIONS")
    print("=" * 60)

    top_price = df.nlargest(n, "price")[CONTEXT_COLS].copy()
    top_price["flagged_as"] = "extreme_price"

    top_area = df.nlargest(n, "area")[CONTEXT_COLS].copy()
    top_area["flagged_as"] = "extreme_area"

    combined = pd.concat([top_price, top_area])
    dup_mask = combined.index.duplicated(keep=False)
    combined.loc[dup_mask, "flagged_as"] = "extreme_price_and_area"
    combined = combined[~combined.index.duplicated(keep="first")]

    combined = combined.sort_values("price", ascending=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TABLES_DIR / "outlier_inspection.csv", index=False)

    print(f"Top price transaction (first 10 of {n}):")
    print(top_price.head(10).to_string(index=False))

    print(f"\nTop area transaction (first 10 of {n}):")
    print(top_area.head(10).to_string(index=False))

    overlap = (combined["flagged_as"] == "extreme_price_and_area").sum()
    print(f"\nRows in BOTH top-{n} price and top-{n} area lists: {overlap}")

    return combined


def summarize_extreme_context(extreme_df: pd.DataFrame, df: pd.DataFrame) -> None:

    print("\n--- Are extreme rows associated with specific segments? ---")

    print("\nproperty_type share among extreme rows vs dataset-wide:")
    extreme_share = extreme_df["property_type"].value_counts(normalize=True) * 100
    overall_share = df["property_type"].value_counts(normalize=True) * 100
    compare = pd.DataFrame(
        {"extreme_rows_pct": extreme_share, "dataset_wide_pct": overall_share}
    ).fillna(0).round(1)
    print(compare.to_string())

    print("\nregion share among extreme rows:")
    print((extreme_df["region"].value_counts(normalize=True) * 100).round(1).to_string())

    print("\narea stats among extreme rows (m2):")
    print(extreme_df["area"].describe().round(1).to_string())

    print("\nprice_per_m2_calculated stats among extreme rows:")
    print(extreme_df["price_per_m2_calculated"].describe().round(1).to_string())


def inspect_max_price_transaction(df: pd.DataFrame) -> None:

    print("\n--- Maximum single price transaction: full context ---")

    max_row = df.loc[[df["price"].idxmax()]]
    print(max_row[CONTEXT_COLS + ["source_file", "reference_number"]].T.to_string())

    row = max_row.iloc[0]

    same_segment = df[
        (df["region"] == row["region"]) & (df["property_type"] == row["property_type"])
    ]
    segment_median_ppm2 = same_segment["price_per_m2_calculated"].median()

    print(
        f"\nFor comparison, median price_per_m2 in {row['region']} / "
        f"{row['property_type']} is {segment_median_ppm2:,.1f} SAR/m2, "
        f"vs {row['price_per_m2_calculated']:,.1f} SAR/m2 for this row."
    )
    print(f"n_properties for this row: {row['n_properties']}")

    siblings = df[
        (df["price"] == row["price"]) & (df.index != max_row.index[0])
    ]
    if len(siblings) > 0:
        print(
            f"\n{len(siblings)} OTHER row(s) share this EXACT price "
            f"({row['price']:,.0f} SAR) with different reference_number/area:"
        )
        print(
            siblings[["reference_number", "area", "city_district", "date_gregorian_raw"]]
            .to_string(index=False)
        )
        print(
            "Same total price repeated across distinct reference numbers on "
            "differently-sized parcels (not a round list price, and area "
            "varies by orders of magnitude) is not consistent with each row "
            "being an independently priced parcel -- it looks like a single "
            "bulk contract's total value was duplicated onto each parcel's "
            "row rather than allocated per parcel, even though each row "
            "records n_properties = 1. This is a SUSPICIOUS pattern flagged "
            "for documented review, not an automatic deletion."
        )
    else:
        print("\nNo other row shares this exact price.")


def detect_suspicious_price_duplication(df: pd.DataFrame, area_ratio_threshold: float = 10.0) -> pd.DataFrame:

    print("\n--- Same price, same day, different parcels: legitimate vs suspicious ---")

    p99 = df["price"].quantile(0.99)
    high = df[df["price"] > p99]

    grouped = high.groupby(["price", "date_gregorian_raw"]).agg(
        n_rows=("reference_number", "nunique"),
        min_area=("area", "min"),
        max_area=("area", "max"),
    )
    clusters = grouped[grouped["n_rows"] > 1].copy()
    clusters["area_ratio"] = clusters["max_area"] / clusters["min_area"].replace(0, np.nan)

    n_rows_in_clusters = clusters["n_rows"].sum()
    print(
        f"Above P99 price ({p99:,.0f} SAR): {len(clusters):,} clusters where "
        f"the same price+date is shared by >1 reference_number "
        f"({n_rows_in_clusters:,} rows total, {n_rows_in_clusters / len(high):.1%} of high-price rows)."
    )

    suspicious = clusters[clusters["area_ratio"] > area_ratio_threshold]
    n_suspicious_rows = suspicious["n_rows"].sum()
    print(
        f"Of those, {len(suspicious):,} clusters ({n_suspicious_rows:,} rows) have "
        f"area varying by more than {area_ratio_threshold:.0f}x within the same "
        "identical total price -- consistent with a shared/duplicated contract "
        "value rather than legitimate per-parcel pricing (SUSPICIOUS, documented "
        "rule above). The remaining clusters (similar areas, often round prices) "
        "look like standardized subdivision plot pricing, a normal market pattern "
        "(a statistically extreme price, but not an invalid or suspicious one)."
    )

    return suspicious.reset_index()


def validate_suspicious_price_groups_removed(df: pd.DataFrame) -> None:
    """Validation only -- reuses the exact same rule data_cleaning.py applies
    to REMOVE suspicious high-price groups (see find_suspicious_price_groups
    in data_io.py), and confirms none remain in the processed dataset.
    Nothing is filtered or modified here.
    """

    print("\n--- Validation: suspicious high-price groups (same rule as data_cleaning.py) ---")

    remaining = find_suspicious_price_groups(df)

    print(
        f"[{'PASS' if len(remaining) == 0 else 'FAIL'}] "
        f"No suspicious high-price groups remain in the processed dataset "
        f"({len(remaining)} found)."
    )

    if len(remaining) > 0:
        print(remaining.to_string(index=False))


# ============================================================
# TASK 4 -- GROUP-LEVEL OUTLIER DIAGNOSTICS
# ============================================================

def robust_price_stats(group: pd.DataFrame) -> pd.Series:

    price = group["price"]
    q1, q3 = price.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr

    return pd.Series({
        "count": len(price),
        "median": price.median(),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "p95": price.quantile(0.95),
        "p99": price.quantile(0.99),
        "p999": price.quantile(0.999),
        "iqr_flagged_pct": (price > upper_fence).mean() * 100,
    })


def group_level_diagnostics(df: pd.DataFrame, min_group_size: int = 1000) -> None:

    print("\n" + "=" * 60)
    print("TASK 4: PRICE DISTRIBUTION WITHIN SEGMENTS (diagnostic, not deletion)")
    print("=" * 60)

    print("\nBy property_type:")
    print(df.groupby("property_type").apply(robust_price_stats, include_groups=False).round(1).to_string())

    print("\nBy region:")
    print(df.groupby("region").apply(robust_price_stats, include_groups=False).round(1).to_string())

    city_counts = df["city"].value_counts()
    large_cities = city_counts[city_counts >= min_group_size].index
    print(f"\nBy city (only cities with >= {min_group_size:,} transactions, top 15 by volume):")
    city_stats = (
        df[df["city"].isin(large_cities)]
        .groupby("city")
        .apply(robust_price_stats, include_groups=False)
        .sort_values("count", ascending=False)
        .head(15)
    )
    print(city_stats.round(1).to_string())

    print(
        "\nIQR fences above are a diagnostic flag per segment only -- a "
        "transaction beyond its own segment's fence is a STATISTICALLY "
        "EXTREME value, not automatically an invalid or suspicious one, "
        "since it may be normal for that market (e.g. Makkah commercial vs "
        "Northern Borders residential). It is never automatically removed."
    )


def outlier_distribution_by_segment(df: pd.DataFrame) -> None:

    print("\n--- Where do the global top 1% of prices come from? ---")

    p99 = df["price"].quantile(0.99)
    outliers = df[df["price"] > p99]

    print(f"Global P99 price threshold: {p99:,.0f} SAR ({len(outliers):,} rows above it)")

    print("\nproperty_type share of these rows vs dataset-wide share:")
    compare = pd.DataFrame({
        "pct_of_p99_outliers": (outliers["property_type"].value_counts(normalize=True) * 100),
        "pct_of_dataset": (df["property_type"].value_counts(normalize=True) * 100),
    }).fillna(0).round(1)
    print(compare.to_string())

    print("\nregion share of these rows vs dataset-wide share:")
    compare_region = pd.DataFrame({
        "pct_of_p99_outliers": (outliers["region"].value_counts(normalize=True) * 100),
        "pct_of_dataset": (df["region"].value_counts(normalize=True) * 100),
    }).fillna(0).round(1)
    print(compare_region.to_string())


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    df = load_clean_data()

    inspect_invalid_values(df)

    extreme_df = flag_extreme_rows(df, n=TOP_N)
    summarize_extreme_context(extreme_df, df)
    inspect_max_price_transaction(df)
    detect_suspicious_price_duplication(df)
    validate_suspicious_price_groups_removed(df)

    group_level_diagnostics(df)
    outlier_distribution_by_segment(df)

    print("\nOutlier analysis complete.")
    print(f"Tables saved to: {TABLES_DIR}")
    print("The cleaned dataset (data/processed/*.csv.gz) was not modified.")


if __name__ == "__main__":
    main()
