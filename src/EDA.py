"""
Exploratory Data Analysis
MOJ Sales, 2020 Q1 - 2025 Q4 (cleaned/processed dataset)

Reads the merged dataset produced by data_cleaning.py and produces
summary tables (outputs/tables/) and figures (outputs/figures/).
Never reads from data/raw and never writes back to data/processed --
all cleaning/row-filtering already happened in data_cleaning.py.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
from data_io import PROCESSED_PATH, load_clean_data  # noqa: E402

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

EXPECTED_YEARS = list(range(2020, 2026))
EXPECTED_QUARTERS = [1, 2, 3, 4]

# data_cleaning.py harmonizes property_type (broad USE category) and
# guarantees it is populated for every row in the processed dataset. Only
# Residential / Commercial / Agricultural are retained -- Industrial /
# Mixed Use / Other were excluded upstream for insufficient sample size.


# ============================================================
# 1. DATASET OVERVIEW
# ============================================================

def dataset_overview(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)

    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print(
        "Date range: "
        f"{df['transaction_date'].min().date()} -> "
        f"{df['transaction_date'].max().date()}"
    )

    print("\nMissing values (top 10):")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(10).to_string())

    n_duplicates = df.duplicated().sum()
    print(f"\nFully duplicated rows: {n_duplicates:,}")

    print("\nNumerical summary:")
    numeric_cols = ["price", "area", "price_per_m2_calculated", "n_properties"]
    print(df[numeric_cols].describe().round(2).to_string())

    print("\nUnique counts (location / categorical):")
    for col in ["region", "city", "city_district", "property_type"]:
        print(f"  {col:<25s}: {df[col].nunique():,}")


# ============================================================
# 2. TRANSACTION PRICE DISTRIBUTION
# ============================================================

def analyze_price(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("2. TRANSACTION PRICE DISTRIBUTION")
    print("=" * 60)

    price = df["price"]

    print(f"Mean:     {price.mean():,.0f} SAR")
    print(f"Median:   {price.median():,.0f} SAR")
    print(f"Min:      {price.min():,.0f} SAR")
    print(f"Max:      {price.max():,.0f} SAR")
    print(f"Skewness: {price.skew():.2f}")

    percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    print("\nPercentiles:")
    print(price.quantile(percentiles).round(0).to_string())

    # Real-estate prices have a long right tail; the median (above) is the
    # more representative summary. For readability, plots are clipped to
    # the 1st-99th percentile range -- this only affects the display,
    # not the statistics above or any saved table.
    lo, hi = price.quantile([0.01, 0.99])
    price_display = price[(price >= lo) & (price <= hi)]
    n_outliers = (price > hi).sum() + (price < lo).sum()
    print(f"\nRows outside [1st, 99th] percentile (display only): {n_outliers:,}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(price_display, bins=80, color="#4C72B0")
    axes[0].axvline(price.median(), color="red", linestyle="--", label="Median")
    axes[0].set_title("Transaction Price Distribution (1st-99th pct)")
    axes[0].set_xlabel("Price (SAR)")
    axes[0].legend()

    axes[1].boxplot(price_display, vert=True)
    axes[1].set_title("Transaction Price Boxplot (1st-99th pct)")
    axes[1].set_ylabel("Price (SAR)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "transaction_price_distribution.png", dpi=150)
    plt.close(fig)


# ============================================================
# 3. PROPERTY AREA DISTRIBUTION
# ============================================================

def analyze_area(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("3. PROPERTY AREA DISTRIBUTION")
    print("=" * 60)

    area = df["area"]

    print(f"Mean:     {area.mean():,.1f} m2")
    print(f"Median:   {area.median():,.1f} m2")
    print(f"Min:      {area.min():,.1f} m2")
    print(f"Max:      {area.max():,.1f} m2")
    print(f"Skewness: {area.skew():.2f}")

    percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    print("\nPercentiles:")
    print(area.quantile(percentiles).round(1).to_string())

    lo, hi = area.quantile([0.01, 0.99])
    n_outliers = (area > hi).sum() + (area < lo).sum()
    print(f"\nRows outside [1st, 99th] percentile: {n_outliers:,}")


# ============================================================
# 4. PRICE VS AREA
# ============================================================

def price_vs_area(df: pd.DataFrame, sample_size: int = 40_000) -> None:

    print("\n" + "=" * 60)
    print("4. PRICE VS AREA (sampled scatter)")
    print("=" * 60)

    price_hi = df["price"].quantile(0.99)
    area_hi = df["area"].quantile(0.99)

    plot_df = df[(df["price"] <= price_hi) & (df["area"] <= area_hi)]

    sample = plot_df.sample(
        n=min(sample_size, len(plot_df)), random_state=42
    )

    print(f"Plotting {len(sample):,} sampled rows (below 99th pct of price and area).")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sample["area"], sample["price"], s=4, alpha=0.3, color="#4C72B0")
    ax.set_xlabel("Area (m2)")
    ax.set_ylabel("Price (SAR)")
    ax.set_title("Price vs Area (sampled, below 99th percentile)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "price_area_relationship.png", dpi=150)
    plt.close(fig)

    plot_df = df[(df["area"] > 0) & (df["price"] > 0)].sample(
        n=min(50_000, len(df)),
        random_state=42
    )

    fig = plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=plot_df,
        x="area",
        y="price",
        alpha=0.25,
        s=15
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Price vs Area — Log Scale")
    plt.xlabel("Area (m², log scale)")
    plt.ylabel("Price (SAR, log scale)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "price_area_log_scale.png", dpi=150)
    plt.close(fig)

    g = sns.relplot(
        data=plot_df,
        x="area",
        y="price",
        col="property_type",
        col_wrap=3,
        alpha=0.25,
        facet_kws={"sharex": False, "sharey": False}
    )

    g.savefig(FIGURES_DIR / "price_area_by_property_type.png", dpi=150)
    plt.close(g.figure)


# ============================================================
# 5. QUARTERLY SUMMARY
# ============================================================

def quarterly_summary(df: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("5. QUARTERLY TRANSACTION VOLUME, 2020-2025")
    print("=" * 60)

    quarterly = (
        df.groupby(["year", "quarter"])["price"]
        .agg(transaction_count="count", total_value="sum", mean_price="mean", median_price="median")
        .reset_index()
        .sort_values(["year", "quarter"])
        .reset_index(drop=True)
    )

    quarterly["period"] = (
        quarterly["year"].astype(str) + "-Q" + quarterly["quarter"].astype(str)
    )

    # Optional trend metrics
    quarterly["qoq_growth_pct"] = quarterly["transaction_count"].pct_change() * 100
    quarterly["yoy_growth_pct"] = (
        quarterly["transaction_count"].pct_change(periods=4) * 100
    )
    quarterly["rolling_4q_avg_count"] = (
        quarterly["transaction_count"].rolling(window=4).mean()
    )

    price_per_m2_median = (
        df.groupby(["year", "quarter"])["price_per_m2_calculated"]
        .median()
        .reset_index(name="median_price_per_m2")
    )
    quarterly = quarterly.merge(price_per_m2_median, on=["year", "quarter"])

    n_periods = len(quarterly)
    print(f"Quarters present: {n_periods} / 24")

    if n_periods < 24:
        present = set(zip(quarterly["year"], quarterly["quarter"]))
        expected = {(y, q) for y in EXPECTED_YEARS for q in EXPECTED_QUARTERS}
        missing = sorted(expected - present)
        print(f"WARNING - missing quarters: {missing}")
    else:
        print("All 24 quarters (2020 Q1 - 2025 Q4) are present.")

    print(quarterly[["period", "transaction_count", "median_price"]].to_string(index=False))

    quarterly.to_csv(TABLES_DIR / "quarterly_transaction_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(quarterly["period"], quarterly["transaction_count"], marker="o", color="#4C72B0")
    ax.set_title("Quarterly Transaction Volume, 2020 Q1 - 2025 Q4")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Transaction Count")
    ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quarterly_transaction_volume.png", dpi=150)
    plt.close(fig)

    return quarterly


# ============================================================
# 6. YEARLY SUMMARY
# ============================================================

def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("6. YEARLY MARKET TRENDS")
    print("=" * 60)

    yearly = (
        df.groupby("year")["price"]
        .agg(transaction_count="count", total_value="sum", median_price="median")
        .reset_index()
        .sort_values("year")
    )

    price_per_m2_median = (
        df.groupby("year")["price_per_m2_calculated"]
        .median()
        .reset_index(name="median_price_per_m2")
    )
    yearly = yearly.merge(price_per_m2_median, on="year")

    missing_years = sorted(set(EXPECTED_YEARS) - set(yearly["year"]))
    if missing_years:
        print(f"WARNING - missing years: {missing_years}")
    else:
        print("All expected years (2020-2025) are present.")

    print(yearly.to_string(index=False))

    yearly.to_csv(TABLES_DIR / "yearly_market_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(yearly["year"].astype(str), yearly["transaction_count"], color="#4C72B0")
    axes[0].set_title("Yearly Transaction Count")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Transaction Count")

    axes[1].plot(yearly["year"].astype(str), yearly["median_price"], marker="o", color="#DD8452")
    axes[1].set_title("Yearly Median Transaction Price")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Median Price (SAR)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "yearly_market_trends.png", dpi=150)
    plt.close(fig)

    return yearly


# ============================================================
# 7. REGION ANALYSIS
# ============================================================

def analyze_regions(df: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("7. REGION ANALYSIS")
    print("=" * 60)

    regional = (
        df.groupby("region")["price"]
        .agg(transaction_count="count", median_price="median")
        .reset_index()
    )

    regional["pct_of_total"] = (
        regional["transaction_count"] / len(df) * 100
    ).round(2)

    price_per_m2_median = (
        df.groupby("region")["price_per_m2_calculated"]
        .median()
        .reset_index(name="median_price_per_m2")
    )
    regional = regional.merge(price_per_m2_median, on="region")
    regional = regional.sort_values("transaction_count", ascending=False).reset_index(drop=True)

    print(regional.to_string(index=False))

    regional.to_csv(TABLES_DIR / "regional_summary.csv", index=False)

    # Region x year volume, printed for reference (not saved separately --
    # it is derivable from the raw data and isn't in the requested output list)
    region_year = df.groupby(["region", "year"]).size().unstack(fill_value=0)
    print("\nTransaction volume by region and year:")
    print(region_year.to_string())

    fig, ax = plt.subplots(figsize=(9, 6))
    plot_data = regional.sort_values("median_price", ascending=True)
    ax.barh(plot_data["region"], plot_data["median_price"], color="#55A868")
    ax.set_title("Median Transaction Price by Region")
    ax.set_xlabel("Median Price (SAR)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "median_price_by_region.png", dpi=150)
    plt.close(fig)

    return regional


# ============================================================
# 8. PROPERTY TYPE ANALYSIS (broad use category)
# ============================================================

def analyze_property_types(df: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("8. PROPERTY TYPE ANALYSIS (property_type -- broad use category)")
    print("=" * 60)

    categories = sorted(df["property_type"].dropna().unique())
    coverage_pct = 100 - df["property_type"].isna().mean() * 100
    print(
        f"property_type ({' / '.join(categories)}) is populated for "
        f"{coverage_pct:.1f}% of rows, all years 2020-2025."
    )

    property_type = (
        df.groupby("property_type")["price"]
        .agg(transaction_count="count", median_price="median")
        .reset_index()
    )

    property_type["pct_of_total"] = (
        property_type["transaction_count"] / len(df) * 100
    ).round(2)

    property_type = property_type.sort_values(
        "transaction_count", ascending=False
    ).reset_index(drop=True)

    print(property_type.to_string(index=False))

    top_share = property_type["pct_of_total"].iloc[0]
    print(
        f"\nDataset is dominated by '{property_type['property_type'].iloc[0]}' "
        f"({top_share:.1f}% of transactions) -- strongly imbalanced."
    )

    property_type.to_csv(TABLES_DIR / "property_type_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data = property_type.sort_values("median_price", ascending=True)
    ax.barh(plot_data["property_type"], plot_data["median_price"], color="#C44E52")
    ax.set_title("Median Transaction Price by Property Type")
    ax.set_xlabel("Median Price (SAR)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "median_price_by_property_type.png", dpi=150)
    plt.close(fig)

    return property_type


# ============================================================
# 9. YEAR x QUARTER MEDIAN PRICE HEATMAP
# ============================================================

def price_quarter_heatmap(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("9. MEDIAN PRICE BY YEAR AND QUARTER")
    print("=" * 60)

    pivot = df.pivot_table(
        index="year", columns="quarter", values="price", aggfunc="median"
    )

    print(pivot.round(0).to_string())

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="YlOrRd", ax=ax)
    ax.set_title("Median Transaction Price: Year x Quarter")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Year")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quarter_year_heatmap.png", dpi=150)
    plt.close(fig)


# ============================================================
# 10. PRICE PER SQUARE METER
# ============================================================

def price_per_sqm_analysis(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("10. PRICE PER SQUARE METER (EDA / market analysis only)")
    print("=" * 60)
    print(
        "price_per_m2_calculated = price / area. Use for EDA/market "
        "analysis only -- NEVER as an input feature for price prediction, "
        "since it is derived from the target and would leak it."
    )

    by_region = (
        df.groupby("region")["price_per_m2_calculated"]
        .median()
        .sort_values(ascending=False)
    )
    print("\nMedian price per m2 by region:")
    print(by_region.round(1).to_string())

    by_year = df.groupby("year")["price_per_m2_calculated"].median()
    print("\nMedian price per m2 by year:")
    print(by_year.round(1).to_string())


# ============================================================
# 11. FEATURE - TARGET RELATIONSHIPS
# ============================================================

def feature_target_relationships(df: pd.DataFrame, top_n_districts: int = 15) -> None:

    print("\n" + "=" * 60)
    print("11. FEATURE-TARGET RELATIONSHIPS (median price)")
    print("=" * 60)

    for col in ["region", "property_type", "year", "quarter"]:
        print(f"\nMedian price by {col}:")
        print(df.groupby(col)["price"].median().round(0).to_string())

    top_districts = df["city_district"].value_counts().head(top_n_districts).index
    district_prices = (
        df[df["city_district"].isin(top_districts)]
        .groupby("city_district")["price"]
        .agg(transaction_count="count", median_price="median")
        .sort_values("transaction_count", ascending=False)
    )
    print(f"\nMedian price for top {top_n_districts} districts by volume:")
    print(district_prices.round(0).to_string())

    area_corr = df["area"].corr(df["price"])
    print(f"\nCorrelation(area, price): {area_corr:.3f}")


# ============================================================
# 12. CORRELATION MATRIX
# ============================================================

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("12. CORRELATION MATRIX")
    print("=" * 60)

    # reference_number is an identifier, not a meaningful numeric variable,
    # so it is excluded.
    numeric_cols = ["price", "area", "price_per_m2_calculated", "n_properties", "year", "quarter"]
    corr = df[numeric_cols].corr()

    print(corr.round(2).to_string())

    return corr


# ============================================================
# VALIDATION CHECKS
# ============================================================

def run_validation_checks(
    df: pd.DataFrame, quarterly: pd.DataFrame, yearly: pd.DataFrame
) -> None:

    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)

    def check(label: str, condition: bool) -> None:
        print(f"[{'PASS' if condition else 'FAIL'}] {label}")

    check(
        "Using processed data (data/processed), not raw",
        "processed" in str(PROCESSED_PATH),
    )

    check(
        "Quarterly counts sum to dataset total",
        quarterly["transaction_count"].sum() == len(df),
    )

    check(
        "Yearly counts sum to dataset total",
        yearly["transaction_count"].sum() == len(df),
    )

    check(
        "Years 2020-2025 all present",
        set(EXPECTED_YEARS) <= set(df["year"].unique()),
    )

    check(
        "Quarters Q1-Q4 all present",
        set(EXPECTED_QUARTERS) <= set(df["quarter"].unique()),
    )

    check(
        "All 24 (year, quarter) combinations present",
        len(quarterly) == 24,
    )


# ============================================================
# KEY FINDINGS
# ============================================================

def print_key_findings(
    df: pd.DataFrame,
    yearly: pd.DataFrame,
    regional: pd.DataFrame,
    property_type_summary: pd.DataFrame,
) -> None:

    median_price = df["price"].median()
    mean_price = df["price"].mean()
    area_price_corr = df["area"].corr(df["price"])

    top_regions = (
        regional.sort_values("transaction_count", ascending=False)["region"]
        .head(3)
        .tolist()
    )

    dominant_row = property_type_summary.sort_values(
        "transaction_count", ascending=False
    ).iloc[0]
    dominant_type = dominant_row["property_type"]
    dominant_share = dominant_row["pct_of_total"]

    yearly_sorted = yearly.sort_values("year")
    peak_row = yearly_sorted.loc[yearly_sorted["median_price"].idxmax()]
    latest_row = yearly_sorted.iloc[-1]
    peak_year = int(peak_row["year"])
    latest_year = int(latest_row["year"])
    trend_note = (
        f"peaked in {peak_year} and are currently below that peak as of {latest_year}"
        if latest_row["median_price"] < peak_row["median_price"]
        else f"reached their highest point in {latest_year}"
    )

    print("\n" + "=" * 60)
    print("EDA KEY FINDINGS / REFLECTION")
    print("=" * 60)
    print(f"""
1. Transaction prices are highly right-skewed. The median price
   ({median_price:,.0f} SAR) is more representative of a typical
   transaction than the mean ({mean_price:,.0f} SAR).

2. Median transaction prices {trend_note}.

3. {", ".join(top_regions)} account for the majority of transaction
   activity.

4. {dominant_type} transactions dominate the dataset, representing
   approximately {dominant_share:.1f}% of transactions.

5. Transaction prices vary substantially across regions and districts,
   suggesting that location is an important factor for price prediction.

6. Property area has a weak linear correlation with price
   (r = {area_price_corr:.3f}). However, this does not necessarily mean
   area is unimportant because its relationship with price may be
   nonlinear and interact with location and property type.

7. Price and area contain extreme values that should be investigated
   before final modeling decisions are made rather than automatically
   removed.

8. price_per_m2_calculated is useful for EDA and market analysis only
   and must not be used as a predictor of price because it is derived
   from the target.
""")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading cleaned dataset from: {PROCESSED_PATH}")
    df = load_clean_data()
    print(f"Usable rows: {len(df):,}")

    dataset_overview(df)
    analyze_price(df)
    analyze_area(df)
    price_vs_area(df)

    quarterly = quarterly_summary(df)
    yearly = yearly_summary(df)

    regional = analyze_regions(df)
    property_type_summary = analyze_property_types(df)

    price_quarter_heatmap(df)
    price_per_sqm_analysis(df)
    feature_target_relationships(df)
    correlation_analysis(df)

    run_validation_checks(df, quarterly, yearly)
    print_key_findings(df, yearly, regional, property_type_summary)

    print("\nEDA complete.")
    print(f"Tables saved to:  {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
