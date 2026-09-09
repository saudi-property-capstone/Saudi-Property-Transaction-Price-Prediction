"""
Outlier Investigation for Price Prediction
MOJ Sales, 2020-2025 (cleaned/processed dataset)

Investigates whether extreme price/area transactions are invalid data,
legitimate extreme transactions, bulk transactions, or suspicious records --
and tests, experimentally, whether excluding them actually helps a baseline
model generalize. Nothing is permanently removed from the cleaned dataset;
all filtering here is in-memory and scoped to this script's experiments.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

sys.path.append(str(Path(__file__).resolve().parent))
from EDA import load_clean_data  # noqa: E402  (reuses the same load + empty-row drop as EDA.py)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

FEATURE_COLS = ["area", "n_properties", "year", "quarter", "region", "property_type"]
CATEGORICAL_COLS = ["region", "property_type"]

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
    }

    report = pd.Series(checks, name="row_count")
    report_pct = (report / n * 100).round(4)

    summary = pd.DataFrame({"row_count": report, "pct_of_dataset": report_pct})
    print(summary.to_string())

    print(
        "\nNote: the dataset's 12 fully-empty footer rows (all core fields "
        "NaN) are already dropped by load_clean_data(), so they do not "
        "appear in these counts."
    )

    return report


# ============================================================
# TASK 2 & 3 -- TOP EXTREME PRICE / AREA TRANSACTIONS
# ============================================================

CONTEXT_COLS = [
    "price", "area", "n_properties", "property_type", "property_classification",
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

    print("\n--- Are extreme rows associated with bulk / specific segments? ---")

    n_extreme = len(extreme_df)
    bulk_share = (extreme_df["n_properties"] > 1).mean() * 100
    baseline_bulk_share = (df["n_properties"] > 1).mean() * 100
    print(
        f"n_properties > 1 among extreme rows: {bulk_share:.1f}% "
        f"(vs {baseline_bulk_share:.1f}% dataset-wide)"
    )

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
            "records n_properties = 1."
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
        "value rather than legitimate per-parcel pricing. The remaining clusters "
        "(similar areas, often round prices) look like standardized subdivision "
        "plot pricing, which is a normal market pattern."
    )

    return suspicious.reset_index()


# ============================================================
# TASK 4 -- BULK VS SINGLE-PROPERTY TRANSACTIONS
# ============================================================

def bulk_vs_single_comparison(df: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("TASK 4: BULK (n_properties > 1) VS SINGLE-PROPERTY TRANSACTIONS")
    print("=" * 60)

    df = df.copy()
    df["bulk"] = np.where(df["n_properties"] > 1, "bulk (>1)", "single (=1)")

    percentiles = [0.5, 0.95, 0.99, 0.999]

    summary = df.groupby("bulk").agg(
        transaction_count=("price", "count"),
        median_price=("price", "median"),
        mean_price=("price", "mean"),
        median_area=("area", "median"),
        mean_area=("area", "mean"),
    ).round(1)

    print(summary.to_string())

    print("\nPrice percentiles by group:")
    print(df.groupby("bulk")["price"].quantile(percentiles).unstack().round(0).to_string())

    print("\nArea percentiles by group:")
    print(df.groupby("bulk")["area"].quantile(percentiles).unstack().round(1).to_string())

    p99_9 = df["price"].quantile(0.999)
    bulk_share_of_extreme = (df.loc[df["price"] > p99_9, "n_properties"] > 1).mean() * 100
    print(
        f"\nOf rows above the global 99.9th price percentile ({p99_9:,.0f} SAR), "
        f"{bulk_share_of_extreme:.1f}% are bulk transactions (n_properties > 1)."
    )

    return summary


# ============================================================
# TASK 5 -- GROUP-LEVEL OUTLIER DIAGNOSTICS
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
    print("TASK 5: PRICE DISTRIBUTION WITHIN SEGMENTS (diagnostic, not deletion)")
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
        "transaction beyond its own segment's fence is not automatically "
        "removed, since it may be normal for that market (e.g. Makkah "
        "commercial vs Northern Borders residential)."
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
# TASK 6 & 7 -- MODEL EXPERIMENTS
# ============================================================

def prepare_temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    print("\n" + "=" * 60)
    print("TASK 6: TEMPORAL TRAIN/TEST SPLIT")
    print("=" * 60)

    # Basic invalid-row cleanup (Dataset A vs B distinction from the proposal
    # collapses into this single step -- only 1 row nationwide has area <= 0,
    # and it falls in the training period, so there is no separate "B" model).
    valid = df[(df["price"] > 0) & (df["area"] > 0)].copy()
    n_dropped = len(df) - len(valid)
    print(f"Dropped {n_dropped} additional invalid row(s) (price<=0 or area<=0).")

    for col in CATEGORICAL_COLS:
        valid[col] = valid[col].astype("category")

    train = valid[valid["year"] <= 2024].copy()
    test = valid[valid["year"] == 2025].copy()

    print(f"Train (2020-2024): {len(train):,} rows")
    print(f"Test  (2025):      {len(test):,} rows (fixed across all experiments below)")

    return train, test


def evaluate(model, X_test: pd.DataFrame, y_test_raw: pd.Series, p95_train: float, log_target: bool) -> dict:

    pred = model.predict(X_test)
    if log_target:
        pred = np.expm1(pred)

    def metrics(mask):
        y_true = y_test_raw[mask]
        y_pred = pred[mask.to_numpy()]
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "n_rows": int(mask.sum()),
        }

    full_mask = pd.Series(True, index=y_test_raw.index)
    normal_mask = y_test_raw <= p95_train
    high_mask = y_test_raw > p95_train

    return {
        "full": metrics(full_mask),
        "normal_price_(<=train_P95)": metrics(normal_mask),
        "high_price_(>train_P95)": metrics(high_mask),
    }


def run_model_experiments(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("TASK 6 & 7: MODEL EXPERIMENTS (HistGradientBoostingRegressor)")
    print("=" * 60)
    print(
        "Same fixed test set (2025) used for every experiment below -- only "
        "the training data or target transform changes. price_per_m2_calculated "
        "is never used as a feature (target leakage)."
    )

    X_test = test[FEATURE_COLS]
    y_test = test["price"]

    p95_train = train["price"].quantile(0.95)
    p999_train = train["price"].quantile(0.999)
    print(f"\nTraining-derived P95 (normal/high split): {p95_train:,.0f} SAR")
    print(f"Training-derived P99.9 (experimental trim threshold): {p999_train:,.0f} SAR")

    experiments = {}

    # --- Experiment 1: full valid training data, raw price target ---
    X_train = train[FEATURE_COLS]
    y_train = train["price"]
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42)
    model.fit(X_train, y_train)
    experiments["baseline_full_train_raw_price"] = {
        "n_train_rows": len(train),
        **evaluate(model, X_test, y_test, p95_train, log_target=False),
    }

    # --- Experiment 2: training data with price > P99.9 (train-derived) excluded ---
    # Experimental only -- this filtering is never applied to the cleaned dataset,
    # only to this in-memory training copy.
    trimmed_train = train[train["price"] <= p999_train]
    X_train_trim = trimmed_train[FEATURE_COLS]
    y_train_trim = trimmed_train["price"]
    model_trim = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42)
    model_trim.fit(X_train_trim, y_train_trim)
    experiments["p999_trimmed_train_raw_price"] = {
        "n_train_rows": len(trimmed_train),
        **evaluate(model_trim, X_test, y_test, p95_train, log_target=False),
    }

    # --- Experiment 3: full valid training data, log1p(price) target ---
    y_train_log = np.log1p(y_train)
    model_log = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42)
    model_log.fit(X_train, y_train_log)
    experiments["baseline_full_train_log1p_price"] = {
        "n_train_rows": len(train),
        **evaluate(model_log, X_test, y_test, p95_train, log_target=True),
    }

    rows = []
    for exp_name, result in experiments.items():
        n_train_rows = result["n_train_rows"]
        for subset_name in ["full", "normal_price_(<=train_P95)", "high_price_(>train_P95)"]:
            m = result[subset_name]
            rows.append({
                "experiment": exp_name,
                "n_train_rows": n_train_rows,
                "test_subset": subset_name,
                "n_test_rows": m["n_rows"],
                "mae": round(m["mae"], 2),
                "rmse": round(m["rmse"], 2),
                "r2": round(m["r2"], 4),
            })

    comparison = pd.DataFrame(rows)
    print("\n" + comparison.to_string(index=False))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(TABLES_DIR / "outlier_model_comparison.csv", index=False)

    return comparison


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

    bulk_vs_single_comparison(df)

    group_level_diagnostics(df)
    outlier_distribution_by_segment(df)

    train, test = prepare_temporal_split(df)
    run_model_experiments(train, test)

    print("\nOutlier analysis complete.")
    print(f"Tables saved to: {TABLES_DIR}")
    print("The cleaned dataset (data/processed/*.csv.gz) was not modified.")


if __name__ == "__main__":
    main()
