"""
MOJ Sales Data Cleaning and Merging Pipeline
2020 Q1 - 2025 Q4

Pipeline order:
 1. Extract the raw ZIP file.
 2. Discover and validate the 24 quarterly files.
 3. Load each file using its appropriate schema.
 4. Standardize each file into a common intermediate schema.
 5. Concatenate the 24 standardized DataFrames.
 6. Record and report the raw merged row count.
 7. Harmonize and clean the common fields.
 8. Apply the global row filters after merging.
 9. Remove redundant and excluded columns.
10. Run final validation checks.
11. Save the final processed dataset.
12. Print one concise final cleaning report.
"""

import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from data_io import SUSPICIOUS_GROUP_COLS, find_suspicious_price_groups  # noqa: E402

# ============================================================
# CONFIG / CONSTANTS
# ============================================================

VERBOSE = False  # set True to also print a preview of the raw merged table

EXPECTED_N_QUARTERLY_FILES = 24
EXPECTED_YEARS = list(range(2020, 2026))
EXPECTED_QUARTERS = [1, 2, 3, 4]

# Required transaction fields: the final processed dataset must have zero
# missing values in every one of these columns.
REQUIRED_TRANSACTION_COLUMNS = [
    "price", "area", "transaction_date", "region_en", "property_type",
]

# property_type categories excluded for insufficient sample size (see
# harmonize_property_type() / USE_CATEGORY_MAP -- these are the only
# values that column can ever take besides the retained categories).
EXCLUDED_PROPERTY_TYPES = ["Industrial", "Mixed Use", "Other"]


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACTED_DIR = RAW_DIR / "extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

ZIP_PATH = RAW_DIR / "MOJ Dataset.zip"
REGION_MAP_PATH = TABLES_DIR / "region_mapping.csv"

OUTPUT_PATH = PROCESSED_DIR / "moj_sales_2020_2025_merged.csv.gz"


# ============================================================
# PROPERTY FIELD HARMONIZATION
# ============================================================
#
# The source files carry two different Arabic columns:
#   - "تصنيف العقار" -- present in ALL 24 quarters. Its values are broad
#     USE categories (Residential/Commercial/Agricultural/...). These are
#     harmonized below into the final "property_type" column.
#   - "نوع العقار"    -- present ONLY in 2023 Q1-Q3. Its values are
#     physical property FORMS (Villa/Land/Apartment/...). The project has
#     decided not to retain a harmonized physical-form column, so this
#     field is loaded (for schema consistency across quarters) and then
#     dropped as unused intermediate data -- see FINAL_OUTPUT_COLUMNS.
#
# The loader functions below name these raw columns "property_classification"
# and "property_type" respectively, by literally translating the Arabic
# header text. That naming is semantically backwards from how this project
# wants to use the term "property_type", so after the merge the raw columns
# are renamed and the final "property_type" is rebuilt from the map below.

# "تصنيف العقار" values -> broad USE category -> final "property_type"
USE_CATEGORY_MAP = {
    "سكني": "Residential",
    "تجاري": "Commercial",
    "زراعي": "Agricultural",
    "صناعي": "Industrial",
    "سكني تجاري": "Mixed Use",
    "أخرى": "Other",
}


# ============================================================
# RAW INTERMEDIATE SCHEMA (pre-harmonization)
# ============================================================

UNIFIED_COLUMNS = [
    "region_raw",
    "city",
    "city_district",
    "reference_number",
    "date_gregorian_raw",
    "date_hijri",
    "property_classification",
    "property_type",
    "plan_number",
    "plot_number",
    "n_properties",
    "price_raw",
    "area_raw",
    "price_per_m2_reported",
    "source_file",
    "year",
    "quarter",
]

# Columns kept in the final processed dataset. This is an allowlist --
# anything not listed here (raw/intermediate helper columns such as
# price_raw, area_raw, use_category_raw, physical_form_raw,
# region_raw_stripped, plan_number, plot_number, price_per_m2_reported,
# date_hijri, and property_classification) is dropped simply by not being
# selected. date_hijri is not needed -- transaction_date (cleaned from
# date_gregorian_raw) is the dataset's date column.
FINAL_OUTPUT_COLUMNS = [
    "region_raw",
    "city",
    "city_district",
    "reference_number",
    "date_gregorian_raw",
    "n_properties",
    "source_file",
    "year",
    "quarter",
    "property_type",
    "price",
    "area",
    "transaction_date",
    "region_en",
    "price_per_m2_calculated",
]


# ============================================================
# FILE LOADERS
# ============================================================

def parse_year_quarter(filename: str):
    match = re.search(r"(\d{4})-Q(\d)", filename)

    if not match:
        raise ValueError(
            f"Could not extract year/quarter from filename: {filename}"
        )

    return int(match.group(1)), int(match.group(2))


def load_standard_10col(path: Path, year: int, quarter: int):

    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    out = pd.DataFrame({
        "region_raw": df["المنطقة"],
        "city": df["المدينة"],
        "city_district": df["المدينة / الحي"],
        "reference_number": df["الرقم المرجعي للصفقة"],
        "date_gregorian_raw": df["تاريخ الصفقة ميلادي"],
        "date_hijri": df["تاريخ الصفقة هجري"],
        "property_classification": df["تصنيف العقار"],
        "property_type": pd.NA,
        "plan_number": pd.NA,
        "plot_number": pd.NA,
        "n_properties": df["عدد العقارات"],
        "price_raw": df["السعر"],
        "area_raw": df["المساحة"],
        "price_per_m2_reported": pd.NA,
    })

    out["source_file"] = path.name
    out["year"] = year
    out["quarter"] = quarter

    return out


def load_2023_q2_q3_11col(path: Path, year: int, quarter: int):

    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    out = pd.DataFrame({
        "region_raw": df["المنطقة"],
        "city": df["المدينة"],
        "city_district": df["المدينة / الحي"],
        "reference_number": df["الرقم المرجعي للصفقة"],
        "date_gregorian_raw": df["تاريخ الصفقة ميلادي"],
        "date_hijri": df["تاريخ الصفقة هجري"],
        "property_classification": df["تصنيف العقار"],
        "property_type": df["نوع العقار"],
        "plan_number": pd.NA,
        "plot_number": pd.NA,
        "n_properties": df["عدد العقارات"],
        "price_raw": df["السعر"],
        "area_raw": df["المساحة"],
        "price_per_m2_reported": pd.NA,
    })

    out["source_file"] = path.name
    out["year"] = year
    out["quarter"] = quarter

    return out


def load_2023_q1_13col(path: Path, year: int, quarter: int):

    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    out = pd.DataFrame({
        "region_raw": df["المنطقة"],
        "city": df["المدينة"],
        "city_district": df["الحي"],
        "reference_number": df["رقم مرجعي"],
        "date_gregorian_raw": df["التاريخ"],
        "date_hijri": pd.NA,
        "property_classification": df["تصنيف العقار"],
        "property_type": df["نوع العقار"],
        "plan_number": df["المخطط"],
        "plot_number": df["رقم القطعة"],
        "n_properties": df["عدد العقارات"],
        "price_raw": df["السعر بالريال السعودي"],
        "area_raw": df["المساحة"],
        "price_per_m2_reported": df["سعر المتر المربع"],
    })

    out["source_file"] = path.name
    out["year"] = year
    out["quarter"] = quarter

    return out


# ============================================================
# FIELD CLEANING HELPERS
# ============================================================

def clean_numeric(series: pd.Series) -> pd.Series:

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(cleaned, errors="coerce")


def parse_date_flexible(series: pd.Series) -> pd.Series:

    series = series.astype("string").str.strip()

    parsed = pd.to_datetime(series, format="%Y/%m/%d", errors="coerce")

    still_missing = parsed.isna() & series.notna()

    if still_missing.any():
        parsed2 = pd.to_datetime(
            series[still_missing], format="%m/%d/%Y", errors="coerce"
        )
        parsed.loc[still_missing] = parsed2

    return parsed


def load_region_mapping():
    """Return {variant: canonical_en} or None if region_mapping.csv is missing."""

    if not REGION_MAP_PATH.exists():
        return None

    region_map = pd.read_csv(REGION_MAP_PATH, encoding="utf-8-sig")

    required_columns = {"variant", "canonical_en"}
    missing_columns = required_columns - set(region_map.columns)

    if missing_columns:
        raise ValueError(
            f"region_mapping.csv is missing columns: {missing_columns}"
        )

    return dict(zip(region_map["variant"], region_map["canonical_en"]))


# ============================================================
# STEP 1-4: LOAD + STANDARDIZE ALL QUARTERLY FILES
# ============================================================

def load_and_standardize_files():
    """Extract the ZIP, discover the quarterly files, and load each one
    into the common intermediate schema (UNIFIED_COLUMNS)."""

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP file not found:\n{ZIP_PATH}")

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACTED_DIR)

    files = sorted(
        path for path in EXTRACTED_DIR.rglob("*")
        if path.is_file()
        and re.match(r"MOJ-Sales-\d{4}-Q[1-4]\.csv(\.gz)?$", path.name)
    )

    if len(files) != EXPECTED_N_QUARTERLY_FILES:
        raise ValueError(
            f"Expected {EXPECTED_N_QUARTERLY_FILES} quarterly files, "
            f"but found {len(files)}."
        )

    frames = []
    schema_log = []

    for f in files:
        year, quarter = parse_year_quarter(f.name)

        if year == 2023 and quarter == 1:
            df = load_2023_q1_13col(f, year, quarter)
            schema = "2023-Q1 (13-col, custom)"
        elif year == 2023 and quarter in (2, 3):
            df = load_2023_q2_q3_11col(f, year, quarter)
            schema = "2023-Q2/Q3 (11-col)"
        else:
            df = load_standard_10col(f, year, quarter)
            schema = "standard (10-col)"

        schema_log.append((f.name, schema, len(df)))
        frames.append(df)

    return files, frames, schema_log


# ============================================================
# STEP 5: CONCATENATION
# ============================================================

def concatenate_frames(frames: list) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)[UNIFIED_COLUMNS]


# ============================================================
# STEP 7a: PROPERTY-CATEGORY HARMONIZATION
# ============================================================

def harmonize_property_type(merged: pd.DataFrame):
    """Rename the raw-extracted columns to neutral names, then rebuild the
    canonical 'property_type' column from use_category_raw."""

    n_rows_before = len(merged)

    merged = merged.rename(
        columns={
            "property_classification": "use_category_raw",
            "property_type": "physical_form_raw",
        }
    )

    merged["property_type"] = (
        merged["use_category_raw"].astype("string").str.strip().map(USE_CATEGORY_MAP)
    )

    unmapped_use_category = merged.loc[
        merged["use_category_raw"].notna() & merged["property_type"].isna(),
        "use_category_raw",
    ].unique().tolist()

    assert len(merged) == n_rows_before, (
        "Row count changed during property_type harmonization."
    )

    return merged, unmapped_use_category


# ============================================================
# STEP 7b: FIELD CLEANING (numeric, date, region, price_per_m2)
# ============================================================

def clean_fields(merged: pd.DataFrame):

    merged["price"] = clean_numeric(merged["price_raw"])
    merged["area"] = clean_numeric(merged["area_raw"])
    merged["n_properties"] = clean_numeric(merged["n_properties"])

    merged["transaction_date"] = parse_date_flexible(merged["date_gregorian_raw"])

    region_mapping = load_region_mapping()
    region_mapping_missing = region_mapping is None

    merged["region_raw_stripped"] = merged["region_raw"].astype("string").str.strip()

    if region_mapping is not None:
        merged["region_en"] = merged["region_raw_stripped"].map(region_mapping)
    else:
        merged["region_en"] = pd.NA

    # price_per_m2_calculated = price / area. Kept in the cleaned dataset
    # for EDA / market analysis only. It must NEVER be used as an input
    # feature (X) for a model that predicts price -- doing so would leak
    # the target into the features. area <= 0 is treated as NaN (never
    # inf) via a safe divisor.
    safe_area = merged["area"].where(merged["area"] > 0)
    merged["price_per_m2_calculated"] = merged["price"] / safe_area

    return merged, region_mapping_missing


# ============================================================
# STEP 8: GLOBAL ROW FILTERS (applied after merging + cleaning)
# ============================================================

def apply_row_filters(merged: pd.DataFrame):

    stats = {}

    # --- Bulk transactions (n_properties > 1) ---
    # Exclude bulk transactions because the project scope is limited
    # to predicting the transaction price of a single property.
    rows_before_bulk = len(merged)
    n_properties_missing_kept = int(merged["n_properties"].isna().sum())
    n_properties_non_positive_kept = int((merged["n_properties"] <= 0).sum())
    n_properties_single = int((merged["n_properties"] == 1).sum())

    bulk_mask = merged["n_properties"] > 1
    bulk_removed = int(bulk_mask.sum())
    merged = merged.loc[~bulk_mask].copy()

    stats["bulk"] = {
        "before": rows_before_bulk,
        "removed": bulk_removed,
        "after": len(merged),
        "n_properties_single": n_properties_single,
        "n_properties_missing_kept": n_properties_missing_kept,
        "n_properties_non_positive_kept": n_properties_non_positive_kept,
    }

    # --- Invalid price / area (price <= 0 or area <= 0) ---
    # A non-positive price or area is not a usable transaction (a data
    # error, not a legitimate value). NaN price/area is left alone here --
    # it is handled by the required-fields filter below.
    rows_before_price_area = len(merged)
    invalid_price_area_mask = (merged["price"] <= 0) | (merged["area"] <= 0)
    invalid_price_area_removed = int(invalid_price_area_mask.sum())
    merged = merged.loc[~invalid_price_area_mask].copy()

    stats["invalid_price_area"] = {
        "before": rows_before_price_area,
        "removed": invalid_price_area_removed,
        "after": len(merged),
    }

    # --- Required transaction fields missing ---
    # dropna(subset=REQUIRED_TRANSACTION_COLUMNS) removes a row when ANY of
    # these columns is missing (not only fully-blank rows) -- this covers
    # the handful of trailing blank rows in the source quarterly files
    # (every field NaN) as well as any row missing just one required field.
    rows_before_required = len(merged)
    merged = merged.dropna(subset=REQUIRED_TRANSACTION_COLUMNS).copy()
    required_fields_missing_removed = rows_before_required - len(merged)

    stats["required_fields_missing"] = {
        "before": rows_before_required,
        "removed": required_fields_missing_removed,
        "after": len(merged),
    }

    # --- Suspicious high-price transaction groups ---
    # Remove a group of rows sharing identical price/transaction_date/
    # region_en/city when: price is above the global P99, the group spans
    # more than one distinct reference_number, and max(area)/min(area) > 10
    # -- consistent with a single bulk contract's total value having been
    # duplicated onto multiple parcels' rows rather than priced per parcel.
    # Applied after the required-fields filter and before duplicate removal.
    group_cols = list(SUSPICIOUS_GROUP_COLS)

    rows_before_suspicious = len(merged)

    suspicious_groups = find_suspicious_price_groups(merged)
    n_suspicious_groups = len(suspicious_groups)

    group_keys = pd.Series(
        list(zip(*[merged[c] for c in group_cols])), index=merged.index
    )
    suspicious_key_set = set(zip(*[suspicious_groups[c] for c in group_cols]))
    suspicious_mask = group_keys.isin(suspicious_key_set)

    affected_rows = merged.loc[suspicious_mask].merge(
        suspicious_groups[group_cols + ["area_ratio"]],
        on=group_cols,
        how="left",
    )
    affected_rows["removal_reason"] = (
        "Suspicious high-price group: identical price/transaction_date/"
        "region_en/city shared by multiple reference_numbers, "
        "max(area)/min(area) > 10, price above the global P99."
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    affected_rows.to_csv(
        TABLES_DIR / "suspicious_transactions_removed.csv", index=False
    )

    suspicious_rows_removed = int(suspicious_mask.sum())
    merged = merged.loc[~suspicious_mask].copy()

    stats["suspicious_groups"] = {
        "before": rows_before_suspicious,
        "removed": suspicious_rows_removed,
        "after": len(merged),
        "n_groups": n_suspicious_groups,
    }

    # --- Duplicate rows ---
    # Computed AFTER the required-fields drop above, so shared NaNs across
    # the (now-removed) blank rows can no longer masquerade as "duplicates"
    # of each other under pandas' NaN-equality rule in .duplicated().
    dup_check_cols = [
        "region_raw", "city", "city_district", "reference_number",
        "date_gregorian_raw", "use_category_raw", "n_properties",
        "price_raw", "area_raw",
    ]

    full_row_duplicates_mask = merged.duplicated(subset=dup_check_cols, keep=False)
    n_full_row_duplicates_flagged = int(full_row_duplicates_mask.sum())

    ref_counts = merged["reference_number"].value_counts()
    duplicated_ref_numbers = ref_counts[ref_counts > 1]
    n_duplicated_ref_numbers = len(duplicated_ref_numbers)
    n_rows_with_duplicated_ref = int(duplicated_ref_numbers.sum())

    cross_file_dupes = 0
    if n_duplicated_ref_numbers > 0:
        dupe_rows = merged[merged["reference_number"].isin(duplicated_ref_numbers.index)]
        files_per_ref = dupe_rows.groupby("reference_number")["source_file"].nunique()
        cross_file_dupes = int((files_per_ref > 1).sum())

    # dup_check_cols / reference_number are re-checked here (not reused
    # from the keep=False masks above, which mark every member of a
    # duplicate group) with keep="first" so exactly one copy of each
    # genuine duplicate is kept.
    rows_before_duplicates = len(merged)
    is_full_row_duplicate = merged.duplicated(subset=dup_check_cols, keep="first")
    is_ref_duplicate = merged["reference_number"].notna() & merged.duplicated(
        subset=["reference_number"], keep="first"
    )
    merged = merged[~(is_full_row_duplicate | is_ref_duplicate)].copy()
    duplicate_rows_removed = rows_before_duplicates - len(merged)

    stats["duplicates"] = {
        "before": rows_before_duplicates,
        "removed": duplicate_rows_removed,
        "after": len(merged),
        "n_full_row_duplicates_flagged": n_full_row_duplicates_flagged,
        "n_duplicated_ref_numbers": n_duplicated_ref_numbers,
        "n_rows_with_duplicated_ref": n_rows_with_duplicated_ref,
        "cross_file_dupes": cross_file_dupes,
    }

    # --- Underrepresented property types ---
    # Exclude property types with insufficient observations for reliable
    # model training and evaluation. The project retains Residential,
    # Commercial, and Agricultural single-property transactions.
    property_type_counts_before = merged["property_type"].value_counts(dropna=False)

    rows_before_type_filter = len(merged)
    rare_type_mask = merged["property_type"].isin(EXCLUDED_PROPERTY_TYPES)
    rare_type_removed = int(rare_type_mask.sum())
    merged = merged.loc[~rare_type_mask].copy()

    property_type_counts_after = merged["property_type"].value_counts(dropna=False)

    stats["rare_property_type"] = {
        "before": rows_before_type_filter,
        "removed": rare_type_removed,
        "after": len(merged),
        "counts_before": property_type_counts_before,
        "counts_after": property_type_counts_after,
    }

    return merged, stats


# ============================================================
# STEP 9: FINAL COLUMN SELECTION
# ============================================================

def select_final_columns(merged: pd.DataFrame) -> pd.DataFrame:
    n_rows_before = len(merged)

    merged = merged[FINAL_OUTPUT_COLUMNS].copy()

    assert len(merged) == n_rows_before, (
        "Row count changed while selecting final columns -- this step must "
        "only select columns, never rows."
    )

    return merged


# ============================================================
# STEP 10: FINAL VALIDATION CHECKS
# ============================================================

def run_validation_checks(merged: pd.DataFrame, files: list, raw_merged_rows: int, stats: dict):

    results = []

    def check(label: str, condition: bool) -> None:
        results.append((label, bool(condition)))

    check("Exactly 24 source files processed", len(files) == EXPECTED_N_QUARTERLY_FILES)
    check("All years 2020-2025 represented", set(EXPECTED_YEARS) <= set(merged["year"].unique()))
    check(
        "All 24 year-quarter combinations represented",
        merged.groupby(["year", "quarter"]).ngroups == EXPECTED_N_QUARTERLY_FILES,
    )
    check(
        "Required transaction fields contain no missing values",
        merged[REQUIRED_TRANSACTION_COLUMNS].isna().sum().sum() == 0,
    )
    check("price > 0 for all rows", bool((merged["price"] > 0).all()))
    check("area > 0 for all rows", bool((merged["area"] > 0).all()))
    check("No rows with n_properties > 1", not (merged["n_properties"] > 1).any())
    check(
        "Excluded property types are absent",
        not merged["property_type"].isin(EXCLUDED_PROPERTY_TYPES).any(),
    )
    check("property_classification column is absent", "property_classification" not in merged.columns)
    check("price_per_m2_reported column is absent", "price_per_m2_reported" not in merged.columns)
    check(
        "price_per_m2_calculated contains no infinite values",
        not np.isinf(merged["price_per_m2_calculated"]).any(),
    )
    check("No final columns are duplicated", not merged.columns.duplicated().any())
    check(
        "No suspicious high-price groups remain",
        len(find_suspicious_price_groups(merged)) == 0,
    )

    total_removed = (
        stats["bulk"]["removed"]
        + stats["invalid_price_area"]["removed"]
        + stats["required_fields_missing"]["removed"]
        + stats["suspicious_groups"]["removed"]
        + stats["duplicates"]["removed"]
        + stats["rare_property_type"]["removed"]
    )
    check(
        "Final row count reconciles: raw_merged_rows - total_removed == final_rows",
        len(merged) == raw_merged_rows - total_removed,
    )

    failed = [label for label, passed in results if not passed]
    assert not failed, "Final validation check(s) failed: " + "; ".join(failed)

    return results, total_removed


# ============================================================
# STEP 11: SAVE
# ============================================================

def save_dataset(merged: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig", compression="gzip")


# ============================================================
# STEP 12: FINAL REPORT
# ============================================================

def print_final_report(
    files, schema_log, rows_loaded_total, raw_merged_rows, merged_preview,
    unmapped_use_category, region_mapping_missing, stats, validation_results,
    total_removed, final_merged,
):
    print("=" * 70)
    print("MOJ SALES DATA CLEANING -- FINAL REPORT")
    print("=" * 70)

    # --- 1. Input discovery ---
    print("\n--- 1. INPUT DISCOVERY ---")
    print(f"ZIP file: {ZIP_PATH}")
    print(f"Extracted to: {EXTRACTED_DIR}")
    print(f"Quarterly files found: {len(files)} / {EXPECTED_N_QUARTERLY_FILES}")
    for f in files:
        print(f"  {f.name}")

    if region_mapping_missing:
        print(
            "\nWARNING: region_mapping.csv not found -- region_en is NaN for "
            "every row until it is added."
        )

    # --- 2. Schema summary ---
    print("\n--- 2. SCHEMA SUMMARY ---")
    for name, schema, n in schema_log:
        print(f"{name:30s} -> {schema:30s} ({n:,} rows)")
    print(f"\nRows loaded from all source files: {rows_loaded_total:,}")

    if unmapped_use_category:
        print(f"\nWARNING: unmapped property_type source values (became NaN): {unmapped_use_category}")

    # --- 3. Raw merge summary ---
    print("\n--- 3. RAW MERGE SUMMARY ---")
    print(f"Rows immediately after concatenation: {raw_merged_rows:,}")

    if VERBOSE and merged_preview is not None:
        print("\nPreview of raw merged table (head):")
        print(merged_preview.to_string())

    # --- 4. Cleaning summary ---
    print("\n--- 4. CLEANING SUMMARY ---")

    b = stats["bulk"]
    print(
        f"Bulk transactions (n_properties > 1): "
        f"{b['before']:,} -> {b['after']:,} rows ({b['removed']:,} removed)"
    )
    print(
        f"  n_properties == 1: {b['n_properties_single']:,} | "
        f"missing (kept for review): {b['n_properties_missing_kept']:,} | "
        f"<= 0 (kept for review): {b['n_properties_non_positive_kept']:,}"
    )

    p = stats["invalid_price_area"]
    print(
        f"\nInvalid price/area (price <= 0 or area <= 0): "
        f"{p['before']:,} -> {p['after']:,} rows ({p['removed']:,} removed)"
    )

    r = stats["required_fields_missing"]
    print(
        f"\nRows removed because one or more required transaction fields "
        f"were missing ({', '.join(REQUIRED_TRANSACTION_COLUMNS)}): "
        f"{r['before']:,} -> {r['after']:,} rows ({r['removed']:,} removed)"
    )

    s = stats["suspicious_groups"]
    print(
        f"\nSuspicious high-price groups (identical price/transaction_date/"
        f"region_en/city, >1 reference_number, area ratio > 10, price above "
        f"global P99): {s['before']:,} -> {s['after']:,} rows "
        f"({s['removed']:,} removed across {s['n_groups']} group(s))"
    )

    d = stats["duplicates"]
    print(
        f"\nDuplicate rows (full-row business-key duplicates or repeated "
        f"reference_number, keeping the first occurrence): "
        f"{d['before']:,} -> {d['after']:,} rows ({d['removed']:,} removed)"
    )
    print(
        f"  Rows flagged as full-row duplicates: {d['n_full_row_duplicates_flagged']:,} | "
        f"distinct reference_numbers repeated: {d['n_duplicated_ref_numbers']:,} "
        f"({d['n_rows_with_duplicated_ref']:,} rows) | "
        f"repeated across >1 file: {d['cross_file_dupes']:,}"
    )

    t = stats["rare_property_type"]
    print(
        f"\nUnderrepresented property types ({', '.join(EXCLUDED_PROPERTY_TYPES)}): "
        f"{t['before']:,} -> {t['after']:,} rows ({t['removed']:,} removed)"
    )
    print("\nproperty_type distribution (before -> after):")
    comparison = pd.DataFrame({
        "before": t["counts_before"],
        "after": t["counts_after"],
    }).fillna(0).astype(int)
    print(comparison.to_string())

    print("\nMissing values by year (right after field cleaning, before row filters):")
    print(stats["missing_by_year"].to_string())

    # --- 5. Final data-quality validation ---
    print("\n--- 5. FINAL DATA-QUALITY VALIDATION ---")
    for label, passed in validation_results:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")

    print(
        f"\nReconciliation: {raw_merged_rows:,} (raw merged) - "
        f"{total_removed:,} (total removed across all filters) = "
        f"{raw_merged_rows - total_removed:,} (matches final rows)"
    )

    # --- 6. Final dataset summary ---
    print("\n--- 6. FINAL DATASET SUMMARY ---")
    print(f"Final rows: {len(final_merged):,}")
    print(f"Final columns ({len(final_merged.columns)}): {final_merged.columns.tolist()}")
    print("\nRows per year:")
    print(final_merged.groupby("year").size().to_string())

    # --- 7. Output location ---
    print("\n--- 7. OUTPUT LOCATION ---")
    print(OUTPUT_PATH)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    files, frames, schema_log = load_and_standardize_files()
    rows_loaded_total = sum(n for _, _, n in schema_log)

    merged = concatenate_frames(frames)
    raw_merged_rows = len(merged)
    merged_preview = merged.head() if VERBOSE else None

    merged, unmapped_use_category = harmonize_property_type(merged)
    merged, region_mapping_missing = clean_fields(merged)

    # Missing-value counts by year, taken right after cleaning and before
    # any row filters (equivalent to the old groupby().apply(isna().sum())
    # but without triggering pandas' groupby-apply FutureWarning).
    missing_by_year = merged.isna().groupby(merged["year"]).sum()

    merged, stats = apply_row_filters(merged)
    stats["missing_by_year"] = missing_by_year[
        ["price", "area", "transaction_date", "region_en"]
    ]

    merged = select_final_columns(merged)

    validation_results, total_removed = run_validation_checks(
        merged, files, raw_merged_rows, stats
    )

    save_dataset(merged)

    print_final_report(
        files, schema_log, rows_loaded_total, raw_merged_rows, merged_preview,
        unmapped_use_category, region_mapping_missing, stats, validation_results,
        total_removed, merged,
    )


if __name__ == "__main__":
    main()
