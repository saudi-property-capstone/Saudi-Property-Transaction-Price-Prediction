"""
MOJ Sales Data Cleaning and Merging Pipeline
2020 Q1 - 2025 Q4

Steps:
1. Extract raw ZIP automatically
2. Load all quarterly MOJ sales files
3. Handle different schemas
4. Clean numeric and date fields
5. Normalize region names
6. Create calculated price per m2 for EDA
7. Check missing values and duplicates
8. Save the merged dataset as compressed CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import re


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACTED_DIR = RAW_DIR / "extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ZIP_PATH = RAW_DIR / "MOJ Dataset.zip"
REGION_MAP_PATH = RAW_DIR / "region_mapping.csv"

OUTPUT_PATH = PROCESSED_DIR / "moj_sales_2020_2025_merged.csv.gz"


# ============================================================
# PROPERTY FIELD HARMONIZATION
# ============================================================
#
# The source files carry two different Arabic columns:
#   - "تصنيف العقار" -- present in ALL 24 quarters. Its values are broad
#     USE categories (Residential/Commercial/Agricultural/...).
#   - "نوع العقار"    -- present ONLY in 2023 Q1-Q3. Its values are
#     physical property FORMS (Villa/Land/Apartment/...).
#
# The loader functions below originally named these "property_classification"
# and "property_type" respectively, by literally translating the Arabic
# header text. That naming is semantically backwards from how this project
# wants to use the two English terms, so after the merge we rename the raw
# columns and rebuild "property_type" / "property_classification" from the
# canonical maps below.

# "تصنيف العقار" values -> broad USE category -> final "property_type"
USE_CATEGORY_MAP = {
    "سكني": "Residential",
    "تجاري": "Commercial",
    "زراعي": "Agricultural",
    "صناعي": "Industrial",
    "سكني تجاري": "Mixed Use",
    "أخرى": "Other",
}

# "نوع العقار" values -> physical FORM -> final "property_classification"
PHYSICAL_FORM_MAP = {
    "قطعة أرض": "Land",
    "شقة": "Apartment",
    "أرض زراعية": "Agricultural Land",
    "بيت": "House",
    "معرض/محل": "Showroom/Shop",
    "فيلا": "Villa",
    "مرفق": "Facility",
    "عمارة": "Building",
    "مركز تجاري": "Commercial Center",
    "إستراحة": "Resthouse",
    "قصر": "Palace",
}


# ============================================================
# FINAL UNIFIED COLUMNS
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


# ============================================================
# EXTRACT ZIP
# ============================================================

def extract_zip():
    """Extract the MOJ Dataset ZIP into data/raw/extracted."""

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"ZIP file not found:\n{ZIP_PATH}"
        )

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ZIP extraction ===")
    print(f"ZIP file: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACTED_DIR)

    print(f"Extracted to: {EXTRACTED_DIR}")
    print()


# ============================================================
# FIND SALES FILES
# ============================================================

def find_sales_files():
    """
    Find all MOJ-Sales quarterly files recursively.
    Handles both .csv and .csv.gz.
    """

    files = []

    for path in EXTRACTED_DIR.rglob("*"):
        if not path.is_file():
            continue

        if re.match(r"MOJ-Sales-\d{4}-Q[1-4]\.csv(\.gz)?$", path.name):
            files.append(path)

    files = sorted(files)

    print("=== Sales files found ===")
    print(f"Number of files: {len(files)}")

    for f in files:
        print(f"  {f.name}")

    print()

    if len(files) != 24:
        raise ValueError(
            f"Expected 24 quarterly files, but found {len(files)}."
        )

    return files


# ============================================================
# YEAR / QUARTER
# ============================================================

def parse_year_quarter(filename: str):
    match = re.search(r"(\d{4})-Q(\d)", filename)

    if not match:
        raise ValueError(
            f"Could not extract year/quarter from filename: {filename}"
        )

    return int(match.group(1)), int(match.group(2))


# ============================================================
# STANDARD 10-COLUMN SCHEMA
# ============================================================

def load_standard_10col(path: Path, year: int, quarter: int):

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str
    )

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


# ============================================================
# 2023 Q2 / Q3 - 11 COLUMNS
# ============================================================

def load_2023_q2_q3_11col(
    path: Path,
    year: int,
    quarter: int
):

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str
    )

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


# ============================================================
# 2023 Q1 - 13 COLUMNS
# ============================================================

def load_2023_q1_13col(
    path: Path,
    year: int,
    quarter: int
):

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str
    )

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
# NUMERIC CLEANING
# ============================================================

def clean_numeric(series: pd.Series) -> pd.Series:

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce"
    )


# ============================================================
# DATE CLEANING
# ============================================================

def parse_date_flexible(series: pd.Series) -> pd.Series:

    series = series.astype("string").str.strip()

    parsed = pd.to_datetime(
        series,
        format="%Y/%m/%d",
        errors="coerce"
    )

    still_missing = parsed.isna() & series.notna()

    if still_missing.any():

        parsed2 = pd.to_datetime(
            series[still_missing],
            format="%m/%d/%Y",
            errors="coerce"
        )

        parsed.loc[still_missing] = parsed2

    return parsed


# ============================================================
# LOAD REGION MAPPING
# ============================================================

def load_region_mapping():

    if not REGION_MAP_PATH.exists():

        print("WARNING:")
        print(
            f"Region mapping file not found:\n"
            f"{REGION_MAP_PATH}"
        )
        print(
            "region_en will remain NaN until "
            "region_mapping.csv is added."
        )
        print()

        return None

    region_map = pd.read_csv(
        REGION_MAP_PATH,
        encoding="utf-8-sig"
    )

    required_columns = {
        "variant",
        "canonical_en"
    }

    missing_columns = required_columns - set(
        region_map.columns
    )

    if missing_columns:

        raise ValueError(
            "region_mapping.csv is missing columns: "
            f"{missing_columns}"
        )

    return dict(
        zip(
            region_map["variant"],
            region_map["canonical_en"]
        )
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # Step 1: Extract ZIP
    # --------------------------------------------------------

    extract_zip()

    # --------------------------------------------------------
    # Step 2: Find files
    # --------------------------------------------------------

    files = find_sales_files()

    frames = []
    schema_log = []

    # --------------------------------------------------------
    # Step 3: Load files according to schema
    # --------------------------------------------------------

    for f in files:

        year, quarter = parse_year_quarter(
            f.name
        )

        if year == 2023 and quarter == 1:

            df = load_2023_q1_13col(
                f,
                year,
                quarter
            )

            schema = "2023-Q1 (13-col, custom)"

        elif year == 2023 and quarter in (2, 3):

            df = load_2023_q2_q3_11col(
                f,
                year,
                quarter
            )

            schema = "2023-Q2/Q3 (11-col)"

        else:

            df = load_standard_10col(
                f,
                year,
                quarter
            )

            schema = "standard (10-col)"

        schema_log.append(
            (
                f.name,
                schema,
                len(df)
            )
        )

        frames.append(df)

    # --------------------------------------------------------
    # Step 4: Merge
    # --------------------------------------------------------

    merged = pd.concat(
        frames,
        ignore_index=True
    )[UNIFIED_COLUMNS]

    print("=== Merge completed ===")
    print(
        f"Total rows: {len(merged):,}"
    )
    print()

    n_rows_before_harmonization = len(merged)

    # --------------------------------------------------------
    # Step 4b: Harmonize property_type / property_classification
    # --------------------------------------------------------
    # Rename the raw-extracted columns to neutral names (nothing is lost,
    # just relabeled), then rebuild the canonical English columns from them.

    merged = merged.rename(
        columns={
            "property_classification": "use_category_raw",
            "property_type": "physical_form_raw",
        }
    )

    merged["property_type"] = (
        merged["use_category_raw"]
        .astype("string")
        .str.strip()
        .map(USE_CATEGORY_MAP)
    )

    merged["property_classification"] = (
        merged["physical_form_raw"]
        .astype("string")
        .str.strip()
        .map(PHYSICAL_FORM_MAP)
    )

    unmapped_use_category = merged.loc[
        merged["use_category_raw"].notna() & merged["property_type"].isna(),
        "use_category_raw",
    ].unique()

    unmapped_physical_form = merged.loc[
        merged["physical_form_raw"].notna()
        & merged["property_classification"].isna(),
        "physical_form_raw",
    ].unique()

    if len(unmapped_use_category) > 0:
        print(
            "WARNING: unmapped use_category_raw values (became NaN): "
            f"{list(unmapped_use_category)}"
        )

    if len(unmapped_physical_form) > 0:
        print(
            "WARNING: unmapped physical_form_raw values (became NaN): "
            f"{list(unmapped_physical_form)}"
        )

    assert len(merged) == n_rows_before_harmonization, (
        "Row count changed during property field harmonization."
    )

    # --------------------------------------------------------
    # Step 5: Clean numeric fields
    # --------------------------------------------------------

    merged["price"] = clean_numeric(
        merged["price_raw"]
    )

    merged["area"] = clean_numeric(
        merged["area_raw"]
    )

    merged["n_properties"] = clean_numeric(
        merged["n_properties"]
    )

    merged["price_per_m2_reported"] = clean_numeric(
        merged["price_per_m2_reported"]
    )

    # --------------------------------------------------------
    # Step 6: Clean dates
    # --------------------------------------------------------

    merged["transaction_date"] = parse_date_flexible(
        merged["date_gregorian_raw"]
    )

    # --------------------------------------------------------
    # Step 7: Region normalization
    # --------------------------------------------------------

    region_mapping = load_region_mapping()

    merged["region_raw_stripped"] = (
        merged["region_raw"]
        .astype("string")
        .str.strip()
    )

    if region_mapping is not None:

        merged["region_en"] = (
            merged["region_raw_stripped"]
            .map(region_mapping)
        )

    else:

        merged["region_en"] = pd.NA

    # --------------------------------------------------------
    # Step 8: Calculate price per m2
    # --------------------------------------------------------
    # For EDA / market analysis only -- NEVER use as an ML input feature
    # when predicting price, since it is derived directly from price and
    # area <= 0 is treated as NaN (never inf) via a safe divisor.

    safe_area = merged["area"].where(merged["area"] > 0)

    merged["price_per_m2_calculated"] = merged["price"] / safe_area

    # price_per_m2_reported (source: "سعر المتر المربع") is only present in
    # 2023 Q1 (96.9% missing overall) -- not usable across 2020-2025, so it
    # is dropped from the final cleaned dataset. price_per_m2_calculated
    # above is the field to use instead.
    merged = merged.drop(columns=["price_per_m2_reported"])

    # --------------------------------------------------------
    # Step 9: Missing values
    # --------------------------------------------------------

    missing_overall = merged.isna().sum()

    missing_by_year = (
        merged
        .groupby("year")
        .apply(lambda g: g.isna().sum())
    )

    # --------------------------------------------------------
    # Step 10: Full-row duplicate check
    # --------------------------------------------------------

    dup_check_cols = [
        "region_raw",
        "city",
        "city_district",
        "reference_number",
        "date_gregorian_raw",
        "use_category_raw",
        "n_properties",
        "price_raw",
        "area_raw",
    ]

    full_row_duplicates_mask = merged.duplicated(
        subset=dup_check_cols,
        keep=False
    )

    n_full_row_duplicates = (
        full_row_duplicates_mask.sum()
    )

    # --------------------------------------------------------
    # Step 11: Reference number duplicates
    # --------------------------------------------------------

    ref_counts = (
        merged["reference_number"]
        .value_counts()
    )

    duplicated_ref_numbers = (
        ref_counts[
            ref_counts > 1
        ]
    )

    n_duplicated_ref_numbers = len(
        duplicated_ref_numbers
    )

    n_rows_with_duplicated_ref = (
        duplicated_ref_numbers.sum()
    )

    cross_file_dupes = 0

    if n_duplicated_ref_numbers > 0:

        dupe_ref_values = (
            duplicated_ref_numbers.index
        )

        dupe_rows = merged[
            merged["reference_number"]
            .isin(dupe_ref_values)
        ]

        files_per_ref = (
            dupe_rows
            .groupby("reference_number")
            ["source_file"]
            .nunique()
        )

        cross_file_dupes = (
            files_per_ref > 1
        ).sum()

    # --------------------------------------------------------
    # Step 12: Save
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    merged.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        compression="gzip"
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 60)
    print("                 MERGE REPORT")
    print("=" * 60)

    print()
    print(f"Files processed: {len(files)}")

    print(
        f"Total rows after merge: "
        f"{len(merged):,}"
    )

    print()
    print("=== Schema used per file ===")

    for name, schema, n in schema_log:

        print(
            f"{name:30s} -> "
            f"{schema:30s} "
            f"({n:,} rows)"
        )

    print()
    print("=== Data Quality Checks ===")

    print(
        "Rows with NaN price: "
        f"{merged['price'].isna().sum():,}"
    )

    print(
        "Rows with NaN area: "
        f"{merged['area'].isna().sum():,}"
    )

    print(
        "Rows with NaN date: "
        f"{merged['transaction_date'].isna().sum():,}"
    )

    print(
        "Rows with unmapped region: "
        f"{merged['region_en'].isna().sum():,}"
    )

    print()
    print("=== Rows per year ===")

    print(
        merged
        .groupby("year")
        .size()
        .to_string()
    )

    print()
    print("=== Property field harmonization checks ===")

    print(
        "price_per_m2_reported dropped from final dataset: "
        f"{'price_per_m2_reported' not in merged.columns}"
    )

    print(
        "price_per_m2_calculated has zero infinite values: "
        f"{not np.isinf(merged['price_per_m2_calculated']).any()}"
    )

    physical_form_values = set(PHYSICAL_FORM_MAP.values())
    use_category_values = set(USE_CATEGORY_MAP.values())

    print(
        "property_type contains no physical-form values: "
        f"{merged['property_type'].dropna().isin(physical_form_values).sum() == 0}"
    )

    print(
        "property_classification contains no use-category values: "
        f"{merged['property_classification'].dropna().isin(use_category_values).sum() == 0}"
    )

    print()
    print("Missing-value % (property_type):", round(merged["property_type"].isna().mean() * 100, 2))
    print("Missing-value % (property_classification):", round(merged["property_classification"].isna().mean() * 100, 2))

    print()
    print("property_type value counts:")
    print(merged["property_type"].value_counts(dropna=False).to_string())

    print()
    print("property_classification value counts:")
    print(merged["property_classification"].value_counts(dropna=False).to_string())

    print()
    print("property_type coverage by year (non-null rows):")
    print(merged.groupby("year")["property_type"].apply(lambda s: s.notna().sum()).to_string())

    print()
    print("property_classification coverage by year (non-null rows):")
    print(merged.groupby("year")["property_classification"].apply(lambda s: s.notna().sum()).to_string())

    print()
    print("=== Full-row duplicate check ===")

    print(
        "Total rows involved in "
        f"full-row duplicates: "
        f"{n_full_row_duplicates:,}"
    )

    print()
    print(
        "=== reference_number duplicate check ==="
    )

    print(
        "Distinct reference_numbers appearing "
        f"more than once: "
        f"{n_duplicated_ref_numbers:,}"
    )

    print(
        "Total rows sharing a duplicated "
        f"reference_number: "
        f"{n_rows_with_duplicated_ref:,}"
    )

    print(
        "Reference_numbers appearing in "
        f"MORE THAN ONE FILE: "
        f"{cross_file_dupes:,}"
    )

    print()
    print("=== Missing values by year ===")

    print(
        missing_by_year[
            [
                "price",
                "area",
                "transaction_date",
                "region_en"
            ]
        ].to_string()
    )

    print()
    print("=" * 60)
    print("Output saved to:")
    print(OUTPUT_PATH)
    print("=" * 60)
    print("the columns of the merged dataset:", merged.columns.tolist())

if __name__ == "__main__":
    main()