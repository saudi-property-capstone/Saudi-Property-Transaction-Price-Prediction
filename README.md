# Saudi-Property-Transaction-Price-Prediction

This project will develop and evaluate predictive models using Saudi real-estate transaction data to estimate property transaction prices. It will also examine which property and transaction characteristics most influence estimated prices, with the goal of providing practical, data-driven support for real-estate pricing decisions.

## Setup

```bash
pip install -r requirements.txt
```

## Data Setup

The dataset files are not included in this repository due to their size.

### Required files

Download the project data files and place them in the following directories:

```text
data/
└── raw/
    └── MOJ Dataset.zip
outputs/
└── tables/
    └── region_mapping.csv
```

## Scripts and Execution Order

Run the scripts in this order:

```bash
python src/data_cleaning.py
python src/EDA.py
python src/outlier_analysis.py
```

`src/data_io.py` is not run directly -- it is a small shared, read-only module that `EDA.py` and `outlier_analysis.py` both import, so neither script repeats cleaning logic or depends on the other.

### 1. Data Cleaning (`src/data_cleaning.py`)

Reads the 24 raw quarterly files and produces the single processed dataset used by every later script. It never modifies the raw ZIP or extracted files.

Steps:

* Extract the dataset ZIP file.
* Load all 24 quarterly MOJ sales files (2020 Q1 - 2025 Q4), handling schema differences across source files.
* Concatenate into one merged table and harmonize `property_type` (broad use category).
* Clean numeric fields (price, area, n_properties), parse transaction dates, and normalize region names using `region_mapping.csv`.
* Apply row filters, in order:
  * Remove bulk transactions (`n_properties > 1`) -- the project scope is single-property transactions only.
  * Remove rows with invalid price or area (`price <= 0` or `area <= 0`).
  * Remove rows missing any required transaction field (`price`, `area`, `transaction_date`, `region_en`, `property_type`).
  * Remove suspicious high-price transaction groups: rows sharing an identical price, transaction date, region, and city, priced above the global 99th percentile, spanning more than one `reference_number`, with `max(area) / min(area) > 10` (consistent with one bulk contract's value being duplicated across several parcels). The removed rows are saved to `outputs/tables/suspicious_transactions_removed.csv` before deletion, with their identifying columns, area ratio, and removal reason.
  * Remove duplicate rows (full-row business-key duplicates or a repeated `reference_number`, keeping the first occurrence).
  * Remove underrepresented property types (`Industrial`, `Mixed Use`, `Other`) -- too few observations for reliable analysis. Only `Residential`, `Commercial`, and `Agricultural` are retained.
* Select the final output columns and run validation checks (row-count reconciliation, no missing required fields, no excluded categories, no infinite `price_per_m2_calculated`, etc.) before saving.
* Print one consolidated cleaning report covering input discovery, schema summary, raw merge count, cleaning summary, validation results, and final dataset summary.

**Columns removed from the final dataset** (kept only in intermediate processing, or never needed): `property_classification` (2023 Q1-Q3-only physical-form field, discontinued), `price_per_m2_reported` (mostly missing, superseded by the calculated field below), `date_hijri` (not needed -- `transaction_date` is the dataset's date column), `plan_number`, `plot_number`, and raw/intermediate helper columns (`price_raw`, `area_raw`, `use_category_raw`, `physical_form_raw`, `region_raw_stripped`).

#### Output

```text
data/processed/moj_sales_2020_2025_merged.csv.gz
```

The final dataset contains approximately 1.39 million single-property transactions (Residential, Commercial, or Agricultural) with these columns: `region_raw`, `city`, `city_district`, `reference_number`, `date_gregorian_raw`, `n_properties`, `source_file`, `year`, `quarter`, `property_type`, `price`, `area`, `transaction_date`, `region_en`, `price_per_m2_calculated`.

### 2. Exploratory Data Analysis (`src/EDA.py`)

Reads only the processed dataset (via `data_io.load_clean_data()`) and never writes back to it or to `data/raw/`. Covers:

* Dataset overview (shape, date range, missing values, duplicates, numerical summary).
* Transaction price and property area distributions (mean/median/percentiles/skewness).
* Sampled price-vs-area relationship, including a log-scale view and a breakdown by property type.
* Quarterly (2020 Q1-2025 Q4) and yearly transaction volume and price trends, including QoQ/YoY growth and a rolling average.
* Regional analysis (transaction share and median price by region).
* Property type analysis (`Residential` / `Commercial` / `Agricultural` -- the only categories retained in the processed dataset).
* Year x quarter median price heatmap.
* Price per m² market analysis by region and over time (EDA only -- never used as a model input feature, since it is derived from price).
* Feature-target relationships and a correlation matrix.
* Validation checks confirming the analysis reconciles with the cleaned dataset totals, and key findings computed from the current dataset (not hard-coded, so they stay accurate as the cleaning rules evolve).

#### Output

Summary tables are saved to `outputs/tables/`:

```text
quarterly_transaction_summary.csv
yearly_market_summary.csv
regional_summary.csv
property_type_summary.csv
```

Figures are saved to `outputs/figures/`:

```text
transaction_price_distribution.png
price_area_relationship.png
price_area_log_scale.png
price_area_by_property_type.png
quarterly_transaction_volume.png
yearly_market_trends.png
median_price_by_region.png
median_price_by_property_type.png
quarter_year_heatmap.png
```

### 3. Outlier Analysis (`src/outlier_analysis.py`)

A read-only diagnostic script -- it inspects the processed dataset without filtering or modifying it, and does not train any model. Covers:

* Invalid-value inspection (should all be zero, since `data_cleaning.py` already enforces them).
* Top extreme price and area transactions, with segment context (property type, region, bulk share).
* Full context on the single maximum-price transaction, and same-price/same-day duplication checks.
* Validation that no suspicious high-price groups remain in the processed dataset, using the exact same rule `data_cleaning.py` uses to remove them (shared via `data_io.find_suspicious_price_groups`).
* Group-level price diagnostics (IQR fences by property type, region, and city) -- flags statistically extreme values without automatically removing them.

#### Output

```text
outputs/tables/outlier_inspection.csv
```

## Notes

* Exact key findings (median/mean price, dominant regions and property type, area-price correlation, etc.) are printed by `src/EDA.py` when it runs, computed live from the current processed dataset.
