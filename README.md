# Saudi-Property-Transaction-Price-Prediction

This project will develop and evaluate predictive models using Saudi real-estate transaction data to estimate property transaction prices. It will also examine which property and transaction characteristics most influence estimated prices, with the goal of providing practical, data-driven support for real-estate pricing decisions.

## Modeling Approach

Three initial regression models were trained and evaluated on the same chronological split of the processed dataset, and compared against a rule-based baseline:

1. XGBoost Regressor
2. CatBoost Regressor
3. A simple MLP neural network for regression

| Split | Source-file years |
|---|---|
| Train | 2020-2023 |
| Validation | 2024 |
| Test | 2025 |

Test 2025 is the held-out evaluation set. It was not used to fit the initial models, fit preprocessing transformations, or select their initial configurations.

| Model | Feature input |
|---|---|
| XGBoost | Unscaled engineered numeric features (frequency-encoded location, target-encoded region, one-hot property type) |
| MLP / Deep Learning | The same engineered numeric features, scaled |
| CatBoost | Native categorical features (region, city, city_district, property_type kept as strings; no frequency, target, or one-hot encoding) |

All preprocessing that learns from data (location frequencies, region target encoding, scaling) is fit on Training 2020-2023 only and then applied unchanged to Validation 2024 and Test 2025, so validation and test rows never leak into it. See "4. Train / Validation / Test Split and Feature Engineering" below for details.

Every model is trained on the natural log of transaction price, `ln(price)`. Predictions are converted back to SAR with `exp(prediction)` (not `expm1`) before computing MAE, RMSE, and R².

### Baseline

A rule-based **historical median baseline** (no machine learning) sets the bar the three models must clear. See "5. Baseline Model: Historical Median" below. The project's success criterion is **at least a 15% reduction in MAE relative to this baseline**. Results are in "6. ML and DL Models — Initial Models" and "7. Hyperparameter Tuning and Stacking".

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
python src/prepare_features.py
```

Then, separately, run `notebooks/baseline_model.ipynb` (see "5. Baseline Model: Historical Median" below) -- it is a standalone Jupyter notebook in `notebooks/`, not a `src/` script, and only needs the split files `prepare_features.py` already produced.

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

### 4. Train / Validation / Test Split and Feature Engineering (`src/prepare_features.py`)

Reads `data/processed/moj_sales_2020_2025_merged.csv.gz` and produces chronological splits and model-ready features. Feature transformations are defined in `src/feature_engineering.py` and called by `prepare_features.py`.

Steps:

* Validate the processed dataset's schema, required values, and source years/quarters before exporting files.
* Split by source-file year before fitting preprocessing: 2020-2023 for training, 2024 for validation, and 2025 for testing.
* Retain each input row in one split and assign a `row_id` to align predictors, targets, and metadata.
* Select area, time, location, and property type as inputs. Exclude price, price-per-m² fields, transaction identifiers, and `row_id` from predictors.
* Create a sequential quarter index and sine/cosine features for quarterly seasonality.
* Encode city and district combinations using their frequencies in training data. Include region in the location keys to distinguish identical names in different areas; assign zero to unseen combinations.
* Target-encode regions using mean training log-price, with five shuffled internal folds (seed 42) and no smoothing. Each training fold is encoded using the other four folds. Validation uses the full-training mapping; unseen regions receive the training log-price mean.
* One-hot encode property type (`Residential` / `Commercial` / `Agricultural`) using the training category vocabulary.
* Export natural-log price targets (`np.log(price)`) for training and validation, alongside original prices in SAR. Convert later log-price predictions back to SAR with `np.exp(log_prediction)`.
* Prepare the same engineered/encoded features (area, quarter time index, quarter sine/cosine, city/district frequency, region target encoding, property-type one-hot) in two variants: unscaled for **XGBoost**, and scaled using training means and population standard deviations (`ddof=0`) for **MLP** (one-hot indicators stay 0/1 and are not scaled).
* Separately, build a native-categorical feature set for **CatBoost**: the same `area`, `time_index`, `quarter_sin`, `quarter_cos` numeric columns, plus `region`, `city`, `city_district`, and `property_type` kept as raw strings -- no frequency encoding, no target encoding, no one-hot encoding, and no scaling. This path has no learned state, so it is built directly for training and validation without a separate fit step.
* Export transformed training and validation matrices, fitted preprocessing settings, and separate raw inputs, targets, and metadata for all three splits. Test inputs remain untransformed for every model path (including CatBoost) until final evaluation.
* Check row preservation, matching feature columns across variants, and finite transformed values. Report transaction-reference overlaps, split counts, feature count, and output paths.

The reference run contains 1,392,059 transactions and produces 10 features per scaled/unscaled variant (XGBoost/MLP) and 8 native-categorical features (CatBoost):

| Split | Source-file years | Rows |
|---|---|---:|
| Train | 2020-2023 | 945,033 |
| Validation | 2024 | 247,088 |
| Test | 2025 | 199,938 |

Counts and feature dimensions are recorded in `feature_split_report.json` and may change with the processed dataset.

Run from the project root after generating the processed dataset:

```bash
python src/prepare_features.py
```

#### Output

Feature matrices are saved to `data/processed/`:

```text
X_train_scaled.npy / X_validation_scaled.npy           -> MLP
X_train_unscaled.npy / X_validation_unscaled.npy       -> XGBoost
X_train_catboost.csv.gz / X_validation_catboost.csv.gz -> CatBoost
```

The scaled/unscaled matrices are plain NumPy arrays (column order is recorded in `feature_names.json`). They preserve the split row order and align with the corresponding raw-input and target files. The CatBoost files are compressed CSV files that preserve column names, string categorical values, and the `row_id` index.

Inputs, targets, metadata, and preprocessing settings are also saved to `data/processed/` (`{split}` represents `train`, `validation`, or `test`):

```text
X_{split}_raw.csv.gz
y_{split}.csv.gz
{split}_metadata.csv.gz
y_train_log.csv.gz
y_validation_log.csv.gz
preprocessor_scaled.json
preprocessor_unscaled.json
```

There is no separate preprocessor file for CatBoost: its feature builder (`catboost_features` in `src/feature_engineering.py`) has no learned state to save.

Reports are saved to `outputs/tables/`:

```text
feature_split_report.json
feature_names.json
feature_target_issues.csv
```

**Modeling input:** For XGBoost and MLP, use the exported training matrices, which contain out-of-fold region target encodings. Transforming the raw training inputs again with a saved preprocessor would use full-training regional means and therefore would not reproduce these training matrices. CatBoost instead uses the exported native-categorical training file directly. Any future model cross-validation must refit learned preprocessing within each training fold; the five target-encoding folds are not a substitute for chronological model validation.

### 5. Baseline Model: Historical Median (`notebooks/baseline_model.ipynb`)

A rule-based baseline that every later model (XGBoost, CatBoost, MLP) must beat. It reuses the split files from `prepare_features.py` directly (`X_{split}_raw.csv.gz` joined with `y_{split}.csv.gz` on `row_id`) instead of re-deriving the split or any preprocessing.

**Rule:** predict a property's price as the training-only (2020-2023) median transaction price of properties sharing the same `city_district` (location) and `property_type`.

Steps:

* Compute the median `price` for every `city_district` + `property_type` combination observed in training only -- 2024 validation and 2025 test rows never contribute to this or any other baseline statistic.
* Apply a two-level fallback for a validation combination unseen in training: first the training median for that `property_type` alone, then the overall training median price.
* Evaluate on 2024 validation with MAE (primary metric), RMSE, and R², overall and broken down by `property_type`.
* Print the project's 15%-improvement target, `target_MAE = baseline_MAE x 0.85`, that XGBoost, CatBoost, and MLP must each reach or beat on 2024 validation.

The reference run groups by 21,642 `city_district` x `property_type` training combinations and reports:

| Model | Dataset | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Baseline - Historical Median | Validation 2024 | 663,987.74 | 8,648,982.03 | 0.0619 |
| Target (baseline MAE x 0.85) | Validation 2024 | 564,389.58 | -- | -- |

Of the 247,088 validation rows, 240,568 matched a training `city_district` + `property_type` combination directly and 6,520 fell back to the training `property_type` median; no row needed the global-median fallback. Per-`property_type` counts and metrics are in the notebook and in `outputs/initial_models/tables/baseline_median_summary.csv`.

#### Output

```text
outputs/initial_models/tables/baseline_median_summary.csv
```

## 6. ML and DL Models — Initial Models

Initial (untuned) XGBoost, CatBoost and MLP models, trained on `ln(price)` and compared with the historical-median baseline on Validation 2024 and Test 2025.

- Report: [`outputs/initial_models/reports/initial_models_report.md`](outputs/initial_models/reports/initial_models_report.md)
- Code: `src/modeling/` · Results: `outputs/initial_models/`

```bash
python -m src.modeling.run_all
```

## 7. Hyperparameter Tuning and Stacking

Seeded random search for XGBoost, CatBoost and MLP, a Ridge stack of the two best tuned models, and one Test 2025 evaluation of the final (frozen) model. All selection decisions use Validation 2024 only.

- Report: [`outputs/tuning/reports/tuning_work_in_progress.md`](outputs/tuning/reports/tuning_work_in_progress.md)
- Code: `src/tuning/` · Results: `outputs/tuning/`

```bash
python -m src.tuning.tune_models --model all
python -m src.tuning.stacking
python -m src.tuning.final_comparison
python -m src.tuning.evaluate_final_test
python -m src.tuning.make_tuning_wip_report
```

Trained model files and row-level predictions stay on disk but are not tracked in Git (size); they are reproducible with the commands above.

## 8. Model Interpretation (SHAP / Feature Importance)

Exact SHAP values of the final stacked model (Tuned XGBoost + Tuned CatBoost), grouped into five characteristics (location, area, time trend, property type, season) to show which ones drive the predicted price.

- Report: section 10 of [`outputs/tuning/reports/tuning_work_in_progress.md`](outputs/tuning/reports/tuning_work_in_progress.md)
- Code: `src/tuning/feature_importance.py` · Figures: `outputs/tuning/figures/feature_importance_*.png`

```bash
python -m src.tuning.feature_importance
python -m src.tuning.make_tuning_wip_report
```

`notebooks/results_analysis.ipynb` displays the saved results of sections 6-8 without retraining.

## Repository Structure

```text
data/
  raw/                        raw MOJ dataset (not tracked in git)
  processed/                  cleaned data, split files, feature matrices (not tracked in git)

src/
  data_cleaning.py  data_io.py  EDA.py  outlier_analysis.py
  feature_engineering.py  prepare_features.py
  modeling/                   ML and DL Initial Modeling code
  tuning/                     hyperparameter tuning and stacking code

notebooks/
  baseline_model.ipynb
  results_analysis.ipynb

outputs/
  figures/  tables/           EDA and feature-pipeline outputs
  initial_models/
    models/  tables/  predictions/  figures/  reports/  logs/  status/  smoke_test/
  tuning/
    models/  tables/  predictions/  reports/  logs/  status/
```

Fitted preprocessing has one canonical location: `data/processed/preprocessor_{scaled,unscaled}.json`, written by `src/prepare_features.py` on Training 2020-2023 and loaded from there by all code. `outputs/initial_models/models/preprocessing/` holds a checksummed snapshot for reproducibility. All output paths are defined once in `src/modeling/config.py`.
