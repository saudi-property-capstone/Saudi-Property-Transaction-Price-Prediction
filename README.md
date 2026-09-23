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

A rule-based **historical median baseline** (no machine learning) sets the bar the three models must clear. See "5. Baseline Model: Historical Median" below. The project's success criterion is **at least a 15% reduction in MAE relative to this baseline**. Results for the three models are in "6. ML and DL Models — Initial Models".

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

This section covers these four entries:

- `Historical Median Baseline`
- `Initial XGBoost`
- `Initial CatBoost`
- `Initial MLP`

XGBoost and CatBoost are the **machine-learning** models; the MLP is the **deep-learning** model. Code is in `src/modeling/` and every result is saved under `outputs/initial_models/`.

**How the models were evaluated**

- **MAE in SAR is the primary metric.** RMSE and R² are supporting metrics; **R² on the original SAR scale is the main R² result.**
- The success criterion is a reduction of at least 15% in MAE relative to the historical-median baseline: `(baseline MAE - model MAE) / baseline MAE x 100`.
- **Validation 2024** was used for early stopping and for comparing the initial models.
- **Test 2025** was used for held-out evaluation after the initial configurations were frozen (SHA-256 hashes are recorded in `outputs/initial_models/models/frozen_configs.json`).
- Models are trained on `ln(price)`; predictions are converted to SAR with `exp`, clipped to the training `ln(price)` range and at zero.

### Initial model configurations

Taken from the saved configuration files in `outputs/initial_models/models/*/`. These are initial settings; no parameter search was run.

| Model | Main initial configuration | Feature input | Early stopping |
|---|---|---|---|
| Initial XGBoost | `hist` tree method, depth 8, learning rate 0.05, up to 3,000 trees, subsample 0.8, colsample_bytree 0.8, squared error on `ln(price)`, seed 42 | Unscaled engineered numeric matrix | 50 rounds on Validation 2024; 1,229 trees kept |
| Initial CatBoost | Depth 8, learning rate 0.1, up to 3,000 iterations, L2 leaf regularization 3.0, RMSE loss on `ln(price)`, seed 42 | Native categorical features (region, city, city_district, property_type) plus numeric time/area features | 50 rounds on Validation 2024, best model kept; 986 trees kept |
| Initial MLP | Hidden layers 128 → 64 → 32 (ReLU), linear output, Adam (learning rate 0.001), MSE loss on `ln(price)`, batch size 2,048, up to 100 epochs, seed 42 | Scaled engineered numeric matrix | `EarlyStopping` on `val_loss`, patience 8, `restore_best_weights=True`; 13 epochs completed (best epoch 5) |

### Initial results: Validation 2024

| Model | MAE (SAR) | RMSE (SAR) | R² (SAR scale) | Training time (s) | Best iteration / epochs | MAE improvement vs baseline | 15% target met |
|---|---:|---:|---:|---:|---:|---:|---|
| Historical Median Baseline | 663,988 | 8,648,982 | 0.0619 | 0.3 | — | — | — (reference) |
| Initial XGBoost | 539,630 | 8,261,632 | 0.1440 | 43.9 | 1,229 | +18.73% | Yes |
| Initial CatBoost | 554,411 | 7,638,329 | 0.2683 | 437.3 | 986 | +16.50% | Yes |
| Initial MLP | 977,813 | 24,512,976 | -6.5355 | 34.1 | 13 | -47.26% | No |

### Initial results: Test 2025

| Model | MAE (SAR) | RMSE (SAR) | R² (SAR scale) | Training time (s) | MAE improvement vs test baseline | 15% target met |
|---|---:|---:|---:|---:|---:|---|
| Historical Median Baseline | 679,496 | 9,480,147 | 0.0204 | 0.3 | — | — (reference) |
| Initial XGBoost | 596,077 | 12,670,756 | -0.7500 | 43.9 | +12.28% | No |
| Initial CatBoost | 554,819 | 7,609,560 | 0.3688 | 437.3 | +18.35% | Yes |
| Initial MLP | 1,186,306 | 29,871,336 | -8.7260 | 34.1 | -74.59% | No |

Values are read from `outputs/initial_models/tables/initial_validation_comparison.csv` and `initial_test_comparison.csv`.

### Interpretation

- Initial XGBoost and Initial CatBoost both exceeded the 15% MAE-improvement target on Validation 2024.
- Initial CatBoost also exceeded the 15% target on Test 2025 and was the most stable individual initial model (its MAE changed by only +0.1% from validation to test).
- Initial XGBoost did not maintain the 15% improvement on Test 2025.
- Initial MLP performed worse than the historical-median baseline in its initial configuration. This result is reported as it is.
- A small number of very large commercial transactions strongly affect RMSE and SAR-scale R², which is why those metrics are low or unstable for several models.
- No rows were removed and no data were modified during Initial Modeling.

### Per-property-type evaluation (Test 2025)

Results were also evaluated separately for Residential, Commercial, and Agricultural properties. Full table: `outputs/initial_models/tables/initial_per_property_type_test_metrics.csv`.

| Model | Property type | Test rows | MAE (SAR) | RMSE (SAR) | R² (SAR scale) |
|---|---|---:|---:|---:|---:|
| Historical Median Baseline | Residential | 181,503 | 480,022 | 5,258,114 | 0.0055 |
| Historical Median Baseline | Commercial | 12,180 | 3,231,369 | 31,822,428 | 0.0207 |
| Historical Median Baseline | Agricultural | 6,255 | 1,498,579 | 9,928,843 | -0.0699 |
| Initial XGBoost | Residential | 181,503 | 381,752 | 4,612,771 | 0.2346 |
| Initial XGBoost | Commercial | 12,180 | 3,363,116 | 47,703,108 | -1.2005 |
| Initial XGBoost | Agricultural | 6,255 | 1,427,106 | 9,126,991 | 0.0960 |
| Initial CatBoost | Residential | 181,503 | 388,364 | 4,602,817 | 0.2379 |
| Initial CatBoost | Commercial | 12,180 | 2,702,117 | 24,344,781 | 0.4269 |
| Initial CatBoost | Agricultural | 6,255 | 1,203,601 | 9,060,417 | 0.1091 |
| Initial MLP | Residential | 181,503 | 608,051 | 10,611,073 | -3.0501 |
| Initial MLP | Commercial | 12,180 | 7,705,820 | 101,442,396 | -8.9510 |
| Initial MLP | Agricultural | 6,255 | 5,270,600 | 72,224,776 | -55.6110 |

### Initial Modeling files and commands

- Code: `src/modeling/`
- Results: `outputs/initial_models/` (`models/`, `tables/`, `predictions/`, `figures/`, `reports/`, `logs/`, `status/`, `smoke_test/`)
- Report: `outputs/initial_models/reports/initial_models_report.md`; `notebooks/results_analysis.ipynb` displays the saved results without retraining.

Run from the repository root, as modules:

```bash
python -m src.modeling.run_all
```

`run_all` runs the full pipeline and safely skips stages whose verified saved outputs already exist, so an interrupted run can be resumed. The individual steps can also be run on their own:

```bash
python -m src.modeling.smoke_test
python -m src.modeling.train_xgboost
python -m src.modeling.train_catboost
python -m src.modeling.train_mlp
python -m src.modeling.initial_tables
```

The three `train_*` commands accept `--force` to retrain a model. `initial_tables` rebuilds the comparison tables from already-saved results without loading any model.

### Trained models and full predictions are not tracked

Trained model files (`outputs/initial_models/models/catboost/catboost_initial.cbm`, about 150 MB; `xgboost/xgboost_initial.json`, about 30 MB; `mlp/mlp_initial.keras`), the full row-level prediction files (`outputs/initial_models/predictions/*.csv.gz`) and the training logs are intentionally excluded from Git because of repository size. They are reproducible and stay on disk locally. Configuration files, status files, comparison tables, per-property-type tables, figures and reports are tracked.

To regenerate everything (finished stages are skipped, so this only rebuilds what is missing):

```bash
python -m src.modeling.run_all
```

or a single model, for example the CatBoost model:

```bash
python -m src.modeling.train_catboost
```

## Repository Structure

```text
data/
  raw/                        raw MOJ dataset (not tracked in git)
  processed/                  cleaned data, split files, feature matrices (not tracked in git)

src/
  data_cleaning.py  data_io.py  EDA.py  outlier_analysis.py
  feature_engineering.py  prepare_features.py
  modeling/                   ML and DL Initial Modeling code

notebooks/
  baseline_model.ipynb
  results_analysis.ipynb

outputs/
  figures/  tables/           EDA and feature-pipeline outputs
  initial_models/
    models/  tables/  predictions/  figures/  reports/  logs/  status/  smoke_test/
```

Fitted preprocessing has one canonical location: `data/processed/preprocessor_{scaled,unscaled}.json`, written by `src/prepare_features.py` on Training 2020-2023 and loaded from there by all code. `outputs/initial_models/models/preprocessing/` holds a checksummed snapshot for reproducibility. All output paths are defined once in `src/modeling/config.py`.
