# Saudi-Property-Transaction-Price-Prediction
This project will develop and evaluate predictive models using Saudi real-estate transaction data to estimate property transaction prices. It will also examine which property and transaction characteristics most influence estimated prices, with the goal of providing practical, data-driven support for real-estate pricing decisions.

## Data Setup

The dataset files are not included in this repository due to their size.

### Required files

Download the project data files and place them in the following directory:

```text
data/
└── raw/
    ├── MOJ Dataset.zip
    └── region_mapping.csv
```

### Run the Data Cleaning Pipeline

After placing the required files in `data/raw/`, run:

```bash
python src/data_cleaning.py
```

The pipeline will automatically:

* Extract the dataset ZIP file.
* Load all 24 quarterly MOJ sales files from 2020 to 2025.
* Handle schema differences across the source files.
* Clean numeric and date fields.
* Normalize region names using `region_mapping.csv`.
* Perform missing value and duplicate checks.
* Merge all transactions into a unified dataset.

### Output

The processed dataset will be generated automatically at:

```text
data/processed/moj_sales_2020_2025_merged.csv.gz
```

The final dataset contains approximately 1.4 million real estate transaction records.

## Exploratory Data Analysis

After the processed dataset exists, run:

```bash
python src/EDA.py
```

The script reads `data/processed/moj_sales_2020_2025_merged.csv.gz` (it never touches `data/raw/`) and covers:

* Dataset overview (shape, date range, missing values, duplicates, numerical summary).
* Transaction price and property area distributions (mean/median/percentiles/skewness).
* Sampled price-vs-area relationship.
* Quarterly (2020 Q1-2025 Q4) and yearly transaction volume and price trends, including QoQ/YoY growth and a rolling average.
* Regional analysis (transaction share and median price by region).
* Property type analysis (broad use category: Residential/Commercial/Agricultural/Industrial) and a separate property classification breakdown (physical form: Villa/Land/Apartment/etc., available only for 2023 Q1-Q3).
* Year x quarter median price heatmap.
* Price per m² market analysis by region and over time (for EDA only -- never used as a model input feature, since it is derived from price).
* Feature-target relationships and a correlation matrix.
* Validation checks confirming the analysis reconciles with the cleaned dataset totals.

### Output

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
quarterly_transaction_volume.png
yearly_market_trends.png
median_price_by_region.png
median_price_by_property_type.png
quarter_year_heatmap.png
```

## EDA Key Findings

- Transaction prices are highly right-skewed; the median price (350K SAR) is more representative than the mean (1.04M SAR).
- Median transaction prices generally increased from 2020 to 2024, with a slight decline in 2025.
- Riyadh, Makkah, and the Eastern Province account for most transaction activity.
- Residential transactions dominate the dataset, representing about 85.4% of records.
- Location shows substantial differences in median transaction prices across regions and districts.
- Extreme price and area values require investigation before final modeling decisions.

