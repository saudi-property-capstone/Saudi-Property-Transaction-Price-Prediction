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

