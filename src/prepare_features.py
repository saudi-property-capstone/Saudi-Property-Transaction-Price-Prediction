"""
Train / Validation / Test Split and Feature Handoff
MOJ Sales, 2020 Q1 - 2025 Q4 (cleaned/processed dataset)

 1. Validate the cleaned input without repairing or removing records.
 2. Split by source year: 2020-2023 train, 2024 validation, 2025 test.
 3. Select predictors and export original and natural-log price targets.
 4. Fit scaled/unscaled preprocessing on training and transform validation.
 5. Build native-categorical CatBoost features for train/validation only.
 6. Validate and save the aligned handoff files.

Run from the project folder: python3 src/prepare_features.py
The existing cleaned dataset is never overwritten. Test inputs are exported
separately without fitting preprocessing or running predictions on them.
This includes CatBoost: only train/validation CatBoost feature files are
built here, so 2025 stays untouched until final model selection.
"""
import json
import sys
from pathlib import Path

# Keep the same import names when running this file directly from VS Code.
# This also lets saved preprocessing objects be loaded by later project scripts.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.feature_engineering import (
    ALLOWED_PROPERTY_TYPES, CATBOOST_CATEGORICAL, CATBOOST_NUMERIC, RANDOM_STATE,
    TARGET_ENCODING_FOLDS, catboost_features, log_price_target, make_preprocessor,
    select_inputs,
)
from src.data_io import REQUIRED_TRANSACTION_COLUMNS

# ============================================================
# PATHS
# ============================================================

# Resolve data paths from the project folder, regardless of the shell location.
DEFAULT_INPUT = ROOT / 'data/processed/moj_sales_2020_2025_merged.csv.gz'


def write_compressed_csv(frame, path):
    """Close each gzip completely before moving it to its final filename."""
    # Explicit compression is needed because the temporary suffix is .tmp.
    temporary = Path(path).with_name(Path(path).name + '.tmp')
    try:
        frame.to_csv(temporary, index=True, compression={'method': 'gzip'})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_feature_matrix(matrix, path):
    """Save a numeric NumPy array, keeping unfinished output under a temporary name."""
    temporary = Path(path).with_name(Path(path).name + '.tmp')
    try:
        with temporary.open('wb') as handle:
            np.save(handle, matrix, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


# ============================================================
# CLEANED SCHEMA CHECK
# ============================================================

def validate_columns(columns):
    """Fail early on an older merged file; never repair or remap it here."""
    required = set(REQUIRED_TRANSACTION_COLUMNS) | {
        'year', 'quarter', 'reference_number', 'n_properties', 'city',
        'city_district', 'region_raw',
    }
    missing = required - set(columns)
    # Main's final output removes these legacy/intermediate fields.
    legacy = {'property_classification', 'price_per_m2_reported',
              'use_category_raw', 'physical_form_raw'} & set(columns)
    if missing or legacy:
        raise ValueError(
            'Input does not match current main cleaned schema; '
            f'missing columns={sorted(missing)}, legacy columns={sorted(legacy)}. '
            'Ask the cleaning owner for the current processed dataset. '
            'No input was modified and no split outputs were written.'
        )


# ============================================================
# CLEANED INPUT CHECK
# ============================================================

def validate_input_contract(df):
    """Read-only checks of main's cleaned-data contract, not more cleaning."""
    validate_columns(df.columns)
    issues = {}
    for col in REQUIRED_TRANSACTION_COLUMNS:
        count = int(df[col].isna().sum())
        if count:
            issues[f'{col}_missing'] = count
    for col in ['price', 'area']:
        values = pd.to_numeric(df[col], errors='coerce')
        invalid = ~np.isfinite(values) | values.le(0)
        if invalid.any():
            issues[f'{col}_invalid'] = int(invalid.sum())
    # Main exports parsed dates in ISO form; raw source date strings are not used.
    dates = pd.to_datetime(df['transaction_date'], format='ISO8601', errors='coerce')
    if dates.isna().any():
        issues['transaction_date_invalid'] = int(dates.isna().sum())
    invalid_types = ~df['property_type'].isin(ALLOWED_PROPERTY_TYPES)
    if invalid_types.any():
        issues['property_type_outside_main_scope'] = int(invalid_types.sum())
    # Main removes bulk rows, but explicitly retains unknown/nonpositive counts.
    # Do not add another cleaning rule for those retained rows here.
    bulk = pd.to_numeric(df['n_properties'], errors='coerce').gt(1)
    if bulk.any():
        issues['bulk_rows_outside_main_scope'] = int(bulk.sum())
    if issues:
        raise ValueError(
            f'Cleaned-data contract issues: {issues}. Report to the cleaning owner; '
            'no records were fixed, removed, or exported.'
        )


# ============================================================
# STEP 2: CHRONOLOGICAL SPLIT
# ============================================================

def split_data(df):
    """Partition rows by year and report issues without changing source records.

    Returns the three split frames, a count/overlap report, and target issue
    row IDs. row_id links each input row to its target and audit metadata.
    """
    required = {'year', 'quarter', 'price', 'area', 'reference_number'}
    if required - set(df):
        raise ValueError(f'Missing split columns: {sorted(required - set(df))}')
    # Work on a copy; row_id preserves input order and aligns X, y, and metadata.
    # It is bookkeeping only and must not be passed to a model as a feature.
    df = df.copy().reset_index(drop=True)
    df.index.name = 'row_id'
    year = pd.to_numeric(df['year'], errors='coerce')
    quarter = pd.to_numeric(df['quarter'], errors='coerce')
    # Python excludes the stop value: range(2020, 2026) means 2020 through 2025.
    # There is no 2026 data in this split; reject unsupported years/quarters.
    if not (year.isin(range(2020, 2026)) & quarter.isin([1, 2, 3, 4])).all():
        raise ValueError('Invalid source year/quarter; correct these before splitting')
    df['year'], df['quarter'] = year.astype(int), quarter.astype(int)
    # Split chronologically BEFORE learning encodings or scaling.
    # These years come from quarterly source files, not parsed transaction dates.
    splits = {name: df.loc[mask].copy() for name, mask in {
        'train': year.between(2020, 2023), 'validation': year.eq(2024), 'test': year.eq(2025)
    }.items()}
    report = {'split_basis': 'source-file year/quarter', 'input_rows': len(df), 'splits': {}}
    # Report label problems to the cleaning owner without deleting any rows.
    target_issues = []
    for name, part in splits.items():
        price = pd.to_numeric(part['price'], errors='coerce')
        invalid = ~np.isfinite(price) | price.le(0)
        # Audit only: preserve all rows and labels; cleaning belongs to its owner.
        issue = part.loc[invalid, ['year', 'quarter']].copy()
        issue['split'], issue['reason'] = name, 'missing_nonfinite_or_nonpositive_target'
        target_issues.append(issue)
        if part.empty:
            raise ValueError(f'{name} has no rows')
        splits[name] = part
        report['splits'][name] = {'source_rows': len(invalid), 'invalid_target_rows_preserved': int(invalid.sum()), 'retained_rows': len(part)}
    # IDs are audit metadata, not features. Repeated references can be legitimate;
    # report overlaps for team review, never silently deduplicate transactions.
    refs = {k: set(v['reference_number'].dropna().astype(str).str.strip()) - {''} for k, v in splits.items()}
    report['reference_overlap_counts'] = {f'{a}/{b}': len(refs[a] & refs[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
    # Every input row must appear in exactly one of the disjoint year groups.
    assert sum(len(v) for v in splits.values()) == len(df)
    return splits, report, pd.concat(target_issues)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run(input_path=DEFAULT_INPUT, project_root=ROOT):
    """Run the two assigned stages and return the report."""
    # --- Step 1: Read and validate the processed dataset ---
    input_path = Path(input_path)
    project_root = Path(project_root)
    output_path = project_root / 'data/processed'
    table_path = project_root / 'outputs/tables'
    # Repeated runs replace only our named exports in this fixed directory.
    # Inspect the header first so a stale 24-column file stops before processing.
    validate_columns(pd.read_csv(input_path, nrows=0).columns)
    # Read identifier/location columns as text, preserving leading zeros.
    df = pd.read_csv(input_path, low_memory=False, dtype={
        'reference_number': 'string', 'city': 'string', 'city_district': 'string',
    })
    validate_input_contract(df)
    # --- Step 2: Partition by year before fitting any preprocessing ---
    splits, report, target_issues = split_data(df)
    # --- Step 3: Keep predictor inputs separate from transaction price ---
    inputs = {name: select_inputs(part) for name, part in splits.items()}
    # Log labels remain y, not X. No test-label transformation is needed yet.
    log_targets = {name: log_price_target(splits[name]['price'])
                   for name in ['train', 'validation']}
    # --- Step 4: Learn from training and reuse the state for validation ---
    preprocessors, matrices, feature_names = {}, {}, {}
    for variant, scaled in [('scaled', True), ('unscaled', False)]:
        preprocessor = make_preprocessor(scale_numeric=scaled)
        # Passing y_train_log is necessary for supervised REGION encoding.
        # fit_transform cross-fits the training encoding; do not replace it
        # with fit(...).transform(X_train), including after loading saved JSON settings.
        matrices[f'train_{variant}'] = preprocessor.fit_transform(
            inputs['train'], log_targets['train'],
        )
        # No validation labels are passed here. Test inputs stay untransformed.
        matrices[f'validation_{variant}'] = preprocessor.transform(inputs['validation'])
        preprocessors[variant] = preprocessor
        feature_names[variant] = preprocessor.get_feature_names_out().tolist()
    if feature_names['scaled'] != feature_names['unscaled']:
        raise ValueError('Scaled and unscaled feature columns are not aligned')
    # --- Step 5: Build CatBoost's separate native-categorical features ---
    # Train/validation only: 2025 test rows stay untransformed until final
    # model selection, matching the existing scaled/unscaled preprocessing.
    catboost_frames = {name: catboost_features(inputs[name]) for name in ['train', 'validation']}
    catboost_feature_names = CATBOOST_NUMERIC + CATBOOST_CATEGORICAL
    # --- Step 6: Validate and export the handoff ---
    for matrix in matrices.values():
        if not np.isfinite(matrix).all():
            raise ValueError('Nonfinite transformed features')
    for name, frame in catboost_frames.items():
        if not np.isfinite(frame[CATBOOST_NUMERIC].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite CatBoost numeric features')
        if len(frame) != len(inputs[name]) or not frame.index.equals(inputs[name].index):
            raise ValueError('CatBoost features are not aligned with their split rows')
    for folder in [output_path, table_path]:
        folder.mkdir(parents=True, exist_ok=True)
    # The previous verification no longer applies once these exports change.
    (table_path / 'feature_verification_report.json').unlink(missing_ok=True)
    # CSVs retain row_id; matrix rows follow each split's CSV order.
    # Original prices remain separate in SAR, including untouched test labels.
    for name, part in splits.items():
        write_compressed_csv(inputs[name], output_path / f'X_{name}_raw.csv.gz')
        write_compressed_csv(part[['price']], output_path / f'y_{name}.csv.gz')
        write_compressed_csv(
            part[['reference_number', 'year', 'quarter']],
            output_path / f'{name}_metadata.csv.gz',
        )
    for name, target in log_targets.items():
        write_compressed_csv(target, output_path / f'y_{name}_log.csv.gz')
    for name, matrix in matrices.items():
        write_feature_matrix(matrix, output_path / f'X_{name}.npy')
    # Compressed CSV keeps column names and string categoricals for CatBoost,
    # unlike the plain NumPy arrays used for the scaled/unscaled matrices above.
    for name, frame in catboost_frames.items():
        write_compressed_csv(frame, output_path / f'X_{name}_catboost.csv.gz')
    # Preprocessors contain learned feature transforms, not prediction models.
    for variant, preprocessor in preprocessors.items():
        preprocessor.save(output_path / f'preprocessor_{variant}.json')
    target_issues.to_csv(table_path / 'feature_target_issues.csv', index=True)
    feature_names['catboost'] = catboost_feature_names
    (table_path / 'feature_names.json').write_text(json.dumps(feature_names, indent=2), encoding='utf-8')
    report.update({
        'feature_count': len(feature_names['scaled']),
        'feature_variants': {
            'scaled': ['MLP'], 'unscaled': ['XGBoost'], 'native_categorical': ['CatBoost'],
        },
        'target_transform': 'natural log: log(price); inverse: exp(log_prediction)',
        'region_encoding_target': 'training log_price',
        'training_region_encoding': 'out-of-fold mean log_price; five shuffled folds inside 2020-2023 only',
        'region_smoothing': 'none',
        'feature_backend': 'pandas and NumPy',
        'matrix_format': 'NumPy NPY numeric arrays; allow_pickle=False',
        'preprocessor_format': 'JSON settings',
        'target_encoding_folds': TARGET_ENCODING_FOLDS,
        'target_encoding_random_state': RANDOM_STATE,
        'fit_years': [2020, 2021, 2022, 2023],
        'test_transformed': False, 'test_log_target_exported': False,
        'schema': 'main_canonical', 'predictive_model_fitted': False,
        'catboost_features': {
            'numeric_columns': CATBOOST_NUMERIC,
            'categorical_columns': CATBOOST_CATEGORICAL,
            'categorical_encoding': 'none: native CatBoost categoricals passed as strings '
                                     '(no one-hot, target encoding, frequency encoding, or scaling)',
            'feature_count': len(catboost_feature_names),
            'matrix_format': 'compressed CSV (.csv.gz) with column names and row_id index',
            'splits_exported': ['train', 'validation'],
            'test_transformed': False,
        },
       'output_directory': output_path.relative_to(project_root).as_posix(),
'preprocessor_directory': output_path.relative_to(project_root).as_posix(),
'table_directory': table_path.relative_to(project_root).as_posix(),
    })
    (table_path / 'feature_split_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


# ============================================================
# RUN SPLITTING AND FEATURE ENGINEERING
# ============================================================

if __name__ == '__main__':
    report = run()
    print(pd.DataFrame(report['splits']).T[['retained_rows']].to_string())
    print('Feature columns per variant (XGBoost/MLP):', report['feature_count'])
    print('CatBoost feature columns:', report['catboost_features']['feature_count'])
    print('Split files and preprocessing:', report['output_directory'])
    print('Reports:', report['table_directory'])
