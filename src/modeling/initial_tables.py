"""
Initial Modeling comparison tables, built ONLY from already-saved results.

Run: python -m src.modeling.initial_tables

Reads the per-model status JSON files written when the initial models were
trained / evaluated. It loads no model, makes no prediction, and never touches
Test 2025 data: it only re-arranges saved numbers into tables.

Two extra columns (R² on ln(price) and median absolute error) were not stored
in the initial status files. They exist in the saved comparison tables for
Initial XGBoost and Initial CatBoost (computed earlier from their saved
predictions); those saved values are reused when they exactly match the saved
MAE. For the baseline and Initial MLP they were never computed, so they are
left EMPTY rather than invented.
"""
import numpy as np
import pandas as pd

from src.modeling import config as cfg
from src.modeling.evaluation import improvement_pct, load_json

# Saved extra metrics live in the tuning stage's earlier comparison tables.
SUPPLEMENT = {
    'validation': (cfg.TUNING_TABLES_DIR / 'v2_validation_comparison.csv',
                   {'xgboost': 'XGBoost v1', 'catboost': 'CatBoost v1'},
                   'Validation R2 (ln price)', 'Median abs error (SAR)', 'Validation MAE (SAR)'),
    'test': (cfg.TUNING_TABLES_DIR / 'v2_test_comparison.csv',
             {'xgboost': 'XGBoost v1', 'catboost': 'CatBoost v1'},
             'Test R2 (ln price)', 'Median abs error (SAR)', 'Test MAE (SAR)'),
}


def display_name(label):
    """Map a label stored in an older result file to the current display name."""
    return cfg.LEGACY_LABELS.get(label, label)


def saved_extras(split, key, saved_mae):
    """(R² on ln price, median abs error) if saved AND consistent with the saved MAE."""
    path, names, r2_col, med_col, mae_col = SUPPLEMENT[split]
    if key not in names or not path.exists():
        return np.nan, np.nan
    table = pd.read_csv(path)
    row = table[table['Model'] == names[key]]
    if row.empty:
        return np.nan, np.nan
    row = row.iloc[0]
    if abs(row[mae_col] - saved_mae) > 0.01:          # provenance guard
        raise ValueError(f'Saved extra metrics for {key}/{split} do not match its saved MAE')
    return float(row[r2_col]), float(row[med_col])


def _model_row(split, key, result, base_mae):
    m = result[split]
    imp = improvement_pct(base_mae, m['mae'])
    r2_ln, med = saved_extras(split, key, m['mae'])
    return {
        f'{split.title()} MAE (SAR)': m['mae'], f'{split.title()} RMSE (SAR)': m['rmse'],
        f'{split.title()} R2 (SAR)': m['r2'], f'{split.title()} R2 (ln price)': r2_ln,
        'Median abs error (SAR)': med,
        'Training time (s)': result['training_seconds'],
        'Best iteration / epochs': result['best_iteration'],
        'Iteration detail': result['iteration_detail'],
        'MAE improvement vs baseline (%)': imp,
        'Met 15% target': 'Yes' if imp >= cfg.TARGET_IMPROVEMENT_PCT else 'No',
    }


def build_comparison(split):
    """Comparison table for 'validation' or 'test' from saved status files."""
    label = 'Validation' if split == 'validation' else 'Test'
    base = load_json(cfg.STATUS_DIR / f'baseline_{split}_metrics.json')
    base_mae = base['metrics']['mae']
    rows = [{'Model': cfg.MODEL_LABELS['baseline'], 'Version': 'Baseline',
             f'{label} MAE (SAR)': base_mae, f'{label} RMSE (SAR)': base['metrics']['rmse'],
             f'{label} R2 (SAR)': base['metrics']['r2'], f'{label} R2 (ln price)': np.nan,
             'Median abs error (SAR)': np.nan, 'Training time (s)': base['fit_seconds'],
             'Best iteration / epochs': np.nan, 'Iteration detail': 'not applicable (rule-based)',
             'MAE improvement vs baseline (%)': 0.0, 'Met 15% target': 'n/a (reference)'}]
    version = 'Initial' if split == 'validation' else 'Initial — Untuned'
    for key in cfg.MODEL_KEYS:
        result = load_json(cfg.validation_metrics_path(key) if split == 'validation'
                           else cfg.test_metrics_path(key))
        rows.append({'Model': cfg.MODEL_LABELS[key], 'Version': version,
                     **_model_row(split, key, result, base_mae)})
    table = pd.DataFrame(rows)
    if split == 'validation':
        ranks = table.iloc[1:][f'{label} MAE (SAR)'].rank(method='min').astype(int)
        table.insert(2, 'Rank (Validation MAE)', [None, *ranks.tolist()])
    return table


def build_per_type_test():
    """Per-property-type Test 2025 metrics from the saved test status files."""
    rows = list(load_json(cfg.STATUS_DIR / 'baseline_test_metrics.json')['per_type'])
    for key in cfg.MODEL_KEYS:
        rows += load_json(cfg.test_metrics_path(key))['per_type']
    table = pd.DataFrame(rows).rename(columns={
        'model': 'Model', 'property_type': 'Property type', 'n_rows': 'Test rows',
        'mae': 'MAE (SAR)', 'rmse': 'RMSE (SAR)', 'r2': 'R2', 'r2_note': 'R2 note'})
    table['Model'] = table['Model'].map(display_name)
    return table


def write_initial_tables():
    """Write the three main Initial Modeling tables (saved values only)."""
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    val, test, per_type = build_comparison('validation'), build_comparison('test'), build_per_type_test()
    val.to_csv(cfg.TABLES_DIR / 'initial_validation_comparison.csv', index=False)
    test.to_csv(cfg.TABLES_DIR / 'initial_test_comparison.csv', index=False)
    per_type.to_csv(cfg.TABLES_DIR / 'initial_per_property_type_test_metrics.csv', index=False)
    return val, test, per_type


if __name__ == '__main__':
    pd.set_option('display.width', 260, 'display.max_columns', 30,
                  'display.float_format', lambda x: f'{x:,.4f}')
    v, t, p = write_initial_tables()
    print(v.to_string(index=False))
    print(t.to_string(index=False))
    print(p.drop(columns='R2 note').to_string(index=False))
