"""
Initial Modeling comparison tables, built ONLY from already-saved results.

Run: python -m src.modeling.initial_tables

Reads the per-model status JSON files written when the initial models were
trained / evaluated, plus the saved per-row prediction files. It loads no
model and makes no new prediction: it only re-arranges saved numbers into
tables.

Two extra columns (R² on ln(price) and median absolute error) were not stored
in the status files, so they are computed here from each model's saved
predictions (actual_price, predicted_price). A provenance guard checks that the
MAE recomputed from those predictions matches the saved MAE, so the extra
columns always describe the same predictions as the rest of the row.
"""
import numpy as np
import pandas as pd

from src.modeling import config as cfg
from src.modeling.evaluation import improvement_pct, load_json


def display_name(label):
    """Map a label stored in an older result file to the current display name."""
    return cfg.LEGACY_LABELS.get(label, label)


def prediction_extras(key, split, saved_mae):
    """(R² on ln price, median abs error in SAR) from the saved predictions file."""
    frame = pd.read_csv(cfg.predictions_path(key, split))
    actual = frame['actual_price'].to_numpy(dtype=np.float64)
    pred = frame['predicted_price'].to_numpy(dtype=np.float64)
    if not ((actual > 0).all() and (pred > 0).all()):
        raise ValueError(f'{key}/{split}: ln(price) needs positive actual and predicted prices')
    error = np.abs(actual - pred)
    if not np.isclose(error.mean(), saved_mae, rtol=1e-6):   # provenance guard
        raise ValueError(f'{key}/{split}: predictions file MAE does not match the saved MAE')
    log_actual, log_pred = np.log(actual), np.log(pred)
    r2_ln = 1.0 - np.sum((log_actual - log_pred) ** 2) / np.sum((log_actual - log_actual.mean()) ** 2)
    return float(r2_ln), float(np.median(error))


def _model_row(split, key, result, base_mae):
    m = result[split]
    imp = improvement_pct(base_mae, m['mae'])
    r2_ln, med = prediction_extras(key, split, m['mae'])
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
    base_r2_ln, base_med = prediction_extras('baseline', split, base_mae)
    rows = [{'Model': cfg.MODEL_LABELS['baseline'], 'Version': 'Baseline',
             f'{label} MAE (SAR)': base_mae, f'{label} RMSE (SAR)': base['metrics']['rmse'],
             f'{label} R2 (SAR)': base['metrics']['r2'], f'{label} R2 (ln price)': base_r2_ln,
             'Median abs error (SAR)': base_med, 'Training time (s)': base['fit_seconds'],
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
