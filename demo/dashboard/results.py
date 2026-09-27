"""Saved evaluation results (metrics tables, final selection, row-level predictions). Read-only; no model is run."""
import json

import numpy as np
import pandas as pd
import streamlit as st

from src.modeling import config as cfg

T = cfg.TUNING_TABLES_DIR
I = cfg.TABLES_DIR

STAGE_ORDER = {'Baseline': 0, 'Initial': 1, 'Initial — Untuned': 1, 'Tuned': 2, 'Tuned stack': 3}


def _csv(path):
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def final_selection():
    path = T / 'final_selection.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


@st.cache_data(show_spinner=False)
def validation_table():
    """All eight models on Validation 2024 (the split used for every selection decision)."""
    v = _csv(T / 'final_validation_comparison.csv')
    if v is None:
        return None
    return pd.DataFrame({'Model': v['Model'], 'Stage': v['Stage'], 'MAE': v['Validation MAE (SAR)'],
                         'RMSE': v['Validation RMSE (SAR)'], 'R²': v['Validation R2 (SAR)'],
                         'R² (ln price)': v['Validation R2 (ln price)'], 'Median AE': v['Median abs error (SAR)'],
                         'MAE vs baseline': v['MAE improvement vs baseline (%)']})


@st.cache_data(show_spinner=False)
def test_table():
    """Test 2025: the baseline, the three initial models and the frozen final model (evaluated once).

    Tuned individual models have no test results by design: only the frozen final selection was tested.
    """
    init, final = _csv(I / 'initial_test_comparison.csv'), _csv(T / 'final_test_comparison.csv')
    if final is None:
        return None
    rows = []
    frames = [(init, 'Version'), (final, None)] if init is not None else [(final, None)]
    for df, stage_col in frames:
        for _, r in df.iterrows():
            rows.append({'Model': r['Model'],
                         'Stage': r[stage_col] if stage_col else ('Baseline' if 'Baseline' in r['Model'] else 'Tuned stack'),
                         'MAE': r['Test MAE (SAR)'], 'RMSE': r['Test RMSE (SAR)'], 'R²': r['Test R2 (SAR)'],
                         'R² (ln price)': r['Test R2 (ln price)'], 'Median AE': r['Median abs error (SAR)'],
                         'MAE vs baseline': r.get('MAE improvement vs baseline (%)',
                                                  r.get('MAE improvement vs test baseline (%)'))})
    out = pd.DataFrame(rows).drop_duplicates('Model')
    out['Stage'] = out['Stage'].replace({'Initial — Untuned': 'Initial'})
    return out.sort_values('Stage', key=lambda s: s.map(STAGE_ORDER), kind='stable').reset_index(drop=True)


@st.cache_data(show_spinner=False)
def per_type_test():
    return _csv(T / 'final_test_per_property_type.csv')


@st.cache_data(show_spinner=False)
def feature_importance():
    return _csv(T / 'feature_importance_by_characteristic.csv'), _csv(T / 'feature_importance_by_model.csv')


def shap_figures():
    names = [('feature_importance_characteristics.png', 'Share of the final model’s SHAP importance per characteristic'),
             ('feature_importance_by_model.png', 'Mean |SHAP| per feature, Tuned XGBoost vs Tuned CatBoost'),
             ('feature_importance_area_effect.png', 'How area moves the predicted price'),
             ('feature_importance_time_effect.png', 'How the time trend moves the predicted price')]
    return [(cfg.TUNING_FIGURES_DIR / f, c) for f, c in names if (cfg.TUNING_FIGURES_DIR / f).exists()]


@st.cache_data(show_spinner='Reading the saved Test 2025 predictions…')
def test_predictions_summary(key, sample_size=6000):
    """Saved Test 2025 predictions of one model: a plotting sample plus error summaries on ALL rows."""
    path = cfg.predictions_path(key, 'test')
    if not path.exists():
        return None
    p = pd.read_csv(path, usecols=['property_type', 'actual_price', 'predicted_price'])
    err = (p['predicted_price'] - p['actual_price']).abs()
    pct_err = (p['predicted_price'] / p['actual_price'] - 1) * 100
    top = err >= err.quantile(0.99)
    ratio_edges = np.arange(-100, 301, 10)
    counts, _ = np.histogram(pct_err.clip(-100, 300), bins=ratio_edges)
    return {
        'n': len(p),
        'sample': p.sample(min(sample_size, len(p)), random_state=cfg.RANDOM_STATE),
        'pct_hist': pd.DataFrame({'lo': ratio_edges[:-1], 'hi': ratio_edges[1:], 'count': counts}),
        'within_25': float((pct_err.abs() <= 25).mean() * 100),
        'within_50': float((pct_err.abs() <= 50).mean() * 100),
        'median_abs_pct': float(pct_err.abs().median()),
        'top1_n': int(top.sum()), 'top1_share': float(err[top].sum() / err.sum() * 100),
        'top1_median_actual': float(p.loc[top, 'actual_price'].median()),
        'all_median_actual': float(p['actual_price'].median()),
        'top1_non_residential': float((p.loc[top, 'property_type'] != 'Residential').mean() * 100),
    }
