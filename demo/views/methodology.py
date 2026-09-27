"""Methodology / About: how the project was built, from raw files to the deployed model."""
import json

import pandas as pd
import streamlit as st

from dashboard import data
from dashboard import theme as th
from dashboard.fmt import count
from src.modeling import config as cfg

th.render_page_header('Methodology',
                      'How the project goes from 24 raw MOJ files to a frozen, explained price model.',
                      icon='account_tree')

th.section('Pipeline')
th.steps([
    ('Data collection', '24 quarterly MOJ sales files, 2020 Q1 – 2025 Q4'),
    ('Data cleaning', 'schema harmonisation, region mapping, invalid / bulk / duplicate / suspicious rows removed'),
    ('Exploratory analysis', 'volume, price, area, regional and seasonal patterns'),
    ('Chronological split', 'train 2020–2023 · validate 2024 · test 2025'),
    ('Feature engineering', 'time index, quarter sin/cos, location frequencies, region target encoding; '
                            'fitted on training data only'),
    ('Baseline', 'historical median price by district + property type'),
    ('ML / DL models', 'XGBoost, CatBoost, MLP; random-search tuning'),
    ('Stacking', 'Ridge combining the two best tuned models (out-of-fold), a new candidate'),
    ('Final selection', 'tuned models vs stack, lowest Validation 2024 MAE wins; frozen before the test'),
    ('Test evaluation', 'MAE, RMSE, R² in SAR; evaluated once on 2025'),
    ('Interpretability', 'exact SHAP values of the final model, grouped by characteristic'),
    ('Interactive dashboard', 'Streamlit app to explore the data, review the results and estimate prices '
                              'with the final model'),
])

# ---------------- Split ----------------
th.section('Chronological train / validation / test split')
split = data.split_report()
c1, c2 = st.columns([1, 1.3], gap='large')
with c1:
    if split:
        s = split['splits']
        st.dataframe(pd.DataFrame([('Training', '2020–2023', s['train']['retained_rows']),
                                   ('Validation', '2024', s['validation']['retained_rows']),
                                   ('Test', '2025', s['test']['retained_rows'])],
                                  columns=['Split', 'Years', 'Transactions']),
                     hide_index=True, width='stretch',
                     column_config={'Transactions': st.column_config.NumberColumn(format='localized')})
        overlap = split.get('reference_overlap_counts', {})
        if overlap and not any(overlap.values()):
            th.note('No MOJ reference number appears in more than one split.')
with c2:
    st.markdown(
        '**Why split by time instead of at random?** The model is meant to price *future* transactions. A random '
        'split would mix 2025 sales into training, letting the model learn from prices of the same period and '
        'market conditions it is evaluated on, which overstates accuracy. Training on the past, choosing models '
        'on 2024 and testing once on 2025 mirrors real use and exposes the effect of market drift (prices rose '
        'strongly over 2020–2025).')
    
# ---------------- Target & baseline ----------------
th.section('Target, baseline and success criterion')
th.info_cards([
    ('Target', 'Natural log of price',
     'Prices are highly skewed, so every model predicts ln(price). Predictions are converted back with exp() and '
     'clipped to the training range before any metric is computed, so all reported metrics are in SAR.'),
    ('Baseline', 'Historical median price',
     'A rule with no machine learning: the median 2020–2023 sale price of the same district and property type. '
     'If that combination never occurred in training, it falls back to the property-type median, then the overall '
     'median.'),
    ('Success criterion', f'≥ {cfg.TARGET_IMPROVEMENT_PCT:.0f}% lower MAE than the baseline',
     'MAE in SAR is the primary metric. RMSE, R² (in SAR and on ln price) and the median absolute error are '
     'reported alongside it.'),
])

# ---------------- Models ----------------
th.section('Models')


@st.cache_data(show_spinner=False)
def trial_counts():
    out = {}
    for m in ('xgboost', 'catboost', 'mlp'):
        path = cfg.TUNING_TABLES_DIR / f'tuning_trials_{m}.csv'
        if path.exists():
            out[m] = len(pd.read_csv(path))
    return out


@st.cache_data(show_spinner=False)
def stack_meta():
    path = cfg.MODEL_FILES[cfg.STACK_KEY]
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


trials = trial_counts()
meta = stack_meta()
t = lambda m: f' {trials[m]} random-search trials.' if m in trials else ''  # noqa: E731
th.info_cards([
    ('Gradient boosting', 'XGBoost',
     'Engineered numeric features: area, quarter time index, quarter sin/cos, city and district frequencies, '
     'target-encoded region and one-hot property type.' + t('xgboost')),
    ('Gradient boosting', 'CatBoost',
     'Same numeric features, with region, city, district and property type kept as native categories (no '
     'frequency or target encoding).' + t('catboost')),
    ('Deep learning', 'MLP neural network',
     'Scaled version of the XGBoost features, built with TensorFlow / Keras. It did not beat the tree models; the '
     'result is reported as-is.' + t('mlp')),
])
st.write('')
if meta:
    folds = ', '.join(str(f['held_out_year']) for f in meta.get('oof_scheme', []))
    w = meta['coefficients']
    th.info_cards([
        ('Final model', 'Ridge stack of the two best tuned models',
         f'A Ridge regression combines the ln(price) predictions of Tuned XGBoost (weight {w["xgboost"]:.2f}) and '
         f'Tuned CatBoost (weight {w["catboost"]:.2f}). It is fit on forward-chaining out-of-fold predictions '
         f'(each of the years {folds} predicted by models trained only on earlier years), so the stack never sees '
         'in-sample predictions.'),
        ('Selection', 'Validation only, test once',
         'Hyperparameters, the stack and the final model were all chosen on Validation 2024 MAE. The chosen model '
         'files were then frozen (checksums saved) before the single evaluation on Test 2025.'),
        ('Interpretability', 'Exact SHAP values',
         'SHAP contributions from XGBoost and CatBoost’s built-in exact tree explainers, weighted by the stack and '
         'grouped into five characteristics: location, area, time trend, property type and season.'),
    ])

# ---------------- Limits ----------------
th.section('Scope and future work')
st.markdown(
    '- The model uses only what the public MOJ data contains: area, date, location and a broad property type. '
    'There is no building age, number of rooms, floor area or detailed property sub-type.\n'
    '- Most of the remaining error comes from rare, very large Commercial and Agricultural deals whose price depends '
    'on such missing attributes. Adding property-attribute data is the most promising way to improve accuracy.\n'
    '- Estimates are indicative market figures, not a professional valuation.')

# ---------------- Stack ----------------
th.section('Technology stack')
th.chips(['Python', 'pandas', 'NumPy', 'scikit-learn', 'XGBoost', 'CatBoost', 'TensorFlow / Keras'])
th.chips(['Matplotlib', 'Seaborn', 'Plotly', 'Streamlit', 'Jupyter'], neutral=True)
th.note('Modeling libraries in green; analysis, visualisation and app libraries in grey. '
        f'Dataset: {count(sum(split["splits"][k]["retained_rows"] for k in split["splits"])) if split else "~1.39M"} '
        'cleaned MOJ transactions.')
