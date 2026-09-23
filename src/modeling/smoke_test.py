"""
Automated smoke test - run BEFORE full training.

Run: python -m src.modeling.smoke_test      (exit code 0 = passed, 1 = failed)

Fits tiny versions of all three models on a reproducible random sample of
Training 2020-2023 (random_state=42) and a small sample of Validation 2024, and
checks that the whole path works: data alignment, column consistency, dtypes,
NaN/inf, fit, predict, exp() inversion, metrics, saving, package versions.

Diagnostic only: its numbers come from tiny models on tiny samples and must
never be reported as results. Test 2025 LABELS are never read here; only the
test FEATURES are built (from raw inputs with the saved train-fitted
preprocessors) to confirm their columns and values are valid.
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import sys
from importlib import metadata

import numpy as np
import pandas as pd

from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling import train_catboost, train_mlp, train_xgboost
from src.modeling.evaluation import (load_json, log_to_price, regression_metrics,
                                     save_json, save_predictions, stage)

TRAIN_SAMPLE, VAL_SAMPLE = 20_000, 5_000
REQUIRED_PACKAGES = ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'catboost',
                     'tensorflow', 'keras', 'matplotlib']


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f'  ok: {message}')


def check_numeric_matrix(name, matrix, n_cols):
    check(matrix.dtype.kind == 'f', f'{name} is numeric ({matrix.dtype}), no string columns')
    check(matrix.shape[1] == n_cols, f'{name} has {n_cols} columns')
    check(not np.isnan(matrix).any(), f'{name} has no NaN')
    check(np.isfinite(matrix).all(), f'{name} has no infinite values')


def sample_positions(n_rows, n_sample):
    """Reproducible sorted random row positions (random_state=42)."""
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    return np.sort(rng.choice(n_rows, size=n_sample, replace=False))


def run():
    report = {'diagnostic_only': True, 'note': 'Tiny models on samples; NOT final results.',
              'random_state': cfg.RANDOM_STATE, 'checks': []}
    cfg.make_output_dirs()
    cfg.SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    with stage('smoke / packages'):
        report['package_versions'] = {p: metadata.version(p) for p in REQUIRED_PACKAGES}
        print('  ', report['package_versions'])

    with stage('smoke / load full train + validation'):
        inputs = {s: dl.load_inputs(s) for s in ('train', 'validation')}
        price = {s: dl.load_price(s) for s in ('train', 'validation')}
        logy = {s: dl.load_log_target(s) for s in ('train', 'validation')}
        for s in inputs:
            dl.check_alignment(s, inputs[s], price[s], logy[s])
            check(len(inputs[s]) == len(price[s]) == len(logy[s]),
                  f'{s}: X, price and log target have {len(price[s]):,} matching rows')
        check(inputs['train']['year'].between(2020, 2023).all(), 'train years are 2020-2023')
        check(inputs['validation']['year'].eq(2024).all(), 'validation year is 2024')
        check(logy['train'].notna().all() and np.isfinite(logy['train']).all(),
              'log target finite (already ln(price); not transformed again)')

    with stage('smoke / feature consistency and values'):
        names = dl.feature_names()
        unscaled = {s: dl.load_matrix('unscaled', s) for s in ('train', 'validation')}
        scaled = {s: dl.load_matrix('scaled', s) for s in ('train', 'validation')}
        cat = {s: dl.load_catboost_frame(s) for s in ('train', 'validation')}
        for s in ('train', 'validation'):
            check(len(unscaled[s]) == len(scaled[s]) == len(cat[s]) == len(price[s]),
                  f'{s}: all feature variants have the same rows as the target')
            check_numeric_matrix(f'{s} unscaled (XGBoost)', unscaled[s], len(names['unscaled']))
            check_numeric_matrix(f'{s} scaled (MLP)', scaled[s], len(names['scaled']))
        check(list(cat['train'].columns) == list(cat['validation'].columns) == names['catboost'],
              'CatBoost train/validation columns are identical and match feature_names.json')
        check(names['unscaled'] == names['scaled'], 'unscaled and scaled column names match')
        forbidden = [c for c in names['unscaled'] + names['catboost']
                     if any(t in c.lower() for t in ('price', 'reference', 'row_id'))]
        check(not forbidden, 'no price / price-per-m2 / reference_number / row_id in features')
        for col in train_catboost.CATBOOST_CATEGORICAL:
            check(cat['train'][col].map(type).eq(str).all() and not cat['train'][col].isna().any(),
                  f'CatBoost column {col}: strings only, no missing values')
        check(np.isfinite(cat['train'][dl.CATBOOST_NUMERIC].to_numpy(float)).all(),
              'CatBoost numeric columns are finite')
        # The saved preprocessors must reproduce the exported validation matrices.
        rows = sample_positions(len(inputs['validation']), 2000)
        for variant, exported in (('unscaled', unscaled['validation']),
                                  ('scaled', scaled['validation'])):
            again = dl.transform_with_saved_preprocessor(variant, inputs['validation'].iloc[rows])
            check(np.allclose(again, exported[rows], atol=1e-5),
                  f'saved {variant} preprocessor reproduces exported validation matrix')
        # Test FEATURES only (no labels are read): same columns, valid values.
        for variant in ('unscaled', 'scaled'):
            test_matrix = dl.build_test_matrix(variant)
            check_numeric_matrix(f'test {variant}', test_matrix, len(names[variant]))
        test_cat = dl.build_test_catboost_frame()
        check(list(test_cat.columns) == names['catboost'], 'test CatBoost columns match train')
        check(set(test_cat['property_type']) <= set(cfg.PROPERTY_TYPES),
              'test property types are the three known types')

    with stage('smoke / sample train + validation (random_state=42)'):
        tr = sample_positions(len(price['train']), TRAIN_SAMPLE)
        va = sample_positions(len(price['validation']), VAL_SAMPLE)
        check(inputs['train'].iloc[tr]['year'].between(2020, 2023).all(),
              f'train sample of {TRAIN_SAMPLE:,} rows is only 2020-2023')
        y_tr, y_va = logy['train'].to_numpy()[tr], logy['validation'].to_numpy()[va]
        price_va = price['validation'].to_numpy()[va]
        meta_va = dl.load_meta('validation', inputs['validation']).iloc[va]

    results = {}
    with stage('smoke / XGBoost tiny fit'):
        model, _, _ = train_xgboost.fit_xgboost(
            unscaled['train'][tr], y_tr, unscaled['validation'][va], y_va,
            overrides={'n_estimators': 30, 'early_stopping_rounds': 5}, verbose=False)
        results['xgboost'] = train_xgboost.predict_log(model, unscaled['validation'][va],
                                                       int(model.best_iteration))
    with stage('smoke / CatBoost tiny fit'):
        model, _, _ = train_catboost.fit_catboost(
            cat['train'].iloc[tr], y_tr, cat['validation'].iloc[va], y_va,
            overrides={'iterations': 30, 'early_stopping_rounds': 5}, verbose=False)
        results['catboost'] = model.predict(cat['validation'].iloc[va])
    with stage('smoke / MLP tiny fit'):
        model, history, _, _ = train_mlp.fit_mlp(
            scaled['train'][tr], y_tr.astype(np.float32), scaled['validation'][va],
            y_va.astype(np.float32), overrides={'max_epochs': 2}, verbose=0)
        check(len(history['loss']) == 2, 'MLP ran 2 epochs and recorded history')
        results['mlp'] = train_mlp.predict_log(model, scaled['validation'][va])

    with stage('smoke / predict, invert, metrics, save'):
        for key, log_pred in results.items():
            check(len(log_pred) == VAL_SAMPLE and np.isfinite(log_pred).all(),
                  f'{key}: predictions finite and one per validation row')
            pred_price = log_to_price(log_pred, dl.train_log_bounds())  # exp(), the project's inverse
            check((pred_price >= 0).all(), f'{key}: SAR predictions non-negative')
            metrics = regression_metrics(price_va, pred_price)
            check(all(np.isfinite(metrics[m]) for m in ('mae', 'rmse', 'r2')),
                  f'{key}: MAE/RMSE/R2 computed (diagnostic MAE {metrics["mae"]:,.0f})')
            path = cfg.SMOKE_DIR / f'smoke_{key}_predictions.csv.gz'
            save_predictions(path, meta_va, price_va, pred_price)
            saved = pd.read_csv(path, index_col='row_id')
            check(len(saved) == VAL_SAMPLE and {'year', 'property_type'} <= set(saved.columns),
                  f'{key}: predictions file saved and readable')
            report[f'{key}_diagnostic_metrics'] = metrics

    report['status'] = 'passed'
    save_json(report, cfg.SMOKE_DIR / 'smoke_test_report.json')
    check(load_json(cfg.SMOKE_DIR / 'smoke_test_report.json')['status'] == 'passed',
          'smoke report saved and reloaded')
    return report


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:                       # any failure -> clear message, exit 1
        print(f'\nSMOKE TEST FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
    print('\nSMOKE TEST PASSED (diagnostic only - not final results)')
