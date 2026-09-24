"""
Stage: Hyperparameter Tuning - ONE Test 2025 evaluation of the frozen final choice.
Writes only under outputs/tuning/.

Run: python -m src.tuning.evaluate_final_test     (after final_comparison.py)

* Refuses to run unless every frozen file still matches its SHA-256.
* Evaluates only the selected model (for a stack: the frozen meta-model applied to
  its two frozen base models). Nothing is fitted; 2025 features come from the saved
  Training 2020-2023 preprocessors (plus the training log1p(area) statistics for an
  MLP that uses them).
* Evaluated once: if its test file exists the script stops. There is no --force.
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd

from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling.evaluation import (count_outside_bounds, improvement_pct, load_json,
                                     log_to_price, per_property_type_metrics, save_json,
                                     save_predictions, sha256_of_file, stage)
from src.tuning import tune_models as tm

TEST_TABLE = cfg.TUNING_TABLES_DIR / 'final_test_comparison.csv'
TEST_PER_TYPE = cfg.TUNING_TABLES_DIR / 'final_test_per_property_type.csv'


def verify_freeze():
    frozen = load_json(cfg.FROZEN_FINAL_JSON)
    for rel, digest in frozen['files'].items():
        if sha256_of_file(cfg.ROOT / rel) != digest:
            raise RuntimeError(f'{rel} changed after the final freeze; refusing to evaluate')
    return frozen['selected_key']


def test_log_prediction(family):
    key = cfg.TUNED_KEYS[family]
    config = load_json(cfg.config_path(key))
    n_iter = int(load_json(cfg.validation_metrics_path(key))['best_iteration'])
    if family == 'xgboost':
        X = dl.build_test_matrix('unscaled')
    elif family == 'catboost':
        X = dl.build_test_catboost_frame()
    else:
        stats = tm.area_log_stats(dl.load_inputs('train')['area'])
        X = tm.mlp_matrix(dl.build_test_matrix('scaled'), dl.load_inputs('test')['area'],
                          config['area_transform'], stats)
    return tm.predict_log(family, tm.load_model(family, cfg.MODEL_FILES[key]), X, n_iter)


def main():
    cfg.make_tuning_output_dirs()
    with stage('final test / verify freeze'):
        key = verify_freeze()
        print(f'  frozen selection: {cfg.MODEL_LABELS[key]}')
    if cfg.test_metrics_path(key).exists():
        print(f'[{cfg.MODEL_LABELS[key]}] test already evaluated -> not repeated')
        return
    with stage('final test / predict 2025'):
        inputs, price = dl.load_inputs('test'), dl.load_price('test')
        dl.check_alignment('test', inputs, price)
        if not inputs['year'].eq(2025).all():
            raise ValueError('Test split contains non-2025 rows')
        if key == cfg.STACK_KEY:
            meta = load_json(cfg.MODEL_FILES[key])
            base = np.column_stack([test_log_prediction(f) for f in meta['base_families']])
            log_pred = meta['intercept'] + base @ np.array([meta['coefficients'][f]
                                                            for f in meta['base_families']])
        else:
            family = {v: k for k, v in cfg.TUNED_KEYS.items()}[key]
            log_pred = test_log_prediction(family)
        bounds = dl.train_log_bounds()
        pred = log_to_price(log_pred, bounds)
        table = save_predictions(cfg.predictions_path(key, 'test'), dl.load_meta('test', inputs),
                                 price.to_numpy(), pred)
    with stage('final test / metrics'):
        m = tm.metrics_both_scales(price, pred)
        base = load_json(cfg.STATUS_DIR / 'baseline_test_metrics.json')
        per_type = per_property_type_metrics(table, cfg.MODEL_LABELS[key])
        save_json({'model_key': key, 'model': cfg.MODEL_LABELS[key], 'test': m,
                   'test_predictions_clipped': count_outside_bounds(log_pred, bounds),
                   'per_type': per_type},
                  cfg.test_metrics_path(key))
        base_m = base['metrics']
        base_frame = pd.read_csv(cfg.predictions_path('baseline', 'test'))
        base_extra = tm.metrics_both_scales(base_frame['actual_price'], base_frame['predicted_price'])
        rows = [{'Model': cfg.MODEL_LABELS['baseline'], 'Test MAE (SAR)': base_m['mae'],
                 'Test RMSE (SAR)': base_m['rmse'], 'Test R2 (SAR)': base_m['r2'],
                 'Test R2 (ln price)': base_extra['r2_ln_price'],
                 'Median abs error (SAR)': base_extra['median_abs_error'],
                 'MAE improvement vs test baseline (%)': 0.0, 'Met 15% target': 'n/a (reference)'},
                {'Model': cfg.MODEL_LABELS[key], 'Test MAE (SAR)': m['mae'],
                 'Test RMSE (SAR)': m['rmse'], 'Test R2 (SAR)': m['r2'],
                 'Test R2 (ln price)': m['r2_ln_price'], 'Median abs error (SAR)': m['median_abs_error'],
                 'MAE improvement vs test baseline (%)': improvement_pct(base_m['mae'], m['mae']),
                 'Met 15% target': 'Yes' if improvement_pct(base_m['mae'], m['mae'])
                 >= cfg.TARGET_IMPROVEMENT_PCT else 'No'}]
        out = pd.DataFrame(rows)
        out.to_csv(TEST_TABLE, index=False)
        per = pd.DataFrame(base['per_type'] + per_type).rename(columns={
            'model': 'Model', 'property_type': 'Property type', 'n_rows': 'Test rows',
            'mae': 'MAE (SAR)', 'rmse': 'RMSE (SAR)', 'r2': 'R2 (SAR)', 'r2_note': 'R2 note'})
        per.to_csv(TEST_PER_TYPE, index=False)
        pd.set_option('display.width', 250, 'display.float_format', lambda x: f'{x:,.4f}')
        print(out.to_string(index=False))
        print(per.drop(columns='R2 note').to_string(index=False))


if __name__ == '__main__':
    main()
