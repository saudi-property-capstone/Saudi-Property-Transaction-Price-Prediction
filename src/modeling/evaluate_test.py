"""
Final one-time evaluation of the FROZEN initial models on Test 2025.

Run: python -m src.modeling.evaluate_test

Safeguards
* Refuses to run unless compare_validation.py has frozen the models and the
  model/config files still match their recorded SHA-256 hashes.
* Each model (and the baseline) is evaluated ONCE; a model that already has a
  test-metrics file is skipped and never re-evaluated. There is no --force.
* Nothing is fitted here. Test features come from the saved training-fitted
  preprocessors; the models only predict.
* The tuning recommendation uses VALIDATION MAE only, not test results.
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


import numpy as np
import pandas as pd

from src.modeling import baseline as baseline_module
from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling import initial_tables
from src.modeling.evaluation import (count_outside_bounds, load_json, log_to_price,
                                     per_property_type_metrics, regression_metrics,
                                     save_json, save_predictions, sha256_of_file, stage)


def verify_frozen():
    if not cfg.FROZEN_CONFIG_JSON.exists():
        raise RuntimeError('Models are not frozen. Run compare_validation.py first.')
    frozen = load_json(cfg.FROZEN_CONFIG_JSON)['models']
    for key in cfg.MODEL_KEYS:
        for path_key, hash_key in (('model_file', 'model_sha256'), ('config_file', 'config_sha256')):
            if sha256_of_file(cfg.ROOT / frozen[key][path_key]) != frozen[key][hash_key]:
                raise RuntimeError(f'{key}: {path_key} changed after the freeze; '
                                   'refusing to evaluate on Test 2025')
    print('[freeze check] all model and config files match the freeze record')


def predict_test_log(key, val_result):
    """Load a frozen model and predict ln(price) for the 2025 rows."""
    if key == 'xgboost':
        import xgboost as xgb
        from src.modeling.train_xgboost import predict_log
        model = xgb.XGBRegressor()
        model.load_model(cfg.MODEL_FILES[key])
        return predict_log(model, dl.build_test_matrix('unscaled'),
                           val_result['best_iteration'] - 1)
    if key == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(cfg.MODEL_FILES[key]))
        return model.predict(dl.build_test_catboost_frame())
    if key == 'mlp':
        from tensorflow import keras
        model = keras.models.load_model(cfg.MODEL_FILES[key])
        return model.predict(dl.build_test_matrix('scaled'), batch_size=8192, verbose=0).ravel()
    raise ValueError(key)


def evaluate_model_on_test(key, meta, price):
    val_result = load_json(cfg.validation_metrics_path(key))
    log_pred = predict_test_log(key, val_result)
    if len(log_pred) != len(price):
        raise ValueError(f'{key}: {len(log_pred)} predictions for {len(price)} test rows')
    bounds = dl.train_log_bounds()      # training-only range, same rule as validation
    pred = log_to_price(log_pred, bounds)
    table = save_predictions(cfg.predictions_path(key, 'test'), meta, price.to_numpy(), pred)
    result = {
        'model_key': key, 'model': cfg.MODEL_LABELS[key], 'version': 'Initial - Untuned',
        'test': regression_metrics(price, pred),
        'test_predictions_clipped': count_outside_bounds(log_pred, bounds),
        'per_type': per_property_type_metrics(table, cfg.MODEL_LABELS[key]),
        'training_seconds': val_result['training_seconds'],
        'best_iteration': val_result['best_iteration'],
        'iteration_detail': val_result['iteration_detail'],
    }
    save_json(result, cfg.test_metrics_path(key))
    return result


def build_selection_summary():
    """Best models plus the tuning recommendation (validation MAE only)."""
    val = {k: load_json(cfg.validation_metrics_path(k))['validation']['mae'] for k in cfg.MODEL_KEYS}
    test = {k: load_json(cfg.test_metrics_path(k))['test']['mae'] for k in cfg.MODEL_KEYS}
    by_val = sorted(val, key=val.get)
    return {
        'best_by_validation_mae': cfg.MODEL_LABELS[by_val[0]],
        'best_by_test_mae': cfg.MODEL_LABELS[min(test, key=test.get)],
        'recommended_for_tuning': [cfg.MODEL_LABELS[k] for k in by_val[:2]],
        'recommendation_rule': 'the two models with the lowest Validation 2024 MAE; '
                               'test results were not used for this choice',
        'validation_mae': {cfg.MODEL_LABELS[k]: v for k, v in val.items()},
        'test_mae': {cfg.MODEL_LABELS[k]: v for k, v in test.items()},
    }


def main():
    cfg.make_output_dirs()
    with stage('test / verify freeze'):
        verify_frozen()
    with stage('test / load 2025 labels and metadata'):
        inputs, price = dl.load_inputs('test'), dl.load_price('test')
        dl.check_alignment('test', inputs, price)
        if not inputs['year'].eq(2025).all():
            raise ValueError('Test split contains non-2025 rows')
        meta = dl.load_meta('test', inputs)
        print(f'  test rows: {len(price):,}')
    base_path = cfg.STATUS_DIR / 'baseline_test_metrics.json'
    if base_path.exists():
        print('[baseline] test already evaluated -> skipping')
    else:
        save_json(baseline_module.run_baseline('test'), base_path)
    for key in cfg.MODEL_KEYS:
        if cfg.test_metrics_path(key).exists():
            print(f'[{cfg.MODEL_LABELS[key]}] test already evaluated -> skipping (evaluate once)')
            continue
        with stage(f'test / {cfg.MODEL_LABELS[key]} predict + metrics'):
            r = evaluate_model_on_test(key, meta, price)
            print(f"  test MAE {r['test']['mae']:,.2f} | RMSE {r['test']['rmse']:,.2f} | "
                  f"R2 {r['test']['r2']:.4f}")
    with stage('test / comparison tables and selection summary'):
        table = initial_tables.build_comparison('test')
        per_type = initial_tables.build_per_type_test()
        table.to_csv(cfg.TABLES_DIR / 'initial_test_comparison.csv', index=False)
        per_type.to_csv(cfg.TABLES_DIR / 'initial_per_property_type_test_metrics.csv', index=False)
        summary = build_selection_summary()
        save_json(summary, cfg.TABLES_DIR / 'model_selection_summary.json')
        pd.set_option('display.width', 250, 'display.max_columns', 20,
                      'display.float_format', lambda x: f'{x:,.4f}')
        print(table.to_string(index=False))
        print(per_type.drop(columns='R2 note').to_string(index=False))
        print(summary)
        # Per-type row counts must add up to the full test set for every model.
        totals = per_type.groupby('Model')['Test rows'].sum()
        if not (totals == len(price)).all():
            raise ValueError(f'Per-type row counts do not sum to {len(price)}: {totals.to_dict()}')


if __name__ == '__main__':
    main()
