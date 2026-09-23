"""
Stage: Hyperparameter Tuning - Work in Progress (not part of Initial Modeling).
Reads initial artifacts when needed; writes only under outputs/tuning/.

Second-round (v2) comparison: validation table, freeze, then ONE test evaluation.

Run: python -m src.tuning.evaluate_v2     (needs train_xgboost_v2.py finished)

v2 candidates
  * XGBoost v2  - v1 settings with min_child_weight = 200
  * Ensemble v2 - equal-weight (50/50) average of XGBoost v2 and the frozen v1
                  CatBoost, averaged in ln(price) (geometric mean of prices).
                  The weights were NOT tuned.

Honesty notes
  * v1 results are untouched (separate files, `_v2` suffix).
  * The v2 design decisions were made on VALIDATION 2024 only. However, Test 2025
    was already looked at once for v1, so it is no longer a perfectly virgin set:
    the v2 test numbers are a second look, not a first.
  * Each v2 model is evaluated on test once; existing test files are never redone.
  * R² is reported on the SAR scale (project metric, dominated by a few giant
    parcels) and also on ln(price) (relative accuracy). Both are shown.
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling import train_xgboost as tx
from src.modeling.evaluation import (improvement_pct, load_json, log_to_price,
                                     per_property_type_metrics, regression_metrics,
                                     save_json, save_predictions, sha256_of_file, stage)

XGB2, ENS2 = cfg.V2_XGB_KEY, cfg.V2_ENSEMBLE_KEY


def metrics_both_scales(actual, predicted):
    """SAR-scale MAE/RMSE/R2 plus R2 on ln(price) and the median absolute error."""
    m = regression_metrics(actual, predicted)
    m['r2_ln_price'] = regression_metrics(np.log(actual), np.log(np.maximum(predicted, 1.0)))['r2']
    m['median_abs_error'] = float(np.median(np.abs(np.asarray(actual) - np.asarray(predicted))))
    return m


def ensemble_prices(price_a, price_b, bounds):
    """Equal-weight average in ln space, clipped to the training range."""
    log_mean = (np.log(np.maximum(price_a, 1.0)) + np.log(np.maximum(price_b, 1.0))) / 2
    return log_to_price(log_mean, bounds)


def load_pred(key, split):
    return pd.read_csv(cfg.predictions_path(key, split), index_col='row_id')


def validation_stage():
    """Ensemble validation predictions + comparison table + freeze."""
    bounds = dl.train_log_bounds()
    x, c = load_pred(XGB2, 'validation'), load_pred('catboost', 'validation')
    if not (x.index.equals(c.index) and np.allclose(x.actual_price, c.actual_price)):
        raise ValueError('XGBoost v2 and CatBoost validation rows are not aligned')
    ens = ensemble_prices(x.predicted_price.to_numpy(), c.predicted_price.to_numpy(), bounds)
    meta = x[['year', 'quarter', 'property_type']]
    save_predictions(cfg.predictions_path(ENS2, 'validation'), meta, x.actual_price.to_numpy(), ens)

    base = load_json(cfg.STATUS_DIR / 'baseline_validation_metrics.json')['metrics']['mae']
    actual = x.actual_price.to_numpy()
    rows = []
    for label, key, pred in [('XGBoost v1', 'xgboost', load_pred('xgboost', 'validation').predicted_price),
                             ('CatBoost v1', 'catboost', c.predicted_price),
                             (cfg.MODEL_LABELS[XGB2], XGB2, x.predicted_price),
                             (cfg.MODEL_LABELS[ENS2], ENS2, ens)]:
        m = metrics_both_scales(actual, np.asarray(pred))
        rows.append({'Model': label, 'Validation MAE (SAR)': m['mae'], 'Validation RMSE (SAR)': m['rmse'],
                     'Validation R2 (SAR)': m['r2'], 'Validation R2 (ln price)': m['r2_ln_price'],
                     'Median abs error (SAR)': m['median_abs_error'],
                     'MAE improvement vs baseline (%)': improvement_pct(base, m['mae'])})
    table = pd.DataFrame(rows)
    table.to_csv(cfg.TUNING_TABLES_DIR / 'v2_validation_comparison.csv', index=False)
    print(table.to_string(index=False))

    if cfg.FROZEN_V2_JSON.exists():
        print('[freeze v2] already frozen -> keeping the original record')
    else:
        save_json({'frozen_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   'ensemble': 'exp(mean(ln(XGBoost v2), ln(CatBoost v1))), weights 0.5/0.5, not tuned',
                   'xgboost_v2_model_sha256': sha256_of_file(cfg.MODEL_FILES[XGB2]),
                   'xgboost_v2_config_sha256': sha256_of_file(cfg.config_path(XGB2)),
                   'catboost_v1_model_sha256': sha256_of_file(cfg.MODEL_FILES['catboost'])},
                  cfg.FROZEN_V2_JSON)


def verify_v2_frozen():
    f = load_json(cfg.FROZEN_V2_JSON)
    checks = [(cfg.MODEL_FILES[XGB2], f['xgboost_v2_model_sha256']),
              (cfg.config_path(XGB2), f['xgboost_v2_config_sha256']),
              (cfg.MODEL_FILES['catboost'], f['catboost_v1_model_sha256'])]
    for path, digest in checks:
        if sha256_of_file(path) != digest:
            raise RuntimeError(f'{path.name} changed after the v2 freeze; refusing to evaluate')


def test_stage():
    """Predict Test 2025 once with XGBoost v2, then build the ensemble from saved v1 CatBoost test predictions."""
    inputs, price = dl.load_inputs('test'), dl.load_price('test')
    dl.check_alignment('test', inputs, price)
    meta = dl.load_meta('test', inputs)
    bounds = dl.train_log_bounds()
    val_x = load_json(cfg.validation_metrics_path(XGB2))
    results = {}
    if cfg.test_metrics_path(XGB2).exists():
        print('[XGBoost v2] test already evaluated -> skipping')
    else:
        model = xgb.XGBRegressor()
        model.load_model(cfg.MODEL_FILES[XGB2])
        log_pred = tx.predict_log(model, dl.build_test_matrix('unscaled'), val_x['best_iteration'] - 1)
        pred = log_to_price(log_pred, bounds)
        table = save_predictions(cfg.predictions_path(XGB2, 'test'), meta, price.to_numpy(), pred)
        save_json({'model': cfg.MODEL_LABELS[XGB2], 'test': metrics_both_scales(price, pred),
                   'per_type': per_property_type_metrics(table, cfg.MODEL_LABELS[XGB2]),
                   'training_seconds': val_x['training_seconds'],
                   'best_iteration': val_x['best_iteration']}, cfg.test_metrics_path(XGB2))
    if cfg.test_metrics_path(ENS2).exists():
        print('[Ensemble v2] test already evaluated -> skipping')
    else:
        x, c = load_pred(XGB2, 'test'), load_pred('catboost', 'test')
        if not (x.index.equals(c.index) and x.index.equals(price.index)):
            raise ValueError('Test predictions are not aligned with the test rows')
        ens = ensemble_prices(x.predicted_price.to_numpy(), c.predicted_price.to_numpy(), bounds)
        table = save_predictions(cfg.predictions_path(ENS2, 'test'), meta, price.to_numpy(), ens)
        save_json({'model': cfg.MODEL_LABELS[ENS2], 'test': metrics_both_scales(price, ens),
                   'per_type': per_property_type_metrics(table, cfg.MODEL_LABELS[ENS2]),
                   'training_seconds': val_x['training_seconds']
                   + load_json(cfg.validation_metrics_path('catboost'))['training_seconds']},
                  cfg.test_metrics_path(ENS2))

    base = load_json(cfg.STATUS_DIR / 'baseline_test_metrics.json')
    base_mae = base['metrics']['mae']
    rows, per_type = [], []
    for label, key in [('XGBoost v1', 'xgboost'), ('CatBoost v1', 'catboost'),
                       (cfg.MODEL_LABELS[XGB2], XGB2), (cfg.MODEL_LABELS[ENS2], ENS2)]:
        p = load_pred(key, 'test')
        m = metrics_both_scales(p.actual_price, p.predicted_price)
        rows.append({'Model': label, 'Test MAE (SAR)': m['mae'], 'Test RMSE (SAR)': m['rmse'],
                     'Test R2 (SAR)': m['r2'], 'Test R2 (ln price)': m['r2_ln_price'],
                     'Median abs error (SAR)': m['median_abs_error'],
                     'MAE improvement vs test baseline (%)': improvement_pct(base_mae, m['mae']),
                     'Met 15% target': 'Yes' if improvement_pct(base_mae, m['mae']) >= cfg.TARGET_IMPROVEMENT_PCT else 'No'})
        per_type += per_property_type_metrics(p, label)
    table = pd.DataFrame(rows)
    table.to_csv(cfg.TUNING_TABLES_DIR / 'v2_test_comparison.csv', index=False)
    pd.DataFrame(per_type).rename(columns={
        'model': 'Model', 'property_type': 'Property type', 'n_rows': 'Test rows',
        'mae': 'MAE (SAR)', 'rmse': 'RMSE (SAR)', 'r2': 'R2', 'r2_note': 'R2 note'}).to_csv(
        cfg.TUNING_TABLES_DIR / 'v2_per_property_type_test_metrics.csv', index=False)
    print(table.to_string(index=False))


def main():
    cfg.make_tuning_output_dirs()
    with stage('v2 / validation comparison + freeze'):
        validation_stage()
    with stage('v2 / verify freeze'):
        verify_v2_frozen()
    with stage('v2 / test evaluation (once)'):
        test_stage()


if __name__ == '__main__':
    main()
