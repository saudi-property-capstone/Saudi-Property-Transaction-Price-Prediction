"""
Stage: Hyperparameter Tuning - Work in Progress (not part of Initial Modeling).
Reads initial artifacts when needed; writes only under outputs/tuning/.

Train XGBoost v2: identical to the frozen v1 configuration except
min_child_weight = 200 (v1: 5), chosen from a small validation-only experiment
(experiments_v2_xgboost.py). Every leaf must now be supported by >= 200 rows,
which stops the trees from copying a handful of huge training prices onto
large-area rows (the main cause of the R² collapse in SAR).

Run:   python -m src.tuning.train_xgboost_v2            (skips if finished)
       python -m src.tuning.train_xgboost_v2 --force
Uses Train 2020-2023 for fitting and Validation 2024 for early stopping.
Test 2025 is not read here.
"""
import argparse

import xgboost as xgb

from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling import train_xgboost as tx
from src.modeling.evaluation import (count_outside_bounds, log_to_price, regression_metrics,
                                     save_validation_results, stage, validation_is_complete)

KEY = cfg.V2_XGB_KEY
V2_OVERRIDES = {'min_child_weight': 200}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    cfg.make_tuning_output_dirs()
    if validation_is_complete(KEY) and not args.force:
        print('[XGBoost v2] already finished -> skipping (use --force to retrain)')
        return
    with stage('XGBoost v2 / load data'):
        inputs_tr, price_tr = dl.load_inputs('train'), dl.load_price('train')
        inputs_va, price_va = dl.load_inputs('validation'), dl.load_price('validation')
        y_tr, y_va = dl.load_log_target('train'), dl.load_log_target('validation')
        dl.check_alignment('train', inputs_tr, price_tr, y_tr)
        dl.check_alignment('validation', inputs_va, price_va, y_va)
        X_tr, X_va = dl.load_matrix('unscaled', 'train'), dl.load_matrix('unscaled', 'validation')
        names = dl.feature_names()['unscaled']
        meta_va = dl.load_meta('validation', inputs_va)
    with stage('XGBoost v2 / fit (early stopping on 2024)'):
        model, config, seconds = tx.fit_xgboost(X_tr, y_tr.to_numpy(), X_va, y_va.to_numpy(),
                                                overrides=V2_OVERRIDES)
        best = int(model.best_iteration)
        print(f'  {best + 1} trees kept; training time {seconds:.1f}s')
    with stage('XGBoost v2 / evaluate and save'):
        bounds = dl.train_log_bounds()
        log_tr, log_va = tx.predict_log(model, X_tr, best), tx.predict_log(model, X_va, best)
        pred_tr, pred_va = log_to_price(log_tr, bounds), log_to_price(log_va, bounds)
        train_metrics = regression_metrics(price_tr, pred_tr)
        cfg.MODEL_FILES[KEY].parent.mkdir(parents=True, exist_ok=True)
        model.save_model(cfg.MODEL_FILES[KEY])
        y_va_ln = regression_metrics(y_va, log_va.clip(*bounds))
        result = save_validation_results(
            KEY, {**config, 'feature_names': names, 'target': cfg.TARGET_TRANSFORM,
                  'changed_from_v1': {'min_child_weight': '5 -> 200'},
                  'best_iteration_0based': best, 'trees_kept': best + 1,
                  'xgboost_version': xgb.__version__},
            meta_va, price_va.to_numpy(), pred_va, train_metrics, seconds, best + 1,
            extra={'validation_r2_ln_price': y_va_ln['r2'],
                   'validation_predictions_clipped': count_outside_bounds(log_va, bounds),
                   'log_clip_bounds': list(bounds),
                   'iteration_detail': f'{best + 1} trees kept (early stopping)'})
        print(f"  validation MAE {result['validation']['mae']:,.2f} | "
              f"R2 {result['validation']['r2']:.4f} | R2(ln) {y_va_ln['r2']:.4f}")


if __name__ == '__main__':
    main()
