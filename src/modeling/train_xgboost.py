"""
Train the initial XGBoost model (no tuning) and evaluate it on Validation 2024.

Run:   python -m src.modeling.train_xgboost            (skips if already finished)
       python -m src.modeling.train_xgboost --force    (retrain from scratch)

Input : unscaled engineered matrices (X_*_unscaled.npy) and ln(price) targets.
Uses  : 2020-2023 to fit, 2024 for early stopping and evaluation. 2025 is never read.
"""
import argparse
import time

import numpy as np
import xgboost as xgb

from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling.evaluation import (count_outside_bounds, log_to_price, regression_metrics,
                                     save_validation_results, stage, validation_is_complete)

KEY = 'xgboost'

# Initial (untuned) configuration. Moderate depth and learning rate; the tree
# count is a generous ceiling and early stopping picks the useful number.
XGB_CONFIG = {
    'objective': 'reg:squarederror',   # squared error on ln(price)
    'eval_metric': 'rmse',             # measured on ln(price) for early stopping
    'tree_method': 'hist',             # fast histogram algorithm for large data
    'n_estimators': 3000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'early_stopping_rounds': 50,
    'n_jobs': -1,                      # all CPU cores
    'random_state': cfg.RANDOM_STATE,
}


def fit_xgboost(X_train, y_train, X_val, y_val, overrides=None, verbose=100):
    """Fit with early stopping on the validation set. Returns (model, config, seconds)."""
    config = {**XGB_CONFIG, **(overrides or {})}
    model = xgb.XGBRegressor(**config)
    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
    return model, config, time.perf_counter() - start


def predict_log(model, X, best_iteration):
    """Predict ln(price) using only the trees up to the best iteration."""
    return model.predict(X, iteration_range=(0, best_iteration + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='retrain even if finished')
    args = parser.parse_args()
    cfg.make_output_dirs()
    if validation_is_complete(KEY) and not args.force:
        print(f'[XGBoost] already finished -> skipping (use --force to retrain)')
        return

    with stage('XGBoost / load data'):
        inputs_tr, price_tr = dl.load_inputs('train'), dl.load_price('train')
        inputs_va, price_va = dl.load_inputs('validation'), dl.load_price('validation')
        y_tr, y_va = dl.load_log_target('train'), dl.load_log_target('validation')
        dl.check_alignment('train', inputs_tr, price_tr, y_tr)
        dl.check_alignment('validation', inputs_va, price_va, y_va)
        X_tr, X_va = dl.load_matrix('unscaled', 'train'), dl.load_matrix('unscaled', 'validation')
        if len(X_tr) != len(y_tr) or len(X_va) != len(y_va):
            raise ValueError('Feature matrix and target row counts differ')
        names = dl.feature_names()['unscaled']
        if X_tr.shape[1] != len(names) or X_va.shape[1] != len(names):
            raise ValueError('Feature matrix width does not match feature_names.json')
        meta_va = dl.load_meta('validation', inputs_va)
        print(f'  train {X_tr.shape}, validation {X_va.shape}, features: {names}')

    with stage('XGBoost / fit (early stopping on 2024)'):
        model, config, seconds = fit_xgboost(X_tr, y_tr.to_numpy(), X_va, y_va.to_numpy())
        best = int(model.best_iteration)
        print(f'  best iteration (0-based): {best}  -> {best + 1} trees kept; '
              f'training time {seconds:.1f}s')

    with stage('XGBoost / evaluate on train and validation'):
        bounds = dl.train_log_bounds()
        log_tr, log_va = predict_log(model, X_tr, best), predict_log(model, X_va, best)
        pred_tr, pred_va = log_to_price(log_tr, bounds), log_to_price(log_va, bounds)
        n_clipped = count_outside_bounds(log_va, bounds)
        print(f"  validation predictions clipped to the training range: {n_clipped}")
        train_metrics = regression_metrics(price_tr, pred_tr)
        cfg.MODEL_FILES[KEY].parent.mkdir(parents=True, exist_ok=True)
        model.save_model(cfg.MODEL_FILES[KEY])

    with stage('XGBoost / save results'):
        result = save_validation_results(
            KEY, {**config, 'feature_names': names, 'target': cfg.TARGET_TRANSFORM,
                  'best_iteration_0based': best, 'trees_kept': best + 1,
                  'xgboost_version': xgb.__version__},
            meta_va, price_va.to_numpy(), pred_va, train_metrics, seconds, best + 1,
            extra={'validation_predictions_clipped': n_clipped,
                   'log_clip_bounds': list(bounds), 'iteration_detail': f'{best + 1} trees kept (early stopping)'})
        print(f"  validation MAE {result['validation']['mae']:,.2f} SAR | "
              f"RMSE {result['validation']['rmse']:,.2f} | R2 {result['validation']['r2']:.4f}")


if __name__ == '__main__':
    main()
