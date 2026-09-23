"""
Train the initial CatBoost model (no tuning) and evaluate it on Validation 2024.

Run:   python -m src.modeling.train_catboost            (skips if already finished)
       python -m src.modeling.train_catboost --force    (retrain from scratch)

Input : the native-categorical files from prepare_features.py
        (X_*_catboost.csv.gz): region, city, city_district, property_type stay
        as strings and CatBoost encodes them itself. No frequency, target, or
        one-hot encodings are mixed in.
Uses  : 2020-2023 to fit, 2024 for early stopping and evaluation. 2025 is never read.
"""
import argparse
import time

import catboost
from catboost import CatBoostRegressor, Pool

from src.feature_engineering import CATBOOST_CATEGORICAL
from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling.evaluation import (count_outside_bounds, log_to_price, regression_metrics,
                                     save_validation_results, stage, validation_is_complete)

KEY = 'catboost'

# Initial (untuned) configuration. The iteration count is a generous ceiling;
# early stopping (od_wait) picks the useful number and use_best_model keeps it.
CATBOOST_CONFIG = {
    'loss_function': 'RMSE',           # squared error on ln(price)
    'eval_metric': 'RMSE',             # measured on ln(price) for early stopping
    'iterations': 3000,
    'learning_rate': 0.1,
    'depth': 8,
    'l2_leaf_reg': 3.0,
    'early_stopping_rounds': 50,
    'use_best_model': True,
    'thread_count': -1,                # all CPU cores
    'random_seed': cfg.RANDOM_STATE,
    'allow_writing_files': False,      # no catboost_info/ folder in the repo
}


def fit_catboost(frame_train, y_train, frame_val, y_val, overrides=None, verbose=100):
    """Fit with native categoricals and early stopping on validation."""
    config = {**CATBOOST_CONFIG, **(overrides or {})}
    train_pool = Pool(frame_train, y_train, cat_features=CATBOOST_CATEGORICAL)
    val_pool = Pool(frame_val, y_val, cat_features=CATBOOST_CATEGORICAL)
    model = CatBoostRegressor(**config, verbose=verbose)
    start = time.perf_counter()
    model.fit(train_pool, eval_set=val_pool)
    return model, config, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='retrain even if finished')
    args = parser.parse_args()
    cfg.make_output_dirs()
    if validation_is_complete(KEY) and not args.force:
        print('[CatBoost] already finished -> skipping (use --force to retrain)')
        return

    with stage('CatBoost / load data'):
        inputs_tr, price_tr = dl.load_inputs('train'), dl.load_price('train')
        inputs_va, price_va = dl.load_inputs('validation'), dl.load_price('validation')
        y_tr, y_va = dl.load_log_target('train'), dl.load_log_target('validation')
        dl.check_alignment('train', inputs_tr, price_tr, y_tr)
        dl.check_alignment('validation', inputs_va, price_va, y_va)
        F_tr, F_va = dl.load_catboost_frame('train'), dl.load_catboost_frame('validation')
        if not (F_tr.index.equals(y_tr.index) and F_va.index.equals(y_va.index)):
            raise ValueError('CatBoost feature rows are not aligned with the targets')
        if list(F_tr.columns) != dl.feature_names()['catboost'] or list(F_va.columns) != list(F_tr.columns):
            raise ValueError('CatBoost columns do not match feature_names.json')
        meta_va = dl.load_meta('validation', inputs_va)
        print(f'  train {F_tr.shape}, validation {F_va.shape}, features: {list(F_tr.columns)}')
        print(f'  categorical: {CATBOOST_CATEGORICAL}')

    with stage('CatBoost / fit (early stopping on 2024)'):
        model, config, seconds = fit_catboost(F_tr, y_tr.to_numpy(), F_va, y_va.to_numpy())
        best = int(model.get_best_iteration())
        print(f'  best iteration (0-based): {best} -> {best + 1} trees kept; '
              f'training time {seconds:.1f}s')

    with stage('CatBoost / evaluate on train and validation'):
        bounds = dl.train_log_bounds()
        log_tr, log_va = model.predict(F_tr), model.predict(F_va)
        pred_tr, pred_va = log_to_price(log_tr, bounds), log_to_price(log_va, bounds)
        n_clipped = count_outside_bounds(log_va, bounds)
        print(f"  validation predictions clipped to the training range: {n_clipped}")
        train_metrics = regression_metrics(price_tr, pred_tr)
        cfg.MODEL_FILES[KEY].parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(cfg.MODEL_FILES[KEY]))

    with stage('CatBoost / save results'):
        result = save_validation_results(
            KEY, {**config, 'feature_names': list(F_tr.columns),
                  'categorical_features': CATBOOST_CATEGORICAL,
                  'target': cfg.TARGET_TRANSFORM, 'best_iteration_0based': best,
                  'trees_kept': best + 1, 'catboost_version': catboost.__version__},
            meta_va, price_va.to_numpy(), pred_va, train_metrics, seconds, best + 1,
            extra={'validation_predictions_clipped': n_clipped,
                   'log_clip_bounds': list(bounds), 'iteration_detail': f'{best + 1} trees kept (early stopping)'})
        print(f"  validation MAE {result['validation']['mae']:,.2f} SAR | "
              f"RMSE {result['validation']['rmse']:,.2f} | R2 {result['validation']['r2']:.4f}")


if __name__ == '__main__':
    main()
