"""
Train the initial MLP (no tuning) and evaluate it on Validation 2024.

Run:   python -m src.modeling.train_mlp            (skips if already finished)
       python -m src.modeling.train_mlp --force    (retrain from scratch)

Input : SCALED engineered matrices (X_*_scaled.npy). The scaler was fitted on
        2020-2023 by prepare_features.py and saved in preprocessor_scaled.json;
        validation/test are transformed with those same training statistics.
Uses  : 2020-2023 to fit, 2024 for early stopping and evaluation. 2025 is never read.
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')   # quieter TensorFlow logs

import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling.evaluation import (count_outside_bounds, log_to_price, regression_metrics,
                                     save_validation_results, stage, validation_is_complete)

KEY = 'mlp'
HISTORY_CSV = cfg.TABLES_DIR / 'mlp_training_history.csv'

# Initial (untuned) configuration: a small network, not a deep one.
MLP_CONFIG = {
    'hidden_units': [128, 64, 32],     # ReLU hidden layers
    'activation': 'relu',
    'loss': 'mse',                     # squared error on ln(price)
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'batch_size': 2048,                # ~460 updates per epoch on 945k rows
    'max_epochs': 100,
    'early_stopping_patience': 8,
    'early_stopping_monitor': 'val_loss',
    'restore_best_weights': True,
    'random_seed': cfg.RANDOM_STATE,
}


def build_mlp(n_features, target_mean, config):
    """Dense ReLU network. The output bias starts at the TRAINING mean of ln(price)
    so the network does not waste epochs learning the target's overall level."""
    layers = [keras.Input(shape=(n_features,))]
    layers += [keras.layers.Dense(units, activation=config['activation'])
               for units in config['hidden_units']]
    layers.append(keras.layers.Dense(
        1, bias_initializer=keras.initializers.Constant(float(target_mean))))
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(config['learning_rate']),
                  loss=config['loss'], metrics=['mae'])
    return model


def fit_mlp(X_train, y_train, X_val, y_val, overrides=None, verbose=2):
    """Fit with EarlyStopping(restore_best_weights=True) on validation loss.

    Returns (model, history dict, config, seconds).
    """
    config = {**MLP_CONFIG, **(overrides or {})}
    keras.utils.set_random_seed(config['random_seed'])      # Python, NumPy, TF seeds
    tf.config.experimental.enable_op_determinism()
    model = build_mlp(X_train.shape[1], float(np.mean(y_train)), config)
    stopper = keras.callbacks.EarlyStopping(
        monitor=config['early_stopping_monitor'],
        patience=config['early_stopping_patience'],
        restore_best_weights=config['restore_best_weights'])
    start = time.perf_counter()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=config['max_epochs'], batch_size=config['batch_size'],
                        callbacks=[stopper], shuffle=True, verbose=verbose)
    return model, history.history, config, time.perf_counter() - start


def predict_log(model, X):
    return model.predict(X, batch_size=8192, verbose=0).ravel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='retrain even if finished')
    args = parser.parse_args()
    cfg.make_output_dirs()
    if validation_is_complete(KEY) and not args.force:
        print('[MLP] already finished -> skipping (use --force to retrain)')
        return

    with stage('MLP / load data'):
        inputs_tr, price_tr = dl.load_inputs('train'), dl.load_price('train')
        inputs_va, price_va = dl.load_inputs('validation'), dl.load_price('validation')
        y_tr, y_va = dl.load_log_target('train'), dl.load_log_target('validation')
        dl.check_alignment('train', inputs_tr, price_tr, y_tr)
        dl.check_alignment('validation', inputs_va, price_va, y_va)
        X_tr, X_va = dl.load_matrix('scaled', 'train'), dl.load_matrix('scaled', 'validation')
        if len(X_tr) != len(y_tr) or len(X_va) != len(y_va):
            raise ValueError('Feature matrix and target row counts differ')
        if not (np.isfinite(X_tr).all() and np.isfinite(X_va).all()):
            raise ValueError('Scaled matrices contain NaN or infinite values')
        names = dl.feature_names()['scaled']
        meta_va = dl.load_meta('validation', inputs_va)
        print(f'  train {X_tr.shape}, validation {X_va.shape}, features: {names}')

    with stage('MLP / fit (early stopping on 2024)'):
        model, history, config, seconds = fit_mlp(X_tr, y_tr.to_numpy(np.float32),
                                                  X_va, y_va.to_numpy(np.float32))
        epochs_completed = len(history['loss'])
        best_epoch = int(np.argmin(history['val_loss'])) + 1
        print(f'  epochs completed: {epochs_completed}, best epoch: {best_epoch}, '
              f'training time {seconds:.1f}s')

    with stage('MLP / evaluate on train and validation'):
        bounds = dl.train_log_bounds()
        log_tr, log_va = predict_log(model, X_tr), predict_log(model, X_va)
        pred_tr, pred_va = log_to_price(log_tr, bounds), log_to_price(log_va, bounds)
        n_clipped = count_outside_bounds(log_va, bounds)
        print(f"  validation predictions clipped to the training range: {n_clipped}")
        train_metrics = regression_metrics(price_tr, pred_tr)
        cfg.MODEL_FILES[KEY].parent.mkdir(parents=True, exist_ok=True)
        model.save(cfg.MODEL_FILES[KEY])
        history_frame = pd.DataFrame(history)
        history_frame.insert(0, 'epoch', np.arange(1, len(history_frame) + 1))
        history_frame.to_csv(HISTORY_CSV, index=False)

    with stage('MLP / save results'):
        result = save_validation_results(
            KEY, {**config, 'input_features': names, 'target': cfg.TARGET_TRANSFORM,
                  'epochs_completed': epochs_completed, 'best_epoch': best_epoch,
                  'n_parameters': int(model.count_params()),
                  'tensorflow_version': tf.__version__, 'keras_version': keras.__version__,
                  'architecture': [f"Dense({u}, relu)" for u in config['hidden_units']]
                                  + ['Dense(1, linear)']},
            meta_va, price_va.to_numpy(), pred_va, train_metrics, seconds, epochs_completed,
            extra={'validation_predictions_clipped': n_clipped, 'log_clip_bounds': list(bounds),
                   'epochs_completed': epochs_completed, 'best_epoch': best_epoch,
                   'iteration_detail': f'{epochs_completed} epochs completed '
                                       f'(best epoch {best_epoch}, weights restored)'})
        print(f"  validation MAE {result['validation']['mae']:,.2f} SAR | "
              f"RMSE {result['validation']['rmse']:,.2f} | R2 {result['validation']['r2']:.4f}")


if __name__ == '__main__':
    main()
