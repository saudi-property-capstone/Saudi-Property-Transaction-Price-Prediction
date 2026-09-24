"""
Stage: Hyperparameter Tuning - seeded random search for XGBoost, CatBoost, MLP.
Reads initial artifacts when needed; writes only under outputs/tuning/.

Run: python -m src.tuning.tune_models --model xgboost      (or catboost / mlp / all)

Protocol (identical for the three models)
* Fit on Training 2020-2023 using the exported features, whose learned
  preprocessing (frequency encoding, out-of-fold region target encoding,
  scaling) was fitted on Training 2020-2023 only by src/prepare_features.py.
* Early stopping on Validation 2024 (RMSE of ln(price)), exactly as the initial
  models did, so the tree / epoch count is one more hyperparameter chosen on
  validation. Validation scores of every tuned model are therefore optimistic
  in the same way as the initial ones.
* Selection metric: Validation 2024 MAE in SAR, after exp() and clipping to the
  training ln(price) range (the project's inverse transform, see evaluation.py).
* Trial 0 re-runs the initial configuration (XGBoost also re-runs the earlier
  v2 setting as trial 1), so the search is anchored at the known starting point.
* All trial settings are drawn up front from a fixed seed, so an interrupted
  run resumes with exactly the same trials. Finished trials are appended to
  outputs/tuning/tables/tuning_trials_<model>.csv and are never re-run.
* Test 2025 is never read in this file.
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling.evaluation import (count_outside_bounds, log_to_price, regression_metrics,
                                     save_json, save_predictions, stage)
from src.modeling.train_catboost import CATBOOST_CONFIG
from src.modeling.train_mlp import MLP_CONFIG
from src.modeling.train_xgboost import XGB_CONFIG
from src.tuning.evaluate_v2 import metrics_both_scales   # same metric definitions as v2

SEARCH_SEED = cfg.RANDOM_STATE
FAMILIES = ['xgboost', 'catboost', 'mlp']

# ---------- Search spaces ----------
# ('loguniform', low, high) | ('uniform', low, high) | ('choice', [options])
SEARCH_SPACES = {
    'xgboost': {
        'learning_rate': ('loguniform', 0.03, 0.15),
        'max_depth': ('choice', [6, 7, 8, 9, 10, 12]),
        'min_child_weight': ('choice', [1, 5, 20, 50, 100, 200, 400, 800]),
        'subsample': ('uniform', 0.6, 1.0),
        'colsample_bytree': ('uniform', 0.6, 1.0),
        'reg_lambda': ('loguniform', 0.1, 50.0),
        'reg_alpha': ('loguniform', 0.001, 5.0),
        'gamma': ('choice', [0.0, 0.01, 0.05, 0.2]),
    },
    'catboost': {
        'learning_rate': ('loguniform', 0.06, 0.25),
        'depth': ('choice', [6, 7, 8, 9]),
        'l2_leaf_reg': ('loguniform', 1.0, 30.0),
        'random_strength': ('loguniform', 0.25, 5.0),
        'bagging_temperature': ('uniform', 0.0, 1.0),
        'one_hot_max_size': ('choice', [2, 16]),
        'max_ctr_complexity': ('choice', [1, 2, 4]),
    },
    'mlp': {
        'hidden_units': ('choice', [[128, 64, 32], [256, 128, 64], [512, 256, 128],
                                    [256, 256, 128, 64]]),
        'learning_rate': ('loguniform', 3e-4, 3e-3),
        'batch_size': ('choice', [1024, 2048, 4096]),
        'loss': ('choice', ['mse', 'huber']),
        'dropout': ('choice', [0.0, 0.1, 0.2]),
        'l2': ('choice', [0.0, 1e-6, 1e-5]),
        'area_transform': ('choice', ['standardized', 'log1p_standardized']),
    },
}

# Settings held fixed during the search (on top of the initial configs).
FIXED = {
    'xgboost': {'n_estimators': 5000, 'early_stopping_rounds': 50},
    'catboost': {'iterations': 3000, 'early_stopping_rounds': 50},
    'mlp': {'max_epochs': 60, 'early_stopping_patience': 8},
}

# Anchor trials: the initial configuration (and, for XGBoost, the earlier v2 run).
ANCHORS = {
    'xgboost': [('initial config', {}), ('earlier v2 setting', {'min_child_weight': 200})],
    'catboost': [('initial config', {})],
    'mlp': [('initial config', {'loss': 'mse', 'dropout': 0.0, 'l2': 0.0,
                                'area_transform': 'standardized', 'max_epochs': 100})],
}
N_RANDOM_TRIALS = {'xgboost': 18, 'catboost': 9, 'mlp': 13}
HUBER_DELTA = 1.0          # on ln(price): errors beyond ~e^1 = 2.7x are penalised linearly


def trials_csv(family):
    return cfg.TUNING_TABLES_DIR / f'tuning_trials_{family}.csv'


def _round(value):
    return float(f'{value:.4g}') if isinstance(value, float) else value


def sample_params(space, rng):
    params = {}
    for name, (kind, *args) in space.items():
        if kind == 'choice':
            params[name] = args[0][rng.randint(len(args[0]))]
        elif kind == 'loguniform':
            params[name] = _round(float(np.exp(rng.uniform(np.log(args[0]), np.log(args[1])))))
        elif kind == 'uniform':
            params[name] = _round(float(rng.uniform(args[0], args[1])))
        else:
            raise ValueError(kind)
    return params


def planned_trials(family):
    """[(trial, source, overrides)] - fully determined by SEARCH_SEED."""
    rng = np.random.RandomState(SEARCH_SEED + FAMILIES.index(family))
    plan = [(i, src, dict(p)) for i, (src, p) in enumerate(ANCHORS[family])]
    for j in range(N_RANDOM_TRIALS[family]):
        plan.append((len(plan), 'random search', sample_params(SEARCH_SPACES[family], rng)))
    return plan


def full_config(family, overrides, anchor=False):
    """Initial config + search-fixed settings + the trial's overrides.

    Anchor trials keep the initial iteration/epoch ceiling so trial 0 is a true re-run.
    """
    base = {'xgboost': XGB_CONFIG, 'catboost': CATBOOST_CONFIG, 'mlp': MLP_CONFIG}[family]
    fixed = {} if anchor else FIXED[family]
    config = {**base, **fixed, **overrides}
    if family == 'mlp':
        config = {'loss': 'mse', 'dropout': 0.0, 'l2': 0.0, 'area_transform': 'standardized',
                  **config}
    return config


# ---------- MLP inputs: optional log1p(area) replaces the standardized raw area ----------

def area_log_stats(raw_area):
    """Mean/std of log1p(area) on the rows the model is fitted on (training only)."""
    values = np.log1p(np.asarray(raw_area, dtype=np.float64))
    return float(values.mean()), float(values.std())


def mlp_matrix(scaled, raw_area, area_transform, log_stats):
    """Column 0 of the scaled matrix is standardized raw area (numeric__area)."""
    if area_transform == 'standardized':
        return scaled
    if area_transform != 'log1p_standardized':
        raise ValueError(area_transform)
    out = scaled.copy()
    out[:, 0] = ((np.log1p(np.asarray(raw_area, dtype=np.float64)) - log_stats[0])
                 / log_stats[1]).astype(np.float32)
    return out


# ---------- Generic fit / predict (also used by stacking with fixed iterations) ----------

def _xgb_fit(config, X_tr, y_tr, X_es, y_es, fixed_iterations):
    import xgboost as xgb
    config = dict(config)
    if fixed_iterations is not None:
        config.pop('early_stopping_rounds', None)
        config['n_estimators'] = int(fixed_iterations)
        model = xgb.XGBRegressor(**config)
        model.fit(X_tr, y_tr, verbose=False)
        return model, int(fixed_iterations), None
    model = xgb.XGBRegressor(**config)
    model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    return model, int(model.best_iteration) + 1, None


def _cat_fit(config, X_tr, y_tr, X_es, y_es, fixed_iterations):
    from catboost import CatBoostRegressor, Pool

    from src.feature_engineering import CATBOOST_CATEGORICAL
    config = dict(config)
    train_pool = Pool(X_tr, y_tr, cat_features=CATBOOST_CATEGORICAL)
    if fixed_iterations is not None:
        config.pop('early_stopping_rounds', None)
        config.update(iterations=int(fixed_iterations), use_best_model=False)
        model = CatBoostRegressor(**config, verbose=0)
        model.fit(train_pool)
        return model, int(fixed_iterations), None
    model = CatBoostRegressor(**config, verbose=0)
    model.fit(train_pool, eval_set=Pool(X_es, y_es, cat_features=CATBOOST_CATEGORICAL))
    return model, int(model.get_best_iteration()) + 1, None


def build_tuned_mlp(n_features, target_mean, config):
    from tensorflow import keras
    reg = keras.regularizers.l2(config['l2']) if config['l2'] else None
    layers = [keras.Input(shape=(n_features,))]
    for units in config['hidden_units']:
        layers.append(keras.layers.Dense(units, activation=config['activation'],
                                         kernel_regularizer=reg))
        if config['dropout']:
            layers.append(keras.layers.Dropout(config['dropout']))
    layers.append(keras.layers.Dense(
        1, bias_initializer=keras.initializers.Constant(float(target_mean))))
    model = keras.Sequential(layers)
    loss = keras.losses.Huber(delta=HUBER_DELTA) if config['loss'] == 'huber' else 'mse'
    model.compile(optimizer=keras.optimizers.Adam(config['learning_rate']), loss=loss)
    return model


def _mlp_fit(config, X_tr, y_tr, X_es, y_es, fixed_iterations):
    import tensorflow as tf
    from tensorflow import keras
    keras.utils.set_random_seed(config['random_seed'])
    tf.config.experimental.enable_op_determinism()
    model = build_tuned_mlp(X_tr.shape[1], float(np.mean(y_tr)), config)
    if fixed_iterations is not None:
        history = model.fit(X_tr, y_tr, epochs=int(fixed_iterations),
                            batch_size=config['batch_size'], shuffle=True, verbose=0)
        return model, int(fixed_iterations), history.history
    stopper = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=config['early_stopping_patience'],
                                            restore_best_weights=True)
    history = model.fit(X_tr, y_tr, validation_data=(X_es, y_es), epochs=config['max_epochs'],
                        batch_size=config['batch_size'], callbacks=[stopper], shuffle=True,
                        verbose=0)
    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    return model, best_epoch, history.history


def fit_model(family, config, X_tr, y_tr, X_es=None, y_es=None, fixed_iterations=None):
    """Fit one model. Either early stopping on (X_es, y_es) or a fixed iteration count.

    Returns (model, iterations_used, history_or_None, seconds). `iterations_used`
    is trees kept (boosting) or best/fixed epoch count (MLP).
    """
    if (X_es is None) == (fixed_iterations is None):
        raise ValueError('Pass either an early-stopping set or fixed_iterations, not both')
    fit = {'xgboost': _xgb_fit, 'catboost': _cat_fit, 'mlp': _mlp_fit}[family]
    start = time.perf_counter()
    model, n_iter, history = fit(config, X_tr, y_tr, X_es, y_es, fixed_iterations)
    return model, n_iter, history, time.perf_counter() - start


def predict_log(family, model, X, n_iter):
    if family == 'xgboost':
        return model.predict(X, iteration_range=(0, int(n_iter)))
    if family == 'catboost':
        return model.predict(X)          # use_best_model already truncated the trees
    return model.predict(X, batch_size=8192, verbose=0).ravel()


def save_model(family, model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if family == 'mlp':
        model.save(path)
    else:
        model.save_model(str(path))


def load_model(family, path):
    if family == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(path)
        return model
    if family == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(path))
        return model
    from tensorflow import keras
    return keras.models.load_model(path)


# ---------- Data for the search (exported train / validation features) ----------

def load_search_data(family):
    inputs_tr, price_tr = dl.load_inputs('train'), dl.load_price('train')
    inputs_va, price_va = dl.load_inputs('validation'), dl.load_price('validation')
    y_tr, y_va = dl.load_log_target('train'), dl.load_log_target('validation')
    dl.check_alignment('train', inputs_tr, price_tr, y_tr)
    dl.check_alignment('validation', inputs_va, price_va, y_va)
    data = {'price_tr': price_tr, 'price_va': price_va, 'meta_va': dl.load_meta('validation', inputs_va),
            'y_tr': y_tr.to_numpy(np.float32 if family == 'mlp' else np.float64),
            'y_va': y_va.to_numpy(np.float32 if family == 'mlp' else np.float64)}
    if family == 'xgboost':
        data['X_tr'], data['X_va'] = dl.load_matrix('unscaled', 'train'), dl.load_matrix('unscaled', 'validation')
    elif family == 'catboost':
        data['X_tr'], data['X_va'] = dl.load_catboost_frame('train'), dl.load_catboost_frame('validation')
    else:
        if dl.feature_names()['scaled'][0] != 'numeric__area':
            raise ValueError('Expected numeric__area as the first scaled column')
        data['X_tr'], data['X_va'] = dl.load_matrix('scaled', 'train'), dl.load_matrix('scaled', 'validation')
        data['area_tr'] = inputs_tr['area'].to_numpy(np.float64)
        data['area_va'] = inputs_va['area'].to_numpy(np.float64)
        unscaled_area = dl.load_matrix('unscaled', 'train')[:, 0]
        if not np.allclose(unscaled_area, data['area_tr'].astype(np.float32)):
            raise ValueError('Raw area does not match the exported area column')
        data['area_log_stats'] = area_log_stats(data['area_tr'])
    if len(data['X_tr']) != len(data['y_tr']) or len(data['X_va']) != len(data['y_va']):
        raise ValueError('Feature and target row counts differ')
    return data


def trial_inputs(family, config, data):
    if family != 'mlp':
        return data['X_tr'], data['X_va']
    stats = data['area_log_stats']
    return (mlp_matrix(data['X_tr'], data['area_tr'], config['area_transform'], stats),
            mlp_matrix(data['X_va'], data['area_va'], config['area_transform'], stats))


# ---------- Search loop ----------

def _load_done(family, plan):
    path = trials_csv(family)
    if not path.exists():
        return pd.DataFrame()
    done = pd.read_csv(path)
    planned = {t: json.dumps(p, sort_keys=True) for t, _, p in plan}
    for _, row in done.iterrows():
        if planned.get(int(row['trial'])) != row['params_json']:
            raise RuntimeError(f'{path.name}: trial {row["trial"]} differs from the seeded plan; '
                               'the search space changed - move the old file away first')
    return done


def save_best(family, key, trial, source, overrides, config, model, n_iter, seconds, data,
              log_va, pred_va, val_metrics, history):
    """Save the model, validation predictions, config and status of the best trial so far."""
    bounds = dl.train_log_bounds()
    X_tr, _ = trial_inputs(family, config, data)
    pred_tr = log_to_price(predict_log(family, model, X_tr, n_iter), bounds)
    save_model(family, model, cfg.MODEL_FILES[key])
    save_predictions(cfg.predictions_path(key, 'validation'), data['meta_va'],
                     data['price_va'].to_numpy(), pred_va)
    unit = 'epochs (best epoch, weights restored)' if family == 'mlp' else 'trees kept (early stopping)'
    save_json({**config, 'target': cfg.TARGET_TRANSFORM, 'selected_trial': trial,
               'trial_source': source, 'searched_overrides': overrides,
               'iterations_used': n_iter}, cfg.config_path(key))
    if history is not None:
        pd.DataFrame(history).rename_axis('epoch_0based').to_csv(
            cfg.TUNING_TABLES_DIR / f'{key}_training_history.csv')
    save_json({'model_key': key, 'model': cfg.MODEL_LABELS[key], 'version': 'Tuned',
               'target_transform': cfg.TARGET_TRANSFORM, 'selected_trial': trial,
               'trial_source': source, 'validation': val_metrics,
               'train': regression_metrics(data['price_tr'], pred_tr),
               'training_seconds': float(seconds), 'best_iteration': int(n_iter),
               'iteration_detail': f'{n_iter} {unit}',
               'validation_predictions_clipped': count_outside_bounds(log_va, bounds),
               'log_clip_bounds': list(bounds)}, cfg.validation_metrics_path(key))


def run_search(family):
    key = cfg.TUNED_KEYS[family]
    plan = planned_trials(family)
    done = _load_done(family, plan)
    best_mae = done['val_mae_sar'].min() if len(done) else np.inf
    todo = [t for t in plan if len(done) == 0 or t[0] not in set(done['trial'])]
    print(f'[{family}] {len(plan)} planned trials, {len(plan) - len(todo)} already finished')
    if not todo:
        return
    with stage(f'{family} / load train + validation'):
        data = load_search_data(family)
        bounds = dl.train_log_bounds()
    for trial, source, overrides in todo:
        config = full_config(family, overrides, anchor=source != 'random search')
        with stage(f'{family} / trial {trial} ({source})'):
            X_tr, X_va = trial_inputs(family, config, data)
            model, n_iter, history, seconds = fit_model(family, config, X_tr, data['y_tr'],
                                                        X_va, data['y_va'])
            log_va = predict_log(family, model, X_va, n_iter)
            pred_va = log_to_price(log_va, bounds)
            m = metrics_both_scales(data['price_va'], pred_va)
            print(f"  {overrides} -> {n_iter} iters, {seconds:.0f}s, val MAE {m['mae']:,.0f} | "
                  f"RMSE {m['rmse']:,.0f} | R2 {m['r2']:.4f} | R2(ln) {m['r2_ln_price']:.4f}", flush=True)
            if m['mae'] < best_mae:
                save_best(family, key, trial, source, overrides, config, model, n_iter, seconds,
                          data, log_va, pred_va, m, history)
                best_mae = m['mae']
                print(f'  new best for {family}; saved', flush=True)
            row = {'trial': trial, 'source': source,
                   'params_json': json.dumps(overrides, sort_keys=True),
                   **{p: json.dumps(overrides[p]) if isinstance(overrides.get(p), list)
                      else overrides.get(p) for p in SEARCH_SPACES[family]},
                   'iterations_used': n_iter, 'val_mae_sar': m['mae'], 'val_rmse_sar': m['rmse'],
                   'val_r2_sar': m['r2'], 'val_r2_ln_price': m['r2_ln_price'],
                   'val_median_abs_error': m['median_abs_error'], 'train_seconds': seconds,
                   'val_predictions_clipped': count_outside_bounds(log_va, bounds)}
            path = trials_csv(family)
            pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)
            del model


def write_search_design():
    save_json({'method': 'seeded random search (numpy RandomState), settings drawn up front',
               'seed': SEARCH_SEED, 'selection_metric': 'Validation 2024 MAE (SAR)',
               'early_stopping': 'Validation 2024, RMSE on ln(price) (same as initial models)',
               'fit_data': 'Training 2020-2023 exported features (training-fitted preprocessing)',
               'test_2025_used': False,
               'spaces': SEARCH_SPACES, 'fixed_settings': FIXED, 'anchor_trials': ANCHORS,
               'n_random_trials': N_RANDOM_TRIALS, 'mlp_huber_delta': HUBER_DELTA},
              cfg.TUNING_MODELS_DIR / 'search_design.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=[*FAMILIES, 'all'], default='all')
    args = parser.parse_args()
    cfg.make_tuning_output_dirs()
    write_search_design()
    for family in FAMILIES if args.model == 'all' else [args.model]:
        run_search(family)
        done = pd.read_csv(trials_csv(family))
        best = done.loc[done['val_mae_sar'].idxmin()]
        status = json.loads(cfg.validation_metrics_path(cfg.TUNED_KEYS[family]).read_text(encoding='utf-8'))
        if int(status['selected_trial']) != int(best['trial']):
            raise RuntimeError(f'{family}: saved model is trial {status["selected_trial"]} but the '
                               f'best trial in the table is {best["trial"]}')
        print(f"[{family}] best trial {int(best['trial'])}: val MAE {best['val_mae_sar']:,.0f}")


if __name__ == '__main__':
    main()
