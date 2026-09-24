"""
Stage: Hyperparameter Tuning - two-model stacking (Ridge meta-model).
Reads tuned-model artifacts; writes only under outputs/tuning/.

Run: python -m src.tuning.stacking          (needs tune_models.py finished for all three)

1. Rank Tuned XGBoost / CatBoost / MLP by Validation 2024 MAE (SAR). Only the
   TWO lowest become base models; nothing is pre-selected.
2. Out-of-fold (OOF) predictions inside Training 2020-2023, forward-chaining by year:
       fit 2020            -> predict 2021
       fit 2020-2021       -> predict 2022
       fit 2020-2022       -> predict 2023
   In every fold ALL learned preprocessing (location frequencies, region target
   encoding, scaling, log-area statistics) is refitted on the fold's training
   years only. Base models use their tuned settings with a FIXED tree / epoch
   count (the count chosen for the tuned model), so the held-out year's labels
   are used neither for fitting nor for early stopping. 2020 rows have no
   earlier data, hence no OOF prediction, and are excluded from the meta-model.
3. The meta-model is Ridge(alpha=1.0) on the two OOF ln(price) predictions,
   target ln(price) of the same 2021-2023 rows. No in-sample base predictions
   and no validation / test labels are used to fit it.
4. Validation 2024: the base models refitted on the full Training 2020-2023 are
   the saved tuned models; they are reloaded and re-predict 2024 (checked
   against their saved predictions), then the meta-model combines them.
Test 2025 is never read in this file.
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.feature_engineering import FeaturePreprocessor, catboost_features
from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling.evaluation import (count_outside_bounds, load_json, log_to_price,
                                     save_json, save_predictions, stage)
from src.tuning import tune_models as tm

KEY = cfg.STACK_KEY
META_ALPHA = 1.0
FOLDS = [([2020], 2021), ([2020, 2021], 2022), ([2020, 2021, 2022], 2023)]
PARTS_DIR = cfg.TUNING_PREDICTIONS_DIR / 'stack_oof_parts'
RANKING_CSV = cfg.TUNING_TABLES_DIR / 'tuned_model_ranking.csv'
FOLD_METRICS_CSV = cfg.TUNING_TABLES_DIR / 'stack_oof_fold_metrics.csv'
OOF_CSV = cfg.TUNING_PREDICTIONS_DIR / 'stack_oof_train_predictions.csv.gz'


def rank_tuned_models():
    rows = []
    for family, key in cfg.TUNED_KEYS.items():
        r = load_json(cfg.validation_metrics_path(key))
        rows.append({'Model': cfg.MODEL_LABELS[key], 'family': family,
                     'Validation MAE (SAR)': r['validation']['mae'],
                     'Validation RMSE (SAR)': r['validation']['rmse'],
                     'Validation R2 (SAR)': r['validation']['r2'],
                     'Validation R2 (ln price)': r['validation']['r2_ln_price']})
    table = pd.DataFrame(rows).sort_values('Validation MAE (SAR)').reset_index(drop=True)
    table.insert(1, 'Rank (Validation MAE)', np.arange(1, len(table) + 1))
    table['Selected for stacking'] = np.where(table['Rank (Validation MAE)'] <= 2, 'Yes', 'No')
    return table


def fold_features(family, config, raw_fit, y_fit, raw_out):
    """Refit all learned preprocessing on the fold's training years only."""
    if family == 'catboost':                     # no learned state in this feature path
        return (dl._clean_catboost_frame(catboost_features(raw_fit)),
                dl._clean_catboost_frame(catboost_features(raw_out)))
    pre = FeaturePreprocessor(scale_numeric=(family == 'mlp'))
    X_fit = pre.fit_transform(raw_fit, y_fit).astype(np.float32)
    X_out = pre.transform(raw_out).astype(np.float32)
    if family == 'mlp':
        stats = tm.area_log_stats(raw_fit['area'])
        X_fit = tm.mlp_matrix(X_fit, raw_fit['area'], config['area_transform'], stats)
        X_out = tm.mlp_matrix(X_out, raw_out['area'], config['area_transform'], stats)
    return X_fit, X_out


def oof_predictions(family, raw, y_log, price):
    """ln(price) OOF predictions for 2021-2023 (one cached file per fold)."""
    key = cfg.TUNED_KEYS[family]
    config = load_json(cfg.config_path(key))
    n_iter = int(load_json(cfg.validation_metrics_path(key))['best_iteration'])
    fit_config = {k: v for k, v in config.items()
                  if k not in ('target', 'selected_trial', 'trial_source', 'searched_overrides',
                               'iterations_used')}
    parts = []
    for fit_years, out_year in FOLDS:
        part_path = PARTS_DIR / f'{family}_{out_year}.csv.gz'
        if part_path.exists():
            print(f'  [{family} -> {out_year}] cached')
            parts.append(pd.read_csv(part_path, index_col='row_id'))
            continue
        fit_mask, out_mask = raw['year'].isin(fit_years), raw['year'].eq(out_year)
        with stage(f'stacking / {family} fit {fit_years} -> predict {out_year}'):
            X_fit, X_out = fold_features(family, fit_config, raw[fit_mask], y_log[fit_mask],
                                         raw[out_mask])
            y_fit = y_log[fit_mask].to_numpy(np.float32 if family == 'mlp' else np.float64)
            model, _, _, seconds = tm.fit_model(family, fit_config, X_fit, y_fit,
                                                fixed_iterations=n_iter)
            bounds = (float(y_fit.min()), float(y_fit.max()))       # fold-training range
            log_out = np.clip(tm.predict_log(family, model, X_out, n_iter), *bounds)
            m = tm.metrics_both_scales(price[out_mask], np.exp(log_out))
            print(f"  held-out {out_year}: MAE {m['mae']:,.0f} | R2(ln) {m['r2_ln_price']:.4f} "
                  f"| {seconds:.0f}s", flush=True)
        part = pd.DataFrame({'log_pred': log_out}, index=raw.index[out_mask])
        PARTS_DIR.mkdir(parents=True, exist_ok=True)
        part.to_csv(part_path, compression={'method': 'gzip'})
        row = {'family': family, 'model': cfg.MODEL_LABELS[key],
               'fit_years': '-'.join(map(str, fit_years)), 'held_out_year': out_year,
               'fit_rows': int(fit_mask.sum()), 'held_out_rows': int(out_mask.sum()),
               'iterations_fixed': n_iter, 'held_out_mae_sar': m['mae'],
               'held_out_rmse_sar': m['rmse'], 'held_out_r2_sar': m['r2'],
               'held_out_r2_ln_price': m['r2_ln_price'], 'fit_seconds': seconds}
        pd.DataFrame([row]).to_csv(FOLD_METRICS_CSV, mode='a', index=False,
                                   header=not FOLD_METRICS_CSV.exists())
        parts.append(part)
        del model
    return pd.concat(parts)['log_pred']


def validation_log_predictions(family):
    """Reload the tuned model (fitted on all of 2020-2023) and predict 2024 in ln(price)."""
    key = cfg.TUNED_KEYS[family]
    config = load_json(cfg.config_path(key))
    n_iter = int(load_json(cfg.validation_metrics_path(key))['best_iteration'])
    data = tm.load_search_data(family)
    _, X_va = tm.trial_inputs(family, config, data)
    log_va = tm.predict_log(family, tm.load_model(family, cfg.MODEL_FILES[key]), X_va, n_iter)
    saved = pd.read_csv(cfg.predictions_path(key, 'validation'), index_col='row_id')
    if not np.allclose(log_to_price(log_va, dl.train_log_bounds()), saved['predicted_price'],
                       rtol=1e-5):
        raise RuntimeError(f'{family}: reloaded model does not reproduce its saved validation predictions')
    return pd.Series(log_va, index=saved.index)


def main():
    cfg.make_tuning_output_dirs()
    with stage('stacking / rank tuned models by Validation MAE'):
        ranking = rank_tuned_models()
        ranking.to_csv(RANKING_CSV, index=False)
        print(ranking.to_string(index=False))
        selected = ranking.loc[ranking['Selected for stacking'] == 'Yes', 'family'].tolist()
        print(f'  base models: {selected}')

    with stage('stacking / load Training 2020-2023'):
        raw, price = dl.load_inputs('train'), dl.load_price('train')
        y_log = dl.load_log_target('train')
        dl.check_alignment('train', raw, price, y_log)

    oof = pd.DataFrame({f: oof_predictions(f, raw, y_log, price) for f in selected})
    if oof.isna().any().any() or not raw.loc[oof.index, 'year'].between(2021, 2023).all():
        raise RuntimeError('OOF predictions are incomplete or cover the wrong years')
    excluded = int(raw['year'].eq(2020).sum())
    oof_out = oof.assign(year=raw.loc[oof.index, 'year'], log_price=y_log.loc[oof.index])
    oof_out.to_csv(OOF_CSV, compression={'method': 'gzip'})

    with stage('stacking / fit Ridge meta-model on OOF 2021-2023'):
        meta = Ridge(alpha=META_ALPHA).fit(oof[selected].to_numpy(), y_log.loc[oof.index].to_numpy())
        oof_fit = meta.predict(oof[selected].to_numpy())
        oof_metrics = tm.metrics_both_scales(price.loc[oof.index], log_to_price(oof_fit, dl.train_log_bounds()))
        coefs = dict(zip(selected, meta.coef_.tolist()))
        print(f'  coefficients {coefs}, intercept {meta.intercept_:.4f}; '
              f'OOF rows {len(oof):,} (2020 rows excluded: {excluded:,})')

    with stage('stacking / Validation 2024'):
        log_va = pd.DataFrame({f: validation_log_predictions(f) for f in selected})
        bounds = dl.train_log_bounds()
        stack_log = meta.predict(log_va[selected].to_numpy())
        inputs_va, price_va = dl.load_inputs('validation'), dl.load_price('validation')
        if not log_va.index.equals(price_va.index):
            raise RuntimeError('Validation predictions are not aligned with the labels')
        pred_va = log_to_price(stack_log, bounds)
        m = tm.metrics_both_scales(price_va, pred_va)
        save_predictions(cfg.predictions_path(KEY, 'validation'), dl.load_meta('validation', inputs_va),
                         price_va.to_numpy(), pred_va)
        save_json({'meta_model': 'sklearn Ridge', 'alpha': META_ALPHA,
                   'features': [f'ln(price) prediction of {cfg.MODEL_LABELS[cfg.TUNED_KEYS[f]]}'
                                for f in selected],
                   'base_families': selected, 'coefficients': coefs,
                   'intercept': float(meta.intercept_), 'target': cfg.TARGET_TRANSFORM,
                   'oof_scheme': [{'fit_years': f, 'held_out_year': h} for f, h in FOLDS],
                   'oof_rows': int(len(oof)), 'rows_excluded_no_oof_2020': excluded,
                   'base_model_iterations_in_folds': 'fixed at the tuned model\'s count (no early stopping)'},
                  cfg.MODEL_FILES[KEY])
        save_json({'model_key': KEY, 'model': cfg.MODEL_LABELS[KEY], 'version': 'Tuned (stack)',
                   'base_models': [cfg.MODEL_LABELS[cfg.TUNED_KEYS[f]] for f in selected],
                   'validation': m, 'meta_fit_on_oof_2021_2023': oof_metrics,
                   'coefficients': coefs, 'intercept': float(meta.intercept_),
                   'training_seconds': float(pd.read_csv(FOLD_METRICS_CSV)['fit_seconds'].sum()),
                   'best_iteration': None,
                   'iteration_detail': 'Ridge on ' + ' + '.join(selected) + ' (OOF 2021-2023)',
                   'validation_predictions_clipped': count_outside_bounds(stack_log, bounds)},
                  cfg.validation_metrics_path(KEY))
        print(f"  stack validation MAE {m['mae']:,.0f} | RMSE {m['rmse']:,.0f} | R2 {m['r2']:.4f} "
              f"| R2(ln) {m['r2_ln_price']:.4f}")


if __name__ == '__main__':
    main()
