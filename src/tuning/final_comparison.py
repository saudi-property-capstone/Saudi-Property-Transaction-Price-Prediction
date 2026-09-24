"""
Stage: Hyperparameter Tuning - Validation 2024 comparison, final selection, freeze.
Reads saved predictions / status files only; writes only under outputs/tuning/.

Run: python -m src.tuning.final_comparison      (after tune_models.py and stacking.py)

* One Validation 2024 table: historical-median baseline, the three initial models,
  the three tuned models and the two-model stack, all recomputed from their saved
  per-row prediction files with the same metric code (provenance-checked against
  the MAE stored in each model's status file).
* Per-property-type validation metrics (diagnostic only; training stays combined).
* Final choice = lowest Validation 2024 MAE among the tuned models and the stack.
  The stack is NOT preferred by default: if an individual tuned model has the lower
  validation MAE, it is kept. Test 2025 is not read.
* The chosen files are frozen (SHA-256) so evaluate_final_test.py can check that
  nothing changed between the choice and the single test evaluation.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.modeling import config as cfg
from src.modeling.evaluation import (improvement_pct, load_json, per_property_type_metrics,
                                     save_json, sha256_of_file, stage)
from src.tuning import tune_models as tm

INITIAL = ['xgboost', 'catboost', 'mlp']
TUNED = [cfg.TUNED_KEYS[f] for f in INITIAL]
CANDIDATES = [*TUNED, cfg.STACK_KEY]
STAGES = {'baseline': 'Baseline', **{k: 'Initial' for k in INITIAL},
          **{k: 'Tuned' for k in TUNED}, cfg.STACK_KEY: 'Tuned stack'}
VAL_TABLE = cfg.TUNING_TABLES_DIR / 'final_validation_comparison.csv'
VAL_PER_TYPE = cfg.TUNING_TABLES_DIR / 'final_validation_per_property_type.csv'
FIT_TABLE = cfg.TUNING_TABLES_DIR / 'tuned_overfitting_check.csv'
SELECTION_JSON = cfg.TUNING_TABLES_DIR / 'final_selection.json'


def status_path(key):
    return cfg.STATUS_DIR / 'baseline_validation_metrics.json' if key == 'baseline' \
        else cfg.validation_metrics_path(key)


def saved_mae(key):
    s = load_json(status_path(key))
    return s['metrics']['mae'] if key == 'baseline' else s['validation']['mae']


def validation_frame(key):
    frame = pd.read_csv(cfg.predictions_path(key, 'validation'), index_col='row_id')
    mae = np.abs(frame['actual_price'] - frame['predicted_price']).mean()
    if not np.isclose(mae, saved_mae(key), rtol=1e-6):          # provenance guard
        raise ValueError(f'{key}: saved predictions do not match the saved validation MAE')
    return frame


def build_tables():
    keys = ['baseline', *INITIAL, *TUNED, cfg.STACK_KEY]
    frames = {k: validation_frame(k) for k in keys}
    ref = frames['baseline']
    for k, f in frames.items():
        if not (f.index.equals(ref.index) and np.allclose(f['actual_price'], ref['actual_price'])):
            raise ValueError(f'{k}: validation rows differ from the baseline rows')
    base_mae = saved_mae('baseline')
    rows, per_type = [], []
    for k in keys:
        f = frames[k]
        m = tm.metrics_both_scales(f['actual_price'], f['predicted_price'])
        status = load_json(status_path(k))
        imp = improvement_pct(base_mae, m['mae'])
        row = {'Model': cfg.MODEL_LABELS[k], 'Stage': STAGES[k],
               'Validation MAE (SAR)': m['mae'], 'Validation RMSE (SAR)': m['rmse'],
               'Validation R2 (SAR)': m['r2'], 'Validation R2 (ln price)': m['r2_ln_price'],
               'Median abs error (SAR)': m['median_abs_error'],
               'MAE improvement vs baseline (%)': imp,
               'Met 15% target': 'n/a (reference)' if k == 'baseline'
               else ('Yes' if imp >= cfg.TARGET_IMPROVEMENT_PCT else 'No'),
               'MAE change vs initial (%)': np.nan,
               'Training time (s)': status.get('fit_seconds', status.get('training_seconds')),
               'Iteration detail': status.get('iteration_detail', 'not applicable (rule-based)')}
        if k in TUNED:
            init_mae = saved_mae(INITIAL[TUNED.index(k)])
            row['MAE change vs initial (%)'] = improvement_pct(init_mae, m['mae'])
        rows.append(row)
        per_type += per_property_type_metrics(f, cfg.MODEL_LABELS[k])
    table = pd.DataFrame(rows)
    per_type = pd.DataFrame(per_type).rename(columns={
        'model': 'Model', 'property_type': 'Property type', 'n_rows': 'Validation rows',
        'mae': 'MAE (SAR)', 'rmse': 'RMSE (SAR)', 'r2': 'R2 (SAR)', 'r2_note': 'R2 note'})
    fit = pd.DataFrame([{
        'Model': cfg.MODEL_LABELS[k],
        'Train MAE (SAR)': load_json(status_path(k))['train']['mae'],
        'Validation MAE (SAR)': saved_mae(k),
        'Validation/Train MAE ratio': saved_mae(k) / load_json(status_path(k))['train']['mae'],
        'Train R2 (SAR)': load_json(status_path(k))['train']['r2'],
        'Validation R2 (SAR)': load_json(status_path(k))['validation']['r2']} for k in TUNED])
    return table, per_type, fit


def files_to_freeze(key):
    if key == cfg.STACK_KEY:
        base = [cfg.TUNED_KEYS[f] for f in load_json(cfg.MODEL_FILES[key])['base_families']]
        return [cfg.MODEL_FILES[key]] + [p for b in base for p in (cfg.MODEL_FILES[b], cfg.config_path(b))]
    return [cfg.MODEL_FILES[key], cfg.config_path(key)]


def select_and_freeze(table):
    cand = table[table['Model'].isin([cfg.MODEL_LABELS[k] for k in CANDIDATES])]
    cand = cand.sort_values('Validation MAE (SAR)')
    label_to_key = {cfg.MODEL_LABELS[k]: k for k in CANDIDATES}
    chosen = label_to_key[cand.iloc[0]['Model']]
    stack_mae = float(table.loc[table['Model'] == cfg.MODEL_LABELS[cfg.STACK_KEY], 'Validation MAE (SAR)'].iloc[0])
    best_single = cand[cand['Model'] != cfg.MODEL_LABELS[cfg.STACK_KEY]].iloc[0]
    selection = {
        'selected_key': chosen, 'selected_model': cfg.MODEL_LABELS[chosen],
        'rule': 'lowest Validation 2024 MAE (SAR) among the tuned models and the two-model stack; '
                'Test 2025 not used',
        'candidates_validation_mae': dict(zip(cand['Model'], cand['Validation MAE (SAR)'])),
        'best_individual_tuned_model': best_single['Model'],
        'stack_beats_best_individual': bool(stack_mae < best_single['Validation MAE (SAR)']),
        'stack_minus_best_individual_mae_sar': stack_mae - float(best_single['Validation MAE (SAR)']),
    }
    save_json(selection, SELECTION_JSON)
    if cfg.FROZEN_FINAL_JSON.exists():
        frozen = load_json(cfg.FROZEN_FINAL_JSON)
        if frozen['selected_key'] != chosen:
            raise RuntimeError('A different model was frozen earlier; refusing to overwrite the freeze')
        print('[freeze] already frozen with the same selection -> keeping the original record')
        return selection
    save_json({'frozen_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'selected_key': chosen, 'selected_model': cfg.MODEL_LABELS[chosen],
               'files': {p.relative_to(cfg.ROOT).as_posix(): sha256_of_file(p)
                         for p in files_to_freeze(chosen)}}, cfg.FROZEN_FINAL_JSON)
    return selection


def main():
    cfg.make_tuning_output_dirs()
    with stage('final comparison / Validation 2024 tables'):
        table, per_type, fit = build_tables()
        table.to_csv(VAL_TABLE, index=False)
        per_type.to_csv(VAL_PER_TYPE, index=False)
        fit.to_csv(FIT_TABLE, index=False)
        pd.set_option('display.width', 260, 'display.max_columns', 30,
                      'display.float_format', lambda x: f'{x:,.4f}')
        print(table.drop(columns='Iteration detail').to_string(index=False))
        print(fit.to_string(index=False))
    with stage('final comparison / select + freeze'):
        print(select_and_freeze(table))


if __name__ == '__main__':
    main()
