"""
Build the Initial validation comparison table and FREEZE the three models.

Run: python -m src.modeling.compare_validation

Needs the three training scripts to have finished. Ranks the models by
Validation 2024 MAE (primary), compares against the historical-median
baseline, then records the SHA-256 of every model and config file in
outputs/initial_models/models/frozen_configs.json. evaluate_test.py refuses to run unless those files
still match, so nothing can change after the freeze.
"""
from datetime import datetime, timezone

import pandas as pd

from src.modeling import baseline as baseline_module
from src.modeling import config as cfg
from src.modeling import initial_tables
from src.modeling.evaluation import (load_json, save_json,
                                     sha256_of_file, stage, validation_is_complete)


def load_baseline_validation():
    path = cfg.STATUS_DIR / 'baseline_validation_metrics.json'
    if not path.exists():
        baseline_module.main()
    return load_json(path)


def build_validation_tables():
    """Validation comparison (from saved metrics) plus the train-vs-validation check."""
    missing = [k for k in cfg.MODEL_KEYS if not validation_is_complete(k)]
    if missing:
        raise RuntimeError(f'Models not trained yet: {missing}. Run their train_*.py scripts '
                           '(finished models are skipped automatically).')
    load_baseline_validation()        # makes sure the baseline status file exists
    table = initial_tables.build_comparison('validation')
    fit_rows = []
    for key in cfg.MODEL_KEYS:
        r = load_json(cfg.validation_metrics_path(key))
        fit_rows.append({
            'Model': cfg.MODEL_LABELS[key],
            'Train MAE (SAR)': r['train']['mae'], 'Validation MAE (SAR)': r['validation']['mae'],
            'Validation/Train MAE ratio': r['validation']['mae'] / r['train']['mae'],
            'Train R2': r['train']['r2'], 'Validation R2': r['validation']['r2'],
        })
    return table, pd.DataFrame(fit_rows), table.iloc[0]['Validation MAE (SAR)']


def freeze_configs():
    """Record hashes once. If already frozen, verify instead of overwriting."""
    if cfg.FROZEN_CONFIG_JSON.exists():
        print('[freeze] already frozen -> keeping the original freeze record')
        return
    frozen = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'note': 'Initial untuned configurations. Not to be changed after this point.',
              'models': {}}
    for key in cfg.MODEL_KEYS:
        frozen['models'][key] = {
            'model_file': cfg.MODEL_FILES[key].relative_to(cfg.ROOT).as_posix(),
            'model_sha256': sha256_of_file(cfg.MODEL_FILES[key]),
            'config_file': cfg.config_path(key).relative_to(cfg.ROOT).as_posix(),
            'config_sha256': sha256_of_file(cfg.config_path(key)),
            'validation_mae': load_json(cfg.validation_metrics_path(key))['validation']['mae'],
        }
    save_json(frozen, cfg.FROZEN_CONFIG_JSON)


def main():
    cfg.make_output_dirs()
    with stage('validation comparison / build table'):
        table, fit_table, base_mae = build_validation_tables()
        table.to_csv(cfg.TABLES_DIR / 'initial_validation_comparison.csv', index=False)
        fit_table.to_csv(cfg.TABLES_DIR / 'initial_overfitting_check.csv', index=False)
        pd.set_option('display.width', 250, 'display.max_columns', 20,
                      'display.float_format', lambda x: f'{x:,.4f}')
        print(table.to_string(index=False))
        print(fit_table.to_string(index=False))
    with stage('validation comparison / freeze configurations'):
        freeze_configs()


if __name__ == '__main__':
    main()
