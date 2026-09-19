"""
Stage: Hyperparameter Tuning - Work in Progress (not part of Initial Modeling).
Reads initial artifacts when needed; writes only under outputs/tuning/.

Validation-only experiment: does requiring more rows per leaf reduce the
extreme over-predictions that wreck R² (in SAR)?

Run: python -m src.tuning.experiments_v2_xgboost

* Uses Train 2020-2023 to fit and Validation 2024 for early stopping and
  comparison. Test 2025 is NOT read anywhere in this file.
* Deliberately small: three values of min_child_weight (minimum rows per leaf
  for squared-error loss), everything else identical to the frozen v1 config.
* The frozen v1 models and results are not modified. Outputs are new files.
"""

import numpy as np
import pandas as pd

from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling import train_xgboost as tx
from src.modeling.evaluation import log_to_price, regression_metrics, stage

MIN_CHILD_WEIGHTS = [50, 200, 1000]          # v1 used 5
OUT_CSV = cfg.TUNING_TABLES_DIR / 'v2_xgboost_validation_experiments.csv'


def main():
    cfg.make_tuning_output_dirs()
    with stage('v2 experiments / load train + validation'):
        y_tr, y_va = dl.load_log_target('train'), dl.load_log_target('validation')
        price_va = dl.load_price('validation')
        X_tr, X_va = dl.load_matrix('unscaled', 'train'), dl.load_matrix('unscaled', 'validation')
        bounds = dl.train_log_bounds()
    rows = []
    for mcw in MIN_CHILD_WEIGHTS:
        with stage(f'v2 experiments / XGBoost min_child_weight={mcw}'):
            model, config, seconds = tx.fit_xgboost(
                X_tr, y_tr.to_numpy(), X_va, y_va.to_numpy(),
                overrides={'min_child_weight': mcw}, verbose=False)
            best = int(model.best_iteration)
            log_pred = tx.predict_log(model, X_va, best)
            pred = log_to_price(log_pred, bounds)
            m = regression_metrics(price_va, pred)
            m_log = regression_metrics(y_va, np.clip(log_pred, *bounds))
            rows.append({'min_child_weight': mcw, 'trees_kept': best + 1,
                         'val_mae_sar': m['mae'], 'val_rmse_sar': m['rmse'],
                         'val_r2_sar': m['r2'], 'val_r2_ln_price': m_log['r2'],
                         'train_seconds': seconds})
            np.save(cfg.TUNING_TABLES_DIR / f'.v2_xgb_mcw{mcw}_val_logpred.npy', log_pred)
            print(f"  mcw={mcw}: MAE {m['mae']:,.0f} | R2 {m['r2']:.4f} | R2(ln) {m_log['r2']:.4f}")
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
