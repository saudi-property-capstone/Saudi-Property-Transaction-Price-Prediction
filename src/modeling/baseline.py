"""
Historical-median baseline (rule-based, no machine learning).

Same rule as notebooks/baseline_model.ipynb: predict the TRAINING-only (2020-2023)
median price of properties sharing the same city_district and property_type,
falling back to the training property_type median and then the overall
training median. This module reimplements that rule so it can also be applied
to 2025, and checks that it reproduces the notebook's saved validation MAE.
"""
import time

import numpy as np
import pandas as pd

from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling.evaluation import (per_property_type_metrics,
                                     regression_metrics, save_json, save_predictions,
                                     stage)

LOCATION_COL = 'city_district'
TYPE_COL = 'property_type'


def fit_baseline(train_inputs, train_price):
    """Learn the three median lookups from training rows only."""
    frame = train_inputs[[LOCATION_COL, TYPE_COL]].assign(price=train_price.to_numpy())
    return {
        'group_median': frame.groupby([LOCATION_COL, TYPE_COL])['price'].median(),
        'type_median': frame.groupby(TYPE_COL)['price'].median(),
        'global_median': float(frame['price'].median()),
    }


def predict_baseline(state, inputs):
    """Apply the saved medians (no learning) with the two-level fallback."""
    keys = pd.MultiIndex.from_frame(inputs[[LOCATION_COL, TYPE_COL]])
    pred = state['group_median'].reindex(keys).to_numpy(dtype=float)
    fallback_type = inputs[TYPE_COL].map(state['type_median']).to_numpy(dtype=float)
    used_group = ~np.isnan(pred)
    pred = np.where(used_group, pred, fallback_type)
    used_type = ~used_group & ~np.isnan(fallback_type)
    pred = np.where(np.isnan(pred), state['global_median'], pred)
    usage = {'group_median_rows': int(used_group.sum()),
             'property_type_fallback_rows': int(used_type.sum()),
             'global_fallback_rows': int((~used_group & ~used_type).sum())}
    return pred, usage


def run_baseline(split):
    """Evaluate the baseline on 'validation' or 'test' (fit on training only)."""
    with stage(f'baseline / load train + {split}'):
        train_inputs, train_price = dl.load_inputs('train'), dl.load_price('train')
        inputs, price = dl.load_inputs(split), dl.load_price(split)
        dl.check_alignment('train', train_inputs, train_price)
        dl.check_alignment(split, inputs, price)
    with stage('baseline / fit medians on training'):
        start = time.perf_counter()
        state = fit_baseline(train_inputs, train_price)
        seconds = time.perf_counter() - start
    with stage(f'baseline / predict {split}'):
        pred, usage = predict_baseline(state, inputs)
        meta = dl.load_meta(split, inputs)
        table = save_predictions(cfg.predictions_path('baseline', split), meta,
                                 price.to_numpy(), pred)
        metrics = regression_metrics(price, pred)
        per_type = per_property_type_metrics(table, cfg.MODEL_LABELS['baseline'])
    return {'metrics': metrics, 'per_type': per_type, 'fallback_usage': usage,
            'fit_seconds': seconds, 'n_groups': int(len(state['group_median']))}


def verify_against_notebook(result):
    """The validation MAE must match the notebook's saved value (sanity check)."""
    summary = pd.read_csv(cfg.BASELINE_SUMMARY_CSV)
    row = summary[(summary['Model'] == 'Baseline - Historical Median')
                  & (summary['Dataset'] == 'Validation 2024')].iloc[0]
    if abs(row['MAE'] - result['metrics']['mae']) > 0.05:
        raise ValueError(
            f"Baseline validation MAE {result['metrics']['mae']:.2f} does not match the "
            f"notebook value {row['MAE']:.2f}; do not compare models until resolved")
    return float(row['MAE'])


def main():
    cfg.make_output_dirs()
    result = run_baseline('validation')
    notebook_mae = verify_against_notebook(result)
    result['notebook_validation_mae'] = notebook_mae
    save_json(result, cfg.STATUS_DIR / 'baseline_validation_metrics.json')
    print(f"Baseline validation MAE {result['metrics']['mae']:,.2f} SAR "
          f"(matches notebook {notebook_mae:,.2f})")


if __name__ == '__main__':
    main()
