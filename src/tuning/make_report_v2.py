"""
Stage: Hyperparameter Tuning - Work in Progress (not part of Initial Modeling).
Reads initial artifacts when needed; writes only under outputs/tuning/.

Write outputs/tuning/reports/v2_r2_improvement_report.md from the saved v2 tables.

Run: python -m src.tuning.make_report_v2
Every number is read from outputs/tuning/tables/*.csv; the text only describes them.
"""

import pandas as pd

from src.modeling import config as cfg
from src.tuning.evaluate_v2 import metrics_both_scales  # noqa: F401  (same metric definitions)
from src.modeling.make_report import md_table


def main():
    cfg.make_tuning_output_dirs()
    exp = pd.read_csv(cfg.TUNING_TABLES_DIR / 'v2_xgboost_validation_experiments.csv')
    val = pd.read_csv(cfg.TUNING_TABLES_DIR / 'v2_validation_comparison.csv')
    test = pd.read_csv(cfg.TUNING_TABLES_DIR / 'v2_test_comparison.csv')
    per_type = pd.read_csv(cfg.TUNING_TABLES_DIR / 'v2_per_property_type_test_metrics.csv').drop(columns='R2 note')
    num = '{:,.0f}'
    text = f"""# Second round (v2): reducing the R² collapse

## Diagnosis (validation 2024 only)
R² on the SAR scale is dominated by a handful of giant Commercial land parcels: about 100 of 247,088
validation rows carry over 90% of the squared error, and the trees over-predict such rows by orders of
magnitude by copying the few huge training prices. On the ln(price) scale the same models score about
0.70-0.72, so relative accuracy is reasonable; the SAR-scale R² is an outlier-driven metric.
Capping predictions at the 99.5th/99.9th training-price percentile was tried and rejected (it hurt,
because real 2024 prices go far above those levels).

## Change 1: minimum rows per leaf (XGBoost only, validation experiment)
{md_table(exp, {'val_mae_sar': num, 'val_rmse_sar': num, 'val_r2_sar': '{:.4f}', 'val_r2_ln_price': '{:.4f}', 'train_seconds': '{:.0f}'})}

`min_child_weight = 200` (v1: 5) gave the best validation MAE and R²; 50 and 1000 were both worse.

## Change 2: equal-weight ensemble
Ensemble v2 = geometric mean (average in ln space) of XGBoost v2 and the frozen v1 CatBoost, weights 0.5/0.5, not tuned.

## Validation 2024
{md_table(val, {c: num for c in val.columns if 'MAE' in c or 'RMSE' in c or 'Median' in c} | {'Validation R2 (SAR)': '{:.4f}', 'Validation R2 (ln price)': '{:.4f}', 'MAE improvement vs baseline (%)': '{:.2f}'})}

## Test 2025 (second look; v1 test results had already been seen)
{md_table(test, {c: num for c in test.columns if 'MAE (SAR)' in c or 'RMSE' in c or 'Median' in c} | {'Test R2 (SAR)': '{:.4f}', 'Test R2 (ln price)': '{:.4f}', 'MAE improvement vs test baseline (%)': '{:.2f}'})}

### By property type (test)
{md_table(per_type, {'Test rows': num, 'MAE (SAR)': num, 'RMSE (SAR)': num, 'R2': '{:.4f}'})}

## Caveats
- All v2 decisions used validation only, but Test 2025 was already inspected for v1, so it is no longer a
  perfectly untouched set. Treat these test numbers as a second look, and prefer a fresh chronological
  split (or 2026 data) for any further decisions.
- Only min_child_weight (3 values) and one fixed 50/50 blend were tried. This is not a full tuning study.
- SAR-scale R² stays low and unstable across years because it depends on a few unpredictable giant parcels.
  Judge models on MAE, median error and ln-scale R² as well.
- The MLP was not revisited.
"""
    (cfg.TUNING_REPORTS_DIR / 'v2_r2_improvement_report.md').write_text(text, encoding='utf-8')
    print('wrote outputs/tuning/reports/v2_r2_improvement_report.md')


if __name__ == '__main__':
    main()
