"""
Write outputs/initial_models/reports/initial_models_report.md.

Run: python -m src.modeling.make_report

Reads only saved tables/status files (no model is loaded, nothing is predicted).
Every number comes from those files; the sentences below are built from them.
The report covers ONLY the Initial Modeling stage: the historical-median
baseline, Initial XGBoost, Initial CatBoost and Initial MLP.
"""
import pandas as pd

from src.modeling import config as cfg
from src.modeling.evaluation import load_json, stage
from src.modeling.initial_tables import display_name


def md_table(frame, formats=None):
    """Render a DataFrame as a Markdown table with per-column number formats."""
    formats = formats or {}
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '|' + '---|' * len(frame.columns)]
    for _, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            v = row[col]
            if pd.isna(v):
                cells.append('—')
            elif col in formats:
                cells.append(formats[col].format(v))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def config_block(key):
    c = load_json(cfg.config_path(key))
    skip = {'feature_names', 'input_features'}
    return '\n'.join(f'- `{k}`: {v}' for k, v in c.items() if k not in skip)


def main():
    cfg.make_output_dirs()
    with stage('report / read saved results'):
        val = pd.read_csv(cfg.TABLES_DIR / 'initial_validation_comparison.csv')
        test = pd.read_csv(cfg.TABLES_DIR / 'initial_test_comparison.csv')
        per_type = pd.read_csv(cfg.TABLES_DIR / 'initial_per_property_type_test_metrics.csv')
        fit = pd.read_csv(cfg.TABLES_DIR / 'initial_overfitting_check.csv')
        fit['Model'] = fit['Model'].map(display_name)
        summary = load_json(cfg.TABLES_DIR / 'model_selection_summary.json')
        smoke = load_json(cfg.SMOKE_DIR / 'smoke_test_report.json')

    v, t = val.set_index('Model'), test.set_index('Model')
    models = [cfg.MODEL_LABELS[k] for k in cfg.MODEL_KEYS]
    base_name = cfg.MODEL_LABELS['baseline']
    met_val = [m for m in models if v.loc[m, 'Met 15% target'] == 'Yes']
    met_test = [m for m in models if t.loc[m, 'Met 15% target'] == 'Yes']
    drift = {m: (t.loc[m, 'Test MAE (SAR)'] / v.loc[m, 'Validation MAE (SAR)'] - 1) * 100 for m in models}
    best_test = min(models, key=lambda m: t.loc[m, 'Test MAE (SAR)'])
    best_val = min(models, key=lambda m: v.loc[m, 'Validation MAE (SAR)'])
    cat, xgb_, mlp = 'Initial CatBoost', 'Initial XGBoost', 'Initial MLP'

    lines = []
    # --- Statements generated from the saved numbers ---
    if best_test == cat and abs(drift[cat]) == min(abs(d) for d in drift.values()):
        lines.append(
            f"- **{cat} is the strongest stable individual initial model:** it has the lowest Test MAE "
            f"({t.loc[cat, 'Test MAE (SAR)']:,.0f} SAR) and the smallest Validation-to-Test MAE change "
            f"({drift[cat]:+.1f}%), versus {drift[xgb_]:+.1f}% for {xgb_}. {xgb_} had the lowest "
            f"Validation MAE ({v.loc[xgb_, 'Validation MAE (SAR)']:,.0f} SAR) but did not hold up on Test 2025.")
    else:
        lines.append(f"- Best by Validation MAE: {best_val}; best by Test MAE: {best_test}. "
                     f"Validation-to-Test MAE change: " + ', '.join(f'{m} {d:+.1f}%' for m, d in drift.items()) + '.')
    lines.append(
        f"- **{mlp} performed worse than the baseline** in its current configuration: Validation MAE "
        f"{v.loc[mlp, 'Validation MAE (SAR)']:,.0f} vs {v.loc[base_name, 'Validation MAE (SAR)']:,.0f} SAR "
        f"({v.loc[mlp, 'MAE improvement vs baseline (%)']:.1f}%), Test MAE {t.loc[mlp, 'Test MAE (SAR)']:,.0f} vs "
        f"{t.loc[base_name, 'Test MAE (SAR)']:,.0f} SAR ({t.loc[mlp, 'MAE improvement vs baseline (%)']:.1f}%). "
        f"This result is reported as it is; it is not removed or hidden.")
    com = per_type[per_type['Property type'] == 'Commercial'].set_index('Model')
    lines.append(
        "- **SAR-scale R² is strongly influenced by a small number of extreme Commercial transactions.** "
        f"For example the test R² on Commercial rows is {com.loc[xgb_, 'R2']:.2f} for {xgb_} and "
        f"{com.loc[cat, 'R2']:.2f} for {cat}, while Residential rows (the vast majority) are stable. "
        "No new outlier audit, filtering, capping or other data modification was performed in this stage.")
    findings = '\n'.join(lines)

    fmt_val = {'Validation MAE (SAR)': '{:,.0f}', 'Validation RMSE (SAR)': '{:,.0f}',
               'Validation R2 (SAR)': '{:.4f}', 'Validation R2 (ln price)': '{:.4f}',
               'Median abs error (SAR)': '{:,.0f}', 'Training time (s)': '{:,.1f}',
               'Best iteration / epochs': '{:.0f}', 'MAE improvement vs baseline (%)': '{:.2f}',
               'Rank (Validation MAE)': '{:.0f}'}
    fmt_test = {**{k.replace('Validation', 'Test'): f for k, f in fmt_val.items() if k != 'Rank (Validation MAE)'}}
    fmt_type = {'Test rows': '{:,.0f}', 'MAE (SAR)': '{:,.0f}', 'RMSE (SAR)': '{:,.0f}', 'R2': '{:.4f}'}
    fmt_fit = {c: '{:,.0f}' for c in ('Train MAE (SAR)', 'Validation MAE (SAR)')}
    fmt_fit.update({'Validation/Train MAE ratio': '{:.2f}', 'Train R2': '{:.4f}', 'Validation R2': '{:.4f}'})
    notes = per_type.loc[per_type['R2 note'].notna(), ['Model', 'Property type', 'R2 note']]
    drop = ['Iteration detail']
    rec = ' and '.join(display_name(m) for m in summary['recommended_for_tuning'])

    text = f"""# Initial Modeling Report

*Stage: **Initial Modeling** — Historical Median Baseline, Initial XGBoost, Initial CatBoost, Initial MLP.
Generated by `src/modeling/make_report.py` from saved result files only.*

> **Scope.** This report covers only the initial models, trained before any intentional parameter experiments.
> Later experiments belong to a separate stage, **Hyperparameter Tuning — Work in Progress**
> (`outputs/tuning/reports/tuning_work_in_progress.md`). Nothing from that stage appears in the tables below.

## 1. Methodology

- **Data and split** (unchanged from `src/prepare_features.py`): chronological. Train 2020–2023, Validation 2024, Test 2025.
- **Target**: {cfg.TARGET_TRANSFORM}. Models were trained on natural `ln(price)` (the project's existing target) and predictions were converted back with `exp()`, **not** `expm1` (the target is `ln`, not `log1p`). Predictions are clipped to the training `ln(price)` range and at zero.
- **Metrics**: **MAE (SAR) is the primary project metric.** **R² on the original SAR scale is the main R² result.** R² on `ln(price)` is a supporting diagnostic only. Empty cells mean the metric was never computed for that model; nothing was estimated or invented.
- **Leakage controls**: predictors are area, time index, quarter sine/cosine, location and property type. Price, `price_per_m2_*`, `reference_number` and `row_id` are not features. All learned preprocessing (location frequencies, region target encoding, scaling) was fitted on 2020–2023 only and reused unchanged for 2024 and 2025.
- **Feature inputs**: XGBoost — unscaled engineered matrix; MLP — scaled matrix; CatBoost — native categorical strings plus numeric time/area features (no frequency/target/one-hot encodings).
- **Baseline**: historical median price by `city_district` + `property_type` from training rows, with property-type and global fallbacks (rule-based; identical to `notebooks/baseline_model.ipynb`).
- **Early stopping** used Validation 2024, so validation scores are slightly optimistic. **Test 2025** was evaluated once for the initial models, after they were frozen (SHA-256 in `outputs/initial_models/models/frozen_configs.json`). It was not used for training, early stopping, preprocessing or any decision at this stage.
- **Smoke test** (diagnostic only): status **{smoke['status']}**, random_state {smoke['random_state']}.
- **Success criterion**: MAE at least {cfg.TARGET_IMPROVEMENT_PCT:.0f}% lower than the baseline: `(baseline MAE − model MAE) / baseline MAE × 100`.

## 2. Initial configurations (no hyperparameter tuning)

### Initial XGBoost
{config_block('xgboost')}

### Initial CatBoost
{config_block('catboost')}

### Initial MLP
{config_block('mlp')}

## 3. Validation results (2024)

{md_table(val.drop(columns=drop), fmt_val)}

Models meeting the 15% target on validation: **{', '.join(met_val) or 'none'}**.

Train vs validation (SAR scale):

{md_table(fit, fmt_fit)}

## 4. Test results (2025; initial models, evaluated once)

{md_table(test.drop(columns=drop), fmt_test)}

Models meeting the 15% target on test: **{', '.join(met_test) or 'none'}**.

- Best by Validation MAE: **{best_val}**; best by Test MAE: **{best_test}**.
- Models recommended for future hyperparameter tuning (chosen by Validation MAE only): **{rec}**.

## 5. Test results by property type

{md_table(per_type.drop(columns='R2 note'), fmt_type)}

{('R² caveats:' + chr(10) + chr(10) + md_table(notes)) if len(notes) else 'R² was defined and based on at least 30 rows for every subgroup.'}

## 6. Findings

{findings}

## 7. Limitations

- These are **initial, untuned** results. No hyperparameter search was part of this stage and none of these settings is claimed to be optimal.
- Models optimise squared error on `ln(price)` but are judged on MAE in SAR. Prices are heavy-tailed, so MAE and RMSE are dominated by a few very expensive transactions and SAR-scale R² is low or unstable.
- `time_index` in 2024–2025 lies beyond the training range. Tree models cannot extrapolate it; the MLP can, but may do so unreliably.
- Region target encodings are out-of-fold for training and full-training for validation/test (as designed in `feature_engineering.py`); this small train/serve difference is inherited.
- CatBoost train-set errors come from predicting rows it was fitted on, so its train–validation gap is indicative only.
- Unseen locations in 2024–2025 get zero frequency (XGBoost/MLP) or an unseen category (CatBoost).
- MLP training uses a fixed seed and op-determinism, but bit-for-bit reproducibility can differ across machines and library versions (versions are stored in each model's config JSON).
- The Initial MLP's clipped predictions and weak results show it is a poor configuration for this data as it stands; it is not a verdict on neural networks.

## 8. Next stage

**Hyperparameter Tuning — Work in Progress.** It uses Training 2020–2023 and Validation 2024 only. Test 2025 was evaluated for the initial models and again in a later experiment, so it must not guide any further model decision.
"""
    with stage('report / write markdown'):
        path = cfg.REPORTS_DIR / 'initial_models_report.md'
        path.write_text(text, encoding='utf-8')
        print(f'  wrote {path.relative_to(cfg.ROOT).as_posix()}')


if __name__ == '__main__':
    main()
