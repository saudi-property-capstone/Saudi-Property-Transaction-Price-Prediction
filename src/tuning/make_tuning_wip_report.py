"""
Stage: Hyperparameter Tuning (not part of Initial Modeling).
Reads saved tuning tables only; writes only under outputs/tuning/.

Write outputs/tuning/reports/tuning_work_in_progress.md

Run: python -m src.tuning.make_tuning_wip_report

This documents the tuning work. It trains nothing and predicts nothing: it only
re-arranges numbers already saved under outputs/tuning/ (the Test 2025 numbers
come from the files written once by evaluate_v2.py and evaluate_final_test.py).

Sections 1-2 are the earlier exploration (unchanged). Sections 3-12 come from
tune_models.py, stacking.py, final_comparison.py, evaluate_final_test.py and
feature_importance.py.
"""
import json

import pandas as pd

from src.modeling import config as cfg
from src.modeling.evaluation import load_json
from src.modeling.make_report import md_table
from src.tuning import final_comparison as fc
from src.tuning import stacking as st
from src.tuning import tune_models as tm

NUM = '{:,.0f}'
FAMILY_LABEL = {'xgboost': 'XGBoost', 'catboost': 'CatBoost', 'mlp': 'MLP'}


def _space_rows():
    rows = []
    for family, space in tm.SEARCH_SPACES.items():
        for name, (kind, *args) in space.items():
            rng = (', '.join(json.dumps(a) for a in args[0]) if kind == 'choice'
                   else f'{args[0]:g} – {args[1]:g}')
            rows.append({'Model': FAMILY_LABEL[family], 'Hyperparameter': f'`{name}`',
                         'Distribution': kind, 'Range / options': rng})
    return pd.DataFrame(rows)


def _search_summary():
    rows, best_rows = [], []
    for family, key in cfg.TUNED_KEYS.items():
        trials = pd.read_csv(tm.trials_csv(family))
        status, conf = load_json(cfg.validation_metrics_path(key)), load_json(cfg.config_path(key))
        best = trials.loc[trials['val_mae_sar'].idxmin()]
        initial = trials.loc[trials['trial'] == 0].iloc[0]
        rows.append({'Model': FAMILY_LABEL[family], 'Trials run': len(trials),
                     'Anchor trials': int((trials['source'] != 'random search').sum()),
                     'Random trials': int((trials['source'] == 'random search').sum()),
                     'Total search time (min)': trials['train_seconds'].sum() / 60,
                     'Median time per trial (s)': trials['train_seconds'].median(),
                     'Trial 0 (initial config) Val MAE': initial['val_mae_sar'],
                     'Best trial': int(best['trial']), 'Best trial source': best['source'],
                     'Best Val MAE (SAR)': best['val_mae_sar'],
                     'Iterations kept': status['iteration_detail']})
        for name in tm.SEARCH_SPACES[family]:
            if name in conf:
                best_rows.append({'Model': FAMILY_LABEL[family], 'Hyperparameter': f'`{name}`',
                                  'Best value': json.dumps(conf[name])})
        best_rows.append({'Model': FAMILY_LABEL[family], 'Hyperparameter': 'iterations kept',
                          'Best value': status['iteration_detail']})
    return pd.DataFrame(rows), pd.DataFrame(best_rows)


def _trial_table(family):
    trials = pd.read_csv(tm.trials_csv(family))
    cols = ['trial', 'source', *tm.SEARCH_SPACES[family], 'iterations_used', 'val_mae_sar',
            'val_rmse_sar', 'val_r2_sar', 'val_r2_ln_price', 'train_seconds']
    fmt = {'val_mae_sar': NUM, 'val_rmse_sar': NUM, 'val_r2_sar': '{:.4f}',
           'val_r2_ln_price': '{:.4f}', 'train_seconds': '{:.0f}', 'iterations_used': '{:.0f}'}
    return md_table(trials[cols], fmt)


def _interpretation(v, sel):
    lines = []
    for family, key in cfg.TUNED_KEYS.items():
        t, i = cfg.MODEL_LABELS[key], cfg.MODEL_LABELS[family]
        lines.append(
            f"- **{t}**: Validation MAE {v.loc[t, 'Validation MAE (SAR)']:,.0f} SAR vs "
            f"{v.loc[i, 'Validation MAE (SAR)']:,.0f} for {i} "
            f"({v.loc[t, 'MAE change vs initial (%)']:+.2f}% MAE reduction from tuning); "
            f"{v.loc[t, 'MAE improvement vs baseline (%)']:.2f}% better than the baseline; "
            f"R² (SAR) {v.loc[i, 'Validation R2 (SAR)']:.3f} → {v.loc[t, 'Validation R2 (SAR)']:.3f}, "
            f"R² (ln price) {v.loc[i, 'Validation R2 (ln price)']:.3f} → {v.loc[t, 'Validation R2 (ln price)']:.3f}.")
    diff = sel['stack_minus_best_individual_mae_sar']
    best_single = sel['best_individual_tuned_model']
    if sel['stack_beats_best_individual']:
        lines.append(
            f"- **Stacking helped on validation:** the stack's MAE is {-diff:,.0f} SAR "
            f"({-diff / v.loc[best_single, 'Validation MAE (SAR)'] * 100:.2f}%) lower than {best_single}, "
            "so the stack is the preferred final configuration.")
    else:
        lines.append(
            f"- **Stacking did not help on validation:** the stack's MAE is {diff:,.0f} SAR higher than "
            f"{best_single}, so the best individual model is kept as the final configuration.")
    chosen = sel['selected_model']
    lines.append(
        f"- **Final configuration (chosen on Validation 2024 only): {chosen}**, Validation MAE "
        f"{v.loc[chosen, 'Validation MAE (SAR)']:,.0f} SAR, "
        f"{v.loc[chosen, 'MAE improvement vs baseline (%)']:.2f}% below the baseline "
        f"({v.loc[cfg.MODEL_LABELS['baseline'], 'Validation MAE (SAR)']:,.0f} SAR).")
    lines.append(
        "- SAR-scale R² stays sensitive to a few very large Commercial transactions (see the per-type "
        "table); MAE, the median absolute error and R² on ln(price) are the steadier signals.")
    return '\n'.join(lines)


def _test_text(sel):
    test_path = cfg.TUNING_TABLES_DIR / 'final_test_comparison.csv'
    if not test_path.exists():
        return '**Not run yet.** `python -m src.tuning.evaluate_final_test` has not been executed.'
    test = pd.read_csv(test_path)
    test_type = pd.read_csv(cfg.TUNING_TABLES_DIR / 'final_test_per_property_type.csv')
    fmt = {'Test MAE (SAR)': NUM, 'Test RMSE (SAR)': NUM, 'Test R2 (SAR)': '{:.4f}',
           'Test R2 (ln price)': '{:.4f}', 'Median abs error (SAR)': NUM,
           'MAE improvement vs test baseline (%)': '{:.2f}'}
    type_fmt = {'Test rows': NUM, 'MAE (SAR)': NUM, 'RMSE (SAR)': NUM, 'R2 (SAR)': '{:.4f}'}
    tt = test.set_index('Model').loc[sel['selected_model']]
    return f"""{md_table(test, fmt)}

By property type (diagnostic):

{md_table(test_type.drop(columns='R2 note'), type_fmt)}

On Test 2025 the frozen {sel['selected_model']} reaches MAE {tt['Test MAE (SAR)']:,.0f} SAR,
{tt['MAE improvement vs test baseline (%)']:.2f}% below the historical-median baseline on the same year
(15% target met: {tt['Met 15% target']})."""


def _importance_text():
    path = cfg.TUNING_TABLES_DIR / 'feature_importance_by_characteristic.csv'
    if not path.exists():
        return '**Not run yet.** `python -m src.tuning.feature_importance` has not been executed.'
    g = pd.read_csv(path)
    top = g.iloc[0]
    g.columns = [c.replace('mean |SHAP|', 'mean abs SHAP') for c in g.columns]   # '|' would split the table
    fmt = {'Stack mean abs SHAP': '{:.3f}', 'XGBoost mean abs SHAP': '{:.3f}', 'CatBoost mean abs SHAP': '{:.3f}',
           'Stack share (%)': '{:.1f}', 'Typical price factor (x)': '{:.2f}'}
    fig = '../figures/feature_importance_'
    return f"""Exact SHAP values of both base models (XGBoost `pred_contribs`, CatBoost `ShapValues`) on 5,000 sampled
Validation 2024 transactions, grouped into five characteristics and combined with the stack weights (the stack is
linear in ln(price), so its SHAP values are the weighted sum). A SHAP value s multiplies the predicted price by exp(s).
Code: `src/tuning/feature_importance.py`.

{md_table(g, fmt)}

- **{top['Characteristic']}** dominates ({top['Stack share (%)']:.1f}% of the total influence); city is the strongest
  single feature in both models.
- The price rises with area, but less than proportionally.
- The time trend is lowest in late 2020, rises through 2021 and peaks in 2022; 2024 stays at the late-2023 level
  because trees cannot extend a trend beyond the training period.
- Property type and season add little once location and area are known.

![Characteristics]({fig}characteristics.png)
![By model]({fig}by_model.png)
![Area effect]({fig}area_effect.png)
![Time effect]({fig}time_effect.png)"""


def tuning_and_stacking_sections():
    """Sections 3-12, built only from saved files."""
    search, best_params = _search_summary()
    design = load_json(cfg.TUNING_MODELS_DIR / 'search_design.json')
    ranking = pd.read_csv(st.RANKING_CSV)
    selected = ranking[ranking['Selected for stacking'] == 'Yes']
    dropped = ranking[ranking['Selected for stacking'] == 'No'].iloc[0]
    folds = pd.read_csv(st.FOLD_METRICS_CSV)
    meta = load_json(cfg.MODEL_FILES[cfg.STACK_KEY])
    stack = load_json(cfg.validation_metrics_path(cfg.STACK_KEY))['validation']
    val = pd.read_csv(fc.VAL_TABLE)
    per_type = pd.read_csv(fc.VAL_PER_TYPE)
    fit = pd.read_csv(fc.FIT_TABLE)
    sel = load_json(fc.SELECTION_JSON)
    v = val.set_index('Model')

    val_fmt = {'Validation MAE (SAR)': NUM, 'Validation RMSE (SAR)': NUM,
               'Validation R2 (SAR)': '{:.4f}', 'Validation R2 (ln price)': '{:.4f}',
               'Median abs error (SAR)': NUM, 'MAE improvement vs baseline (%)': '{:.2f}',
               'MAE change vs initial (%)': '{:+.2f}', 'Training time (s)': '{:,.0f}'}
    type_fmt = {'Validation rows': NUM, 'MAE (SAR)': NUM, 'RMSE (SAR)': NUM, 'R2 (SAR)': '{:.4f}'}
    rank_fmt = {'Validation MAE (SAR)': NUM, 'Validation RMSE (SAR)': NUM,
                'Validation R2 (SAR)': '{:.4f}', 'Validation R2 (ln price)': '{:.4f}'}
    fold_fmt = {'fit_rows': NUM, 'held_out_rows': NUM, 'iterations_fixed': '{:.0f}',
                'held_out_mae_sar': NUM, 'held_out_rmse_sar': NUM, 'held_out_r2_sar': '{:.4f}',
                'held_out_r2_ln_price': '{:.4f}', 'fit_seconds': '{:.0f}'}
    fit_fmt = {'Train MAE (SAR)': NUM, 'Validation MAE (SAR)': NUM,
               'Validation/Train MAE ratio': '{:.2f}', 'Train R2 (SAR)': '{:.4f}',
               'Validation R2 (SAR)': '{:.4f}'}
    search_fmt = {'Total search time (min)': '{:.1f}', 'Median time per trial (s)': '{:.0f}',
                  'Trial 0 (initial config) Val MAE': NUM, 'Best Val MAE (SAR)': NUM}
    budgets = ', '.join(f'{FAMILY_LABEL[f]} {n}' for f, n in design['n_random_trials'].items())
    fixed = '; '.join(f'{FAMILY_LABEL[f]} {json.dumps(s)}' for f, s in design['fixed_settings'].items())
    coefs = ', '.join(f'{FAMILY_LABEL[f]} {c:.4f}' for f, c in meta['coefficients'].items())
    candidates = '\n'.join(f'- {m}: {mae:,.0f}' for m, mae in sel['candidates_validation_mae'].items())

    return f"""## 3. Hyperparameter search (XGBoost, CatBoost, MLP)

**Protocol.** {design['method']}; seed `{design['seed']}`. Each trial is fitted on Training 2020–2023
(exported features whose frequency encoding, out-of-fold region target encoding and scaling were fitted on
2020–2023 only), with early stopping on Validation 2024 (RMSE of ln(price), as for the initial models), and is
scored by **Validation 2024 MAE in SAR** after `exp()` of the ln(price) prediction, clipped to the training
ln(price) range. The target is natural `ln(price)` (checked on load: `exp(y_log) == price`); `expm1` is never
used. Predictors are area, year/quarter features and location / property-type encodings only — no
`price_per_m2` or other target-derived column. **Test 2025 was not read by the search.**

Trial 0 re-runs each initial configuration (XGBoost trial 1 re-runs the earlier v2 setting), so the initial
results are reproduced inside the same search and tuning cannot select a configuration that is worse on
validation than the initial one. Budget, chosen for a 4-core / 16 GB laptop CPU: {budgets} random trials on top
of the anchor trials. Fixed during the search: {fixed}. MLP Huber delta {design['mlp_huber_delta']} (on ln(price)).
The MLP option `log1p_standardized` replaces the standardized raw area with standardized `log1p(area)` whose
mean / std come from the training rows.

### 3.1 Search spaces
{md_table(_space_rows())}

### 3.2 Search summary
{md_table(search, search_fmt)}

### 3.3 Best hyperparameters (searched keys; all other settings as in the initial configs)
{md_table(best_params)}

Full configurations: `outputs/tuning/models/<model>_tuned/<model>_tuned_config.json`.

### 3.4 All trials (blank cells in anchor trials = the initial configuration's value)
**XGBoost**

{_trial_table('xgboost')}

**CatBoost**

{_trial_table('catboost')}

**MLP**

{_trial_table('mlp')}

## 4. Model ranking and choice of stacking base models

The tuned models are ranked by Validation 2024 MAE (SAR). Only the two lowest become stacking base models;
no pair was chosen in advance.

{md_table(ranking.drop(columns='family'), rank_fmt)}

**Selected: {' and '.join(selected['Model'])}**, the two lowest validation MAEs.
{dropped['Model']} is left out because its validation MAE ({dropped['Validation MAE (SAR)']:,.0f} SAR) is the highest of the three.

## 5. Stacking (Ridge meta-model on the two selected models)

- **Out-of-fold scheme (forward-chaining inside Training 2020–2023):** fit 2020 → predict 2021; fit 2020–2021 →
  predict 2022; fit 2020–2022 → predict 2023. In each fold all learned preprocessing (location frequencies, region
  target encoding, scaling, log-area statistics) is refitted on the fold's training years only.
- Base models use their tuned settings with a **fixed** tree / epoch count (the count kept by the tuned model), so a
  fold's held-out labels are used neither for fitting nor for early stopping.
- **2020 rows ({meta['rows_excluded_no_oof_2020']:,}) have no earlier data and so no out-of-fold prediction; they are
  excluded**, not filled. The meta-model is fitted on {meta['oof_rows']:,} OOF rows (2021–2023).
- **Meta-model:** Ridge(alpha={meta['alpha']}) on the two ln(price) predictions → ln(price); coefficients {coefs};
  intercept {meta['intercept']:.4f}. No in-sample base prediction and no validation / test label was used to fit it.
- **Validation 2024:** the tuned base models refitted on all of 2020–2023 (reloaded, re-predicted and checked
  against their saved predictions) feed the meta-model.

Base-model accuracy on each held-out training year (fixed iterations, fold-fitted preprocessing):

{md_table(folds.drop(columns=['family']), fold_fmt)}

Stack on Validation 2024: MAE {stack['mae']:,.0f} SAR, RMSE {stack['rmse']:,.0f}, R² (SAR) {stack['r2']:.4f},
R² (ln price) {stack['r2_ln_price']:.4f}.

## 6. Validation 2024 comparison: baseline, initial, tuned, stack

All rows are recomputed from each model's saved Validation 2024 predictions with the same code.
Improvement = (baseline MAE − model MAE) / baseline MAE × 100. "MAE change vs initial" applies the same formula
against the matching initial model. The stack's training time is the time of its out-of-fold fits.

{md_table(val, val_fmt)}

### 6.1 By property type (diagnostic only; every model is trained on all property types together)
{md_table(per_type.drop(columns='R2 note'), type_fmt)}

### 6.2 Train vs validation (tuned models, SAR scale)
{md_table(fit, fit_fmt)}

## 7. Interpretation

{_interpretation(v, sel)}

## 8. Final selection (frozen before any Test 2025 use)

Rule: {sel['rule']}. Candidates' Validation MAE (SAR):
{candidates}

Selected: **{sel['selected_model']}**. Its files were hashed into `outputs/tuning/models/frozen_final_selection.json`,
and the test script refuses to run if any of them changed.

## 9. Test 2025: frozen final model only (reported separately)

All decisions in sections 3–8 used Validation 2024 only. In this stage Test 2025 was used only to evaluate
the frozen selection; no other model was evaluated on it.

{_test_text(sel)}

## 10. What drives the predicted price (feature importance)

{_importance_text()}

## 11. Limitations

- Validation 2024 is used both for early stopping and for choosing hyperparameters and the final model, so every
  tuned validation score (and the initial ones) is optimistic; Test 2025 is the estimate on unseen-year data.
- Random search with a laptop-sized budget; a larger search or another seed may find better settings. One training
  seed per configuration; seed-to-seed variation was not measured.
- Stacking base models in the folds use the tuned models' iteration counts (chosen on 2024) rather than an inner early
  stop, and fold models see less data (one to three years) than the full-training base models used at validation time.
  The Ridge alpha was fixed at {meta['alpha']} (two inputs, hundreds of thousands of rows, so its effect is negligible).
- Trees cannot extrapolate the time index beyond 2023; price growth in 2024–2025 is captured only as level shifts.
- SAR-scale RMSE and R² are dominated by a small number of very expensive Commercial transactions.

## 12. Commands (from the repository root)

```bash
# earlier exploration (sections 1-2)
python -m src.tuning.experiments_v2_xgboost
python -m src.tuning.train_xgboost_v2
python -m src.tuning.evaluate_v2          # performs a Test 2025 evaluation for the v2 models; see section 1
python -m src.tuning.make_report_v2
# search, stacking, selection, final test (sections 3-9)
python -m src.tuning.tune_models --model all      # resumable; finished trials are skipped
python -m src.tuning.stacking                     # resumable; finished folds are cached
python -m src.tuning.final_comparison             # Validation 2024 tables, selection, freeze
python -m src.tuning.evaluate_final_test          # one Test 2025 evaluation of the frozen choice
python -m src.tuning.feature_importance            # SHAP importance of the final model
python -m src.tuning.make_tuning_wip_report
```
"""



def main():
    cfg.make_tuning_output_dirs()
    exp = pd.read_csv(cfg.TUNING_TABLES_DIR / 'v2_xgboost_validation_experiments.csv')
    val = pd.read_csv(cfg.TUNING_TABLES_DIR / 'v2_validation_comparison.csv')
    test = pd.read_csv(cfg.TUNING_TABLES_DIR / 'v2_test_comparison.csv')
    best = exp.loc[exp['val_mae_sar'].idxmin()]
    num = '{:,.0f}'
    val_fmt = {c: num for c in val.columns if 'MAE (SAR)' in c or 'RMSE' in c or 'Median' in c}
    val_fmt.update({'Validation R2 (SAR)': '{:.4f}', 'Validation R2 (ln price)': '{:.4f}',
                    'MAE improvement vs baseline (%)': '{:.2f}'})
    test_fmt = {c: num for c in test.columns if 'MAE (SAR)' in c or 'RMSE' in c or 'Median' in c}
    test_fmt.update({'Test R2 (SAR)': '{:.4f}', 'Test R2 (ln price)': '{:.4f}',
                     'MAE improvement vs test baseline (%)': '{:.2f}'})
    exp_fmt = {'min_child_weight': '{:.0f}', 'trees_kept': '{:.0f}', 'val_mae_sar': num,
               'val_rmse_sar': num, 'val_r2_sar': '{:.4f}', 'val_r2_ln_price': '{:.4f}',
               'train_seconds': '{:.0f}'}

    text = f"""# Hyperparameter Tuning

> **Status:** sections 1–2 are the earlier exploration and are kept as they were. Sections 3–9 add the
> hyperparameter search for XGBoost, CatBoost and MLP, the two-model stack, the Validation 2024 comparison,
> the final selection, and a single Test 2025 evaluation of the frozen selection. Section 10 shows which
> characteristics drive the predicted price.
> The Initial Modeling stage (baseline + Initial XGBoost / CatBoost / MLP) is reported separately in
> `outputs/initial_models/reports/initial_models_report.md`.

Generated by `src/tuning/make_tuning_wip_report.py` from saved tuning tables only.

## 1. Test-set methodology note (important)

- **Test 2025 was evaluated for the initial models.**
- **It was later evaluated again for the v2 experiment** (XGBoost v2 and the 50/50 ensemble).
- Therefore the v2 Test 2025 results are a **second look** and are **not equivalent to a completely untouched final holdout**.
- **No future hyperparameter or model decision may use Test 2025.** All future tuning must use
  **Training 2020–2023 and Validation 2024 only**.
- Existing Test 2025 results are preserved unchanged and were not recalculated when this report was produced.

## 2. Earlier exploration (before the full search)

### 2.1 XGBoost `min_child_weight` trials (validation only)
Initial XGBoost used `min_child_weight = 5`. The values 50, 200 and 1000 were then evaluated on Validation 2024
(everything else unchanged, early stopping on Validation 2024):

{md_table(exp, exp_fmt)}

Among these three trials, `min_child_weight = {best['min_child_weight']:.0f}` gave the strongest validation result
(lowest Validation MAE, {best['val_mae_sar']:,.0f} SAR). This is only a three-value trial, not a search of the parameter space.

### 2.2 XGBoost v2 — a tuning-stage model
XGBoost v2 is Initial XGBoost with `min_child_weight = 200`. It is a **tuning-stage experiment, not an initial model**,
and it has **not** been selected as the outcome of tuning. Its files are under `outputs/tuning/models/xgboost_v2/`.

### 2.3 50/50 XGBoost–CatBoost ensemble — a tuning-stage experiment
The ensemble averages XGBoost v2 and Initial CatBoost in `ln(price)` space with fixed equal weights (0.5 / 0.5).
**The ensemble weights were not tuned**: 50/50 was fixed in advance. During exploration a single 70/30 alternative was
glanced at on validation; it was not adopted, and no weight search was run. The ensemble is not an initial model.

### 2.4 Prediction-capping experiments (exploration only)
Capping predicted prices at high training-price percentiles was tried on Validation 2024 during the diagnosis of the
SAR-scale R² problem. Caps at lower percentiles made results worse; only a very loose cap looked useful, and none was adopted.
**These numbers were only inspected interactively and were not saved to a file**, so they are not reproduced here.
The experiment must be redone in a saved, reproducible script if it is pursued.

### 2.5 Validation 2024 comparison (saved)
Rows labelled v1 are the initial models, shown for reference.

{md_table(val, val_fmt)}

### 2.6 Test 2025 comparison (saved; second look, see section 1)

{md_table(test, test_fmt)}

Per-property-type results for this experiment: `outputs/tuning/tables/v2_per_property_type_test_metrics.csv`.
The earlier diagnosis and interpretation are in `outputs/tuning/reports/v2_r2_improvement_report.md`.

{tuning_and_stacking_sections()}"""
    path = cfg.TUNING_REPORTS_DIR / 'tuning_work_in_progress.md'
    path.write_text(text, encoding='utf-8')
    print(f'wrote {path.relative_to(cfg.ROOT).as_posix()}')


if __name__ == '__main__':
    main()
