"""
Stage: Hyperparameter Tuning - which characteristics drive the predicted price?
Reads the frozen final models; writes only under outputs/tuning/ (tables/, figures/).

Run: python -m src.tuning.feature_importance

* SHAP values are computed natively (XGBoost `pred_contribs`, CatBoost
  `ShapValues`), exact for tree models; no extra library is needed.
* Both models predict ln(price), and the final stack is
  intercept + w_xgb * XGBoost + w_cat * CatBoost in ln(price). SHAP values are
  additive, so the stack's SHAP value for a feature is exactly
  w_xgb * SHAP_xgb + w_cat * SHAP_cat.
* The two models use different encodings of the same raw inputs, so features are
  grouped into five characteristics: Area, Location, Time trend, Season
  (quarter) and Property type.
* Importance = mean |SHAP| on a fixed random sample of 5,000 Validation 2024
  rows (seed 42). One SHAP unit is one unit of ln(price): a value s multiplies the
  predicted price by exp(s). The time-trend figure also uses a sample of
  Training 2020-2023 rows, because Validation 2024 covers a single year.
* Test 2025 is not read.
"""
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import Pool

from src.feature_engineering import CATBOOST_CATEGORICAL
from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling.evaluation import load_json, stage
from src.tuning import tune_models as tm

N_VAL, N_TRAIN = 5_000, 3_000       # exact SHAP on the depth-12 XGBoost is ~35 s per 1,000 rows
FEATURE_TABLE = cfg.TUNING_TABLES_DIR / 'feature_importance_by_model.csv'
GROUP_TABLE = cfg.TUNING_TABLES_DIR / 'feature_importance_by_characteristic.csv'
GROUPS = ['Location', 'Area', 'Property type', 'Time trend', 'Season (quarter)']
XGB_GROUP = {'numeric__area': 'Area', 'numeric__time_index': 'Time trend',
             'numeric__quarter_sin': 'Season (quarter)', 'numeric__quarter_cos': 'Season (quarter)',
             'numeric__city_frequency': 'Location', 'numeric__district_frequency': 'Location',
             'region_target__region': 'Location'}
CAT_GROUP = {'area': 'Area', 'time_index': 'Time trend', 'quarter_sin': 'Season (quarter)',
             'quarter_cos': 'Season (quarter)', 'region': 'Location', 'city': 'Location',
             'city_district': 'Location', 'property_type': 'Property type'}
READABLE = {'numeric__area': 'area', 'numeric__time_index': 'time index (quarter since 2020)',
            'numeric__quarter_sin': 'quarter (sin)', 'numeric__quarter_cos': 'quarter (cos)',
            'numeric__city_frequency': 'city frequency', 'numeric__district_frequency': 'district frequency',
            'region_target__region': 'region (target-encoded)', 'time_index': 'time index (quarter since 2020)',
            'quarter_sin': 'quarter (sin)', 'quarter_cos': 'quarter (cos)', 'city_district': 'district'}

# Reference palette (dataviz skill, light mode): blue = slot 1, orange = slot 2.
BLUE, ORANGE = '#2a78d6', '#eb6834'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def xgb_group(name):
    return XGB_GROUP.get(name, 'Property type' if name.startswith('category__property_type_') else None)


def sample_rows(n_rows, size, seed):
    return np.sort(np.random.RandomState(seed).choice(n_rows, size=min(size, n_rows), replace=False))


def shap_xgboost(X, n_iter):
    model = tm.load_model('xgboost', cfg.MODEL_FILES[cfg.TUNED_KEYS['xgboost']])
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True,
                                          iteration_range=(0, int(n_iter)))
    return contrib[:, :-1], contrib[:, -1]            # features, bias


def shap_catboost(frame):
    model = tm.load_model('catboost', cfg.MODEL_FILES[cfg.TUNED_KEYS['catboost']])
    values = model.get_feature_importance(Pool(frame, cat_features=CATBOOST_CATEGORICAL),
                                          type='ShapValues')
    return values[:, :-1], values[:, -1]


def grouped(values, names, mapper):
    out = pd.DataFrame(0.0, index=range(len(values)), columns=GROUPS)
    for j, name in enumerate(names):
        out[mapper(name)] += values[:, j]
    return out


def explain(split, rows):
    """SHAP per feature (both models) and per characteristic (both models + stack)."""
    x_names = dl.feature_names()['unscaled']
    X = dl.load_matrix('unscaled', split)[rows]
    F = dl.load_catboost_frame(split).iloc[rows]
    n_x = int(load_json(cfg.validation_metrics_path(cfg.TUNED_KEYS['xgboost']))['best_iteration'])
    sx, bx = shap_xgboost(X, n_x)
    sc, bc = shap_catboost(F)
    # Additivity check: SHAP values + bias reproduce each model's prediction.
    pred_x = tm.predict_log('xgboost', tm.load_model('xgboost', cfg.MODEL_FILES[cfg.TUNED_KEYS['xgboost']]), X, n_x)
    if not np.allclose(sx.sum(1) + bx, pred_x, atol=1e-3):
        raise RuntimeError('XGBoost SHAP values do not add up to its predictions')
    pred_c = tm.predict_log('catboost', tm.load_model('catboost', cfg.MODEL_FILES[cfg.TUNED_KEYS['catboost']]), F, None)
    if not np.allclose(sc.sum(1) + bc, pred_c, atol=1e-3):
        raise RuntimeError('CatBoost SHAP values do not add up to its predictions')
    meta = load_json(cfg.MODEL_FILES[cfg.STACK_KEY])
    w = meta['coefficients']
    gx, gc = grouped(sx, x_names, xgb_group), grouped(sc, list(F.columns), CAT_GROUP.get)
    stack = w['xgboost'] * gx + w['catboost'] * gc
    return {'sx': sx, 'sc': sc, 'x_names': x_names, 'c_names': list(F.columns),
            'gx': gx, 'gc': gc, 'stack': stack, 'weights': w, 'X': X, 'F': F}


def importance_tables(r):
    rows = []
    for model, values, names, mapper in [('Tuned XGBoost', r['sx'], r['x_names'], xgb_group),
                                         ('Tuned CatBoost', r['sc'], r['c_names'], CAT_GROUP.get)]:
        imp = np.abs(values).mean(0)
        for name, v in zip(names, imp):
            rows.append({'Model': model, 'Feature': READABLE.get(name, name.replace('category__', '')),
                         'Characteristic': mapper(name), 'Mean |SHAP| (ln price)': v,
                         'Share of model total (%)': v / imp.sum() * 100})
    features = pd.DataFrame(rows).sort_values(['Model', 'Mean |SHAP| (ln price)'], ascending=[False, False])
    g = pd.DataFrame({'Characteristic': GROUPS,
                      'Stack mean |SHAP|': r['stack'].abs().mean().values,
                      'XGBoost mean |SHAP|': r['gx'].abs().mean().values,
                      'CatBoost mean |SHAP|': r['gc'].abs().mean().values})
    g['Stack share (%)'] = g['Stack mean |SHAP|'] / g['Stack mean |SHAP|'].sum() * 100
    g['Typical price factor (x)'] = np.exp(g['Stack mean |SHAP|'])
    g = g.sort_values('Stack mean |SHAP|', ascending=False).reset_index(drop=True)
    g.insert(0, 'Rank', np.arange(1, len(g) + 1))
    return features, g


def _style(ax, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_title(title, loc='left', fontsize=11.5, color=INK, fontweight='bold', pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2, va='bottom')


def _save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    path = cfg.TUNING_FIGURES_DIR / name
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_characteristics(g):
    g = g.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh(g['Characteristic'], g['Stack share (%)'], color=BLUE, height=0.55)
    for y, share in enumerate(g['Stack share (%)']):
        ax.text(share + 0.8, y, f'{share:.1f}%', va='center', fontsize=9, color=INK)
    ax.set_xlim(0, g['Stack share (%)'].max() * 1.15)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Share of total mean |SHAP| (%)', color=INK2, fontsize=9)
    ax.tick_params(axis='y', labelsize=10, labelcolor=INK)
    _style(ax, 'What drives the predicted price (final stacked model)',
           f'Mean |SHAP| on {N_VAL:,} sampled Validation 2024 transactions')
    return _save(fig, 'feature_importance_characteristics.png')


def fig_models(features):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (model, color) in zip(axes, [('Tuned XGBoost', BLUE), ('Tuned CatBoost', ORANGE)]):
        part = features[features['Model'] == model].sort_values('Share of model total (%)')
        ax.barh(part['Feature'], part['Share of model total (%)'], color=color, height=0.6)
        for y, v in enumerate(part['Share of model total (%)']):
            ax.text(v + 0.6, y, f'{v:.1f}%', va='center', fontsize=8, color=INK)
        ax.set_xlim(0, part['Share of model total (%)'].max() * 1.2)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('Share of model total mean |SHAP| (%)', color=INK2, fontsize=8.5)
        ax.tick_params(axis='y', labelcolor=INK)
        _style(ax, model)
    return _save(fig, 'feature_importance_by_model.png')


def fig_area(r):
    area = r['F']['area'].to_numpy(float)
    effect = np.exp(r['stack']['Area'].to_numpy())
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.scatter(area, effect, s=4, color=BLUE, alpha=0.18, linewidths=0)
    bins = np.logspace(np.log10(np.percentile(area, 0.5)), np.log10(np.percentile(area, 99.5)), 30)
    idx = np.digitize(area, bins)
    mids = [np.sqrt(bins[i - 1] * bins[i]) for i in range(1, len(bins)) if (idx == i).sum() >= 30]
    meds = [np.median(effect[idx == i]) for i in range(1, len(bins)) if (idx == i).sum() >= 30]
    ax.plot(mids, meds, color=INK, lw=2, label='median effect')
    ax.axhline(1, color=INK2, lw=0.8, ls='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(max(effect.min(), 0.05), min(effect.max(), 20))
    ax.set_xlabel('Area (m², log scale)', color=INK2, fontsize=9)
    ax.set_ylabel('Effect on predicted price (x, log scale)', color=INK2, fontsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    _style(ax, 'How area changes the predicted price',
           'Each dot is one Validation 2024 transaction; 1.0 = no effect relative to the average prediction')
    return _save(fig, 'feature_importance_area_effect.png')


def fig_time(r_train, r_val):
    frames = []
    for r in (r_train, r_val):
        frames.append(pd.DataFrame({'time_index': r['F']['time_index'].to_numpy(),
                                    'effect': np.exp(r['stack']['Time trend'].to_numpy())}))
    t = pd.concat(frames).groupby('time_index')['effect'].median()
    labels = [f"{2020 + int(i) // 4} Q{int(i) % 4 + 1}" for i in t.index]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.plot(range(len(t)), t.values, color=BLUE, lw=2, marker='o', ms=5)
    ax.axhline(1, color=INK2, lw=0.8, ls='--')
    ax.axvline(15.5, color=GRID, lw=1.2)
    ax.text(15.6, ax.get_ylim()[1], ' Validation 2024', color=INK2, fontsize=8, va='top')
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Median effect on predicted price (x)', color=INK2, fontsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    _style(ax, 'How the transaction quarter changes the predicted price (time trend)',
           f'{N_TRAIN:,} sampled Training 2020-2023 + {N_VAL:,} Validation 2024 transactions; 1.0 = no effect')
    return _save(fig, 'feature_importance_time_effect.png')


def main():
    cfg.make_tuning_output_dirs()
    with stage('importance / SHAP on Validation 2024 sample'):
        r_val = explain('validation', sample_rows(len(dl.load_log_target('validation')), N_VAL, cfg.RANDOM_STATE))
    with stage('importance / SHAP on Training 2020-2023 sample (time trend only)'):
        r_train = explain('train', sample_rows(len(dl.load_log_target('train')), N_TRAIN, cfg.RANDOM_STATE + 1))
    with stage('importance / tables + figures'):
        features, g = importance_tables(r_val)
        features.to_csv(FEATURE_TABLE, index=False)
        g.to_csv(GROUP_TABLE, index=False)
        paths = [fig_characteristics(g), fig_models(features), fig_area(r_val), fig_time(r_train, r_val)]
        pd.set_option('display.width', 220, 'display.float_format', lambda x: f'{x:,.4f}')
        print(g.to_string(index=False))
        print(features.to_string(index=False))
        print('stack weights', json.dumps(r_val['weights']))
        for p in paths:
            print('wrote', p.relative_to(cfg.ROOT).as_posix())


if __name__ == '__main__':
    main()
