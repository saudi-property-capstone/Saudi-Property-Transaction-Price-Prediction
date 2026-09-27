"""
Demo: single-transaction price estimate with the frozen final model.

Used by demo/app.py (Streamlit). Nothing is trained here: it loads the frozen
Ridge stack (Tuned XGBoost + Tuned CatBoost), the saved training-fitted
preprocessor, and training-only statistics.

    python -m src.demo.predictor      self-test: reproduces saved Validation 2024
                                      predictions of the final model

* Features are built exactly as for evaluation: the saved Training 2020-2023
  preprocessor for XGBoost, the stateless native-categorical builder for CatBoost.
* Explanation: exact SHAP values of both models, grouped into the five
  characteristics of src/tuning/feature_importance.py and combined with the stack
  weights, shown as multiplicative factors on the price.
* Typical range: the 25th-75th percentile of actual / predicted price of the final
  model on Validation 2024, per property type (Test 2025 is not read).
* Historical median: the project baseline rule (training-only medians).
* 'Check a real 2025 sale' mode: shows the actual price of a Test 2025 sale next to the
  frozen model's estimate. Display only; nothing is refitted or re-selected.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import Pool

from src.feature_engineering import CATBOOST_CATEGORICAL, FeaturePreprocessor, catboost_features
from src.modeling import baseline as bl
from src.modeling import config as cfg
from src.modeling import data_loading as dl
from src.modeling.evaluation import load_json, log_to_price
from src.tuning import feature_importance as fi
from src.tuning import tune_models as tm

LAST_TRAINING_YEAR = 2023


class PricePredictor:
    def __init__(self):
        self.pre = FeaturePreprocessor.load(cfg.DATA_DIR / 'preprocessor_unscaled.json')
        self.x_names = dl.feature_names()['unscaled']
        x_key, c_key = cfg.TUNED_KEYS['xgboost'], cfg.TUNED_KEYS['catboost']
        self.n_x = int(load_json(cfg.validation_metrics_path(x_key))['best_iteration'])
        self.xgb = tm.load_model('xgboost', cfg.MODEL_FILES[x_key])
        self.cat = tm.load_model('catboost', cfg.MODEL_FILES[c_key])
        meta = load_json(cfg.MODEL_FILES[cfg.STACK_KEY])
        self.w, self.intercept = meta['coefficients'], float(meta['intercept'])
        self.bounds = dl.train_log_bounds()

        train_inputs, train_price = dl.load_inputs('train'), dl.load_price('train')
        self.baseline = bl.fit_baseline(train_inputs, train_price)
        counts = train_inputs.groupby(['region', 'city', 'city_district', 'property_type']).size()
        self.district_counts = counts
        self.locations = (train_inputs.groupby(['region', 'city', 'city_district']).size()
                          .rename('n').reset_index())

        val = pd.read_csv(cfg.predictions_path(cfg.STACK_KEY, 'validation'))
        ratio = val['actual_price'] / val['predicted_price']
        self.ratio_q = ratio.groupby(val['property_type']).quantile([0.25, 0.75]).unstack()
        self._sales = None

    # ---------- real 2025 sales (display only: the frozen model is not refitted or re-selected) ----------
    def sales_2025(self):
        """Test 2025 inputs with their actual price, for the demo's 'check a real sale' mode."""
        if self._sales is None:
            inputs, price = dl.load_inputs('test'), dl.load_price('test')
            dl.check_alignment('test', inputs, price)
            self._sales = inputs.assign(actual_price=price.to_numpy())
        return self._sales

    def filter_sales(self, region=None, city=None, district=None, property_type=None):
        s = self.sales_2025()
        for col, val in (('region', region), ('city', city), ('city_district', district),
                         ('property_type', property_type)):
            if val is not None:
                s = s[s[col] == val]
        return s

    def sale_choices(self, column, **filters):
        """Values of `column` among the matching 2025 sales, most frequent first."""
        return self.filter_sales(**filters)[column].value_counts().index.tolist()

    # ---------- location choices (training data only) ----------
    def regions(self):
        return self.locations.groupby('region')['n'].sum().sort_values(ascending=False).index.tolist()

    def cities(self, region):
        part = self.locations[self.locations['region'] == region]
        return part.groupby('city')['n'].sum().sort_values(ascending=False).index.tolist()

    def districts(self, region, city):
        part = self.locations[(self.locations['region'] == region) & (self.locations['city'] == city)]
        return part.sort_values('n', ascending=False)['city_district'].tolist()

    # ---------- prediction ----------
    def _frame(self, region, city, district, property_type, area, year, quarter):
        return pd.DataFrame({'area': [float(area)], 'year': [int(year)], 'quarter': [int(quarter)],
                             'region': [region], 'city': [city], 'city_district': [district],
                             'property_type': [property_type]})

    def predict_frame(self, raw):
        """ln(price) of the stack for raw input rows (same columns as X_*_raw)."""
        X = self.pre.transform(raw).astype(np.float32)
        F = dl._clean_catboost_frame(catboost_features(raw))
        log_x = tm.predict_log('xgboost', self.xgb, X, self.n_x)
        log_c = tm.predict_log('catboost', self.cat, F, None)
        return self.intercept + self.w['xgboost'] * log_x + self.w['catboost'] * log_c, X, F

    def estimate(self, region, city, district, property_type, area, year, quarter):
        raw = self._frame(region, city, district, property_type, area, year, quarter)
        stack_log, X, F = self.predict_frame(raw)
        price = float(log_to_price(stack_log, self.bounds)[0])

        # Exact SHAP of both models, grouped and stack-weighted (additive in ln price).
        sx = self.xgb.get_booster().predict(xgb.DMatrix(X), pred_contribs=True,
                                            iteration_range=(0, self.n_x))
        sc = self.cat.get_feature_importance(Pool(F, cat_features=CATBOOST_CATEGORICAL), type='ShapValues')
        gx = fi.grouped(sx[:, :-1], self.x_names, fi.xgb_group).iloc[0]
        gc = fi.grouped(sc[:, :-1], list(F.columns), fi.CAT_GROUP.get).iloc[0]
        contrib = self.w['xgboost'] * gx + self.w['catboost'] * gc
        base_log = self.intercept + self.w['xgboost'] * sx[0, -1] + self.w['catboost'] * sc[0, -1]
        factors = np.exp(contrib).sort_values(key=lambda s: -np.abs(np.log(s)))

        lo, hi = self.ratio_q.loc[property_type, 0.25], self.ratio_q.loc[property_type, 0.75]
        hist, usage = bl.predict_baseline(self.baseline, raw)
        source = ('district + property type' if usage['group_median_rows'] else
                  'property type (no training sales of this type in this district)')
        n_train = int(self.district_counts.get((region, city, district, property_type), 0))
        return {'price': price, 'low': price * lo, 'high': price * hi,
                'base_price': float(np.exp(base_log)), 'factors': factors,
                'historical_median': float(hist[0]), 'historical_source': source,
                'training_sales_in_district_type': n_train,
                'beyond_training_period': int(year) > LAST_TRAINING_YEAR,
                'clipped': bool(stack_log[0] < self.bounds[0] or stack_log[0] > self.bounds[1])}


def self_test(n=200):
    """The demo path must reproduce the saved Validation 2024 predictions of the final model."""
    p = PricePredictor()
    saved = pd.read_csv(cfg.predictions_path(cfg.STACK_KEY, 'validation'), index_col='row_id')
    raw = dl.load_inputs('validation')
    rows = raw.sample(n, random_state=cfg.RANDOM_STATE)
    stack_log, _, _ = p.predict_frame(rows)
    ours = log_to_price(stack_log, p.bounds)
    ok = np.allclose(ours, saved.loc[rows.index, 'predicted_price'].to_numpy(), rtol=1e-5)
    r = rows.iloc[0]
    one = p.estimate(r['region'], r['city'], r['city_district'], r['property_type'], r['area'], r['year'], r['quarter'])
    additive = np.isclose(np.log(one['base_price']) + np.log(one['factors']).sum(), np.log(one['price']), atol=1e-4) \
        or one['clipped']
    saved_t = pd.read_csv(cfg.predictions_path(cfg.STACK_KEY, 'test'), index_col='row_id')
    sales = p.sales_2025().sample(n, random_state=cfg.RANDOM_STATE)
    ok_t = np.allclose(log_to_price(p.predict_frame(sales.drop(columns='actual_price'))[0], p.bounds),
                       saved_t.loc[sales.index, 'predicted_price'].to_numpy(), rtol=1e-5)         and np.allclose(sales['actual_price'], saved_t.loc[sales.index, 'actual_price'])
    ok = ok and ok_t
    print(f'{n} validation rows reproduced: {ok}; {n} Test 2025 sales match saved predictions: {ok_t}; '
          f'single-row SHAP factors multiply to the price: {additive}')
    print({k: v for k, v in one.items() if k != 'factors'})
    print(one['factors'].round(3).to_dict())
    if not (ok and additive):
        raise SystemExit(1)


if __name__ == '__main__':
    self_test()
