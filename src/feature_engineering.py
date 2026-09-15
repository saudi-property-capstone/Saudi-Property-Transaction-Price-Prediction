"""
Feature Engineering
MOJ Sales, 2020 Q1 - 2025 Q4 (cleaned/processed dataset)

 1. Select the permitted predictor columns from the current cleaned schema.
 2. Build numeric/date features and composite location keys.
 3. Learn location frequencies from training rows only.
 4. Prepare a natural-log price target separately from predictors.
 5. Target-encode regions and one-hot encode broad property classifications.
 6. Provide scaled MLP inputs and unscaled XGBoost inputs.
 7. Provide a separate native-categorical feature path for CatBoost (no
    frequency/target/one-hot encoding, no scaling; categories stay as strings).

Called by prepare_features.py after the chronological split. Transaction price
is not a predictor, and learned preprocessing is not refitted on validation/test.
The source dataset is never modified by these transformations.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
# ============================================================
# CONFIG / FEATURE COLUMNS
# ============================================================

# The current main branch retains only these broad property-use categories.
# Do not import the cleaning script or rerun any cleaning from feature code.
ALLOWED_PROPERTY_TYPES = frozenset({'Residential', 'Commercial', 'Agricultural'})

# Explicit allowlist: extra dataset columns cannot silently become predictors.
# Price, price-per-m² fields, and transaction identifiers are excluded.
RAW_FEATURES = ['area', 'year', 'quarter', 'region', 'city',
                'city_district', 'property_type']
# Fixed engineered-column order, also used when saving feature names.
# Used by XGBoost (unscaled) and MLP (scaled).
NUMERIC = ['area', 'time_index',
           'quarter_sin', 'quarter_cos', 'city_frequency', 'district_frequency']
CATEGORICAL = ['region', 'property_type']
TARGET_ENCODING_FOLDS = 5
RANDOM_STATE = 42

# CatBoost uses its own feature set: the same numeric/time columns, but raw
# categorical columns kept as strings instead of frequency/target/one-hot encoded.
CATBOOST_NUMERIC = ['area', 'time_index', 'quarter_sin', 'quarter_cos']
CATBOOST_CATEGORICAL = ['region', 'city', 'city_district', 'property_type']


# ============================================================
# CATEGORY LABEL HELPER
# ============================================================

def category(series):
    """Return consistent text labels without learning from another split."""
    # Use one consistent label for missing/blank categories in every split.
    return series.astype('string').str.strip().replace('', pd.NA).fillna('Unknown').astype(str)


def _area_and_time(X):
    """Validate and build the area/time columns shared by every model's features."""
    area = pd.to_numeric(X['area'], errors='coerce')
    if not (np.isfinite(area) & area.gt(0)).all():
        raise ValueError('Area must be finite and positive; report invalid input to the cleaning owner')
    year = pd.to_numeric(X['year'], errors='raise')
    quarter = pd.to_numeric(X['quarter'], errors='raise')
    if not (np.isfinite(year) & (year % 1 == 0) & quarter.isin([1, 2, 3, 4])).all():
        raise ValueError('Year must be an integer and quarter must be 1-4')
    # A sequential index represents the quarter's position since 2020 Q1.
    # Sine/cosine represent annual seasonality with Q4 adjacent to Q1.
    time_index = (year - 2020) * 4 + quarter - 1
    quarter_sin = np.sin(2 * np.pi * (quarter - 1) / 4)
    quarter_cos = np.cos(2 * np.pi * (quarter - 1) / 4)
    return area, time_index, quarter_sin, quarter_cos


# ============================================================
# STEP 1: PREDICTOR SELECTION
# ============================================================

def select_inputs(frame):
    """Select predictors from the current main branch's cleaned schema."""
    required = {'area', 'year', 'quarter', 'city', 'city_district',
                'property_type', 'region_en', 'region_raw'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    # Build a separate feature frame; do not rewrite the cleaned source dataset.
    out = frame[['area', 'year', 'quarter', 'city', 'city_district']].copy()
    # Prefer mapped English regions; retain the raw label when mapping is missing.
    region = frame['region_en'].astype('string').str.strip().replace('', pd.NA)
    out['region'] = region.fillna(frame['region_raw'].astype('string'))
    # Main already supplies canonical broad-use names. No schema conversion or
    # cleaning runs here; the entry script checks that the input matches main.
    out['property_type'] = frame['property_type']
    unexpected = set(out['property_type'].dropna()) - ALLOWED_PROPERTY_TYPES
    if unexpected:
        raise ValueError('Unrecognized property_type values; report them to the cleaning owner')
    # Main has already removed physical classification. It is not a predictor.
    return out[RAW_FEATURES]


# ============================================================
# STEPS 2-3: FEATURE CONSTRUCTION AND TRAINING FREQUENCIES
# ============================================================

class PropertyFeatures:
    """Learn only location frequencies; never read transaction price."""
    # --- Composite keys keep same-name locations separate ---
    def _keys(self, X):
        """Build region/city and region/city/district keys; preserve row order."""
        region = category(X['region'])
        city = category(X['city'])
        district = category(X['city_district'])
        # JSON tuple keys avoid name collisions between different regions/cities.
        cities = pd.Series([json.dumps(x, ensure_ascii=False) for x in zip(region, city)], index=X.index)
        districts = pd.Series([json.dumps(x, ensure_ascii=False) for x in zip(region, city, district)], index=X.index)
        return cities, districts

    # --- Fit: learn location frequencies from the training frame only ---
    def fit(self, X, y=None):
        """Store location shares from the supplied training inputs; y is unused."""
        if len(X) == 0:
            raise ValueError('Cannot fit on an empty training set')
        cities, districts = self._keys(X)
        # normalize=True returns each location's share of TRAINING rows.
        # y is unused: these encodings do not depend on transaction prices.
        self.city_frequencies_ = cities.value_counts(normalize=True).to_dict()
        self.district_frequencies_ = districts.value_counts(normalize=True).to_dict()
        return self

    # --- Transform: apply features without updating learned state ---
    def transform(self, X):
        """Return engineered columns in a fixed order for the supplied rows."""
        # transform only applies existing state; it does not update frequencies.
        if not hasattr(self, 'city_frequencies_'):
            raise ValueError('Fit training location frequencies before transforming inputs')
        out = pd.DataFrame(index=X.index)
        # --- Area, quarter trend and seasonality (shared with CatBoost features) ---
        # Keep area unchanged here. Scaling, if requested, happens downstream.
        # Price is the only logarithmic transformation required by the proposal.
        area, time_index, quarter_sin, quarter_cos = _area_and_time(X)
        out['area'] = area
        out['time_index'] = time_index
        out['quarter_sin'] = quarter_sin
        out['quarter_cos'] = quarter_cos
        cities, districts = self._keys(X)
        # --- Training-derived location encoding and categorical labels ---
        # Locations absent from training receive zero; validation/test rows do not
        # contribute to these frequencies. Composite keys separate same-name places.
        out['city_frequency'] = cities.map(self.city_frequencies_).fillna(0)
        out['district_frequency'] = districts.map(self.district_frequencies_).fillna(0)
        for col in CATEGORICAL:
            out[col] = category(X[col])
        return out[NUMERIC + CATEGORICAL]

    def fit_transform(self, X, y=None):
        """Learn training location frequencies, then construct training features."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Expose the column order used by the downstream preprocessing steps."""
        return np.asarray(NUMERIC + CATEGORICAL, dtype=object)


# ============================================================
# CATBOOST FEATURES (separate, native-categorical path)
# ============================================================

def catboost_features(X):
    """Build CatBoost inputs: engineered numerics plus raw string categoricals.

    Unlike PropertyFeatures, there is no learned state here: no city/district
    frequency encoding, no region target encoding, no one-hot encoding, and no
    scaling. CatBoost consumes region/city/city_district/property_type as
    native categorical strings directly. Row order and index are preserved.
    """
    area, time_index, quarter_sin, quarter_cos = _area_and_time(X)
    out = pd.DataFrame(index=X.index)
    out['area'] = area
    out['time_index'] = time_index
    out['quarter_sin'] = quarter_sin
    out['quarter_cos'] = quarter_cos
    for col in CATBOOST_CATEGORICAL:
        # Reuse the same missing-category label used by the other feature path.
        out[col] = category(X[col])
    return out[CATBOOST_NUMERIC + CATBOOST_CATEGORICAL]


# ============================================================
# STEP 4: LOG-PRICE TARGET
# ============================================================

def log_price_target(price):
    """Return ln(price) with the original row index; never repair invalid labels."""
    values = pd.to_numeric(price, errors='coerce')
    if not (np.isfinite(values) & values.gt(0)).all():
        raise ValueError('Log-price target requires finite positive prices; report to the cleaning owner')
    # Natural log is used, NOT log1p. The matching inverse is np.exp(prediction).
    # Keep original prices separately for eventual MAE/RMSE/R2 evaluation in SAR.
    return np.log(values).rename('log_price')


# ============================================================
# STEPS 5-6: REGION ENCODING AND MODEL-SPECIFIC PREPROCESSING
# ============================================================

class FeaturePreprocessor:
    """Fit encodings/scaling with pandas and NumPy, and save state as plain JSON."""

    def __init__(self, scale_numeric=True, target_cv=TARGET_ENCODING_FOLDS):
        self.scale_numeric = bool(scale_numeric)
        self.target_cv = target_cv

    @staticmethod
    def _region_means(regions, targets):
        """Compute arithmetic means of log-price labels for each training region."""
        frame = pd.DataFrame({'region': regions.to_numpy(), 'target': targets})
        return frame.groupby('region')['target'].mean().to_dict()

    def _combine(self, features, region_values):
        """Build the fixed numeric, region and one-hot column order."""
        # Categories come only from training. An unseen type gets all zeros.
        one_hot = np.column_stack([
            features['property_type'].eq(label).to_numpy(dtype=float)
            for label in self.state_['property_types']
        ])
        return np.column_stack([
            features[NUMERIC].to_numpy(dtype=float), region_values, one_hot,
        ])

    def _scale(self, matrix):
        """Apply training means/stds to numeric and regional columns only."""
        result = matrix.copy()
        if self.scale_numeric:
            count = len(NUMERIC) + 1
            result[:, :count] = (
                result[:, :count] - np.asarray(self.state_['scaling_mean'])
            ) / np.asarray(self.state_['scaling_std'])
        return result

    def fit_transform(self, X, y):
        """Learn from training and return its out-of-fold region encodings.

        y must already contain ln(price), aligned to X. Encoding folds stay
        inside the outer training split; no validation or test labels are used.
        """
        # --- Check labels and row alignment before learning anything ---
        targets = np.asarray(y, dtype=float)
        if targets.ndim != 1 or len(targets) != len(X) or not np.isfinite(targets).all():
            raise ValueError('Training log targets must be finite and match input rows')
        if isinstance(y, pd.Series) and not y.index.equals(X.index):
            raise ValueError('Training inputs and log targets have different row indices')
        if not isinstance(self.target_cv, int) or not 2 <= self.target_cv <= len(X):
            raise ValueError('Encoding folds must be an integer between 2 and training row count')

        # --- Learn input-only frequency mappings and category vocabulary ---
        self.features_ = PropertyFeatures()
        features = self.features_.fit_transform(X)
        regions = features['region']
        self.state_ = {
            'format': 'numpy-pandas-preprocessor-v1',
            'scale_numeric': self.scale_numeric,
            'training_rows': len(X),
            'target_encoding_folds': self.target_cv,
            'random_state': RANDOM_STATE,
            'region_encoding': 'arithmetic mean of training log_price; no smoothing',
            'city_frequencies': self.features_.city_frequencies_,
            'district_frequencies': self.features_.district_frequencies_,
            'property_types': sorted(features['property_type'].unique().tolist()),
            'region_means': self._region_means(regions, targets),
            'region_global_mean': float(targets.mean()),
        }

        # --- Encode each training fold using only the OTHER training folds ---
        # Shuffling uses only row positions. These are internal encoding folds,
        # not forward-only time folds or a replacement for chronological testing.
        positions = np.random.RandomState(RANDOM_STATE).permutation(len(X))
        encoded = np.empty(len(X), dtype=float)
        for held_out in np.array_split(positions, self.target_cv):
            fit_mask = np.ones(len(X), dtype=bool)
            fit_mask[held_out] = False
            mapping = self._region_means(regions.iloc[fit_mask], targets[fit_mask])
            # A region absent from the other folds uses their global mean.
            # The held-out fold's targets never enter its mapping or fallback.
            encoded[held_out] = regions.iloc[held_out].map(mapping).fillna(
                float(targets[fit_mask].mean())
            ).to_numpy(dtype=float)

        # --- Fit scaling on actual out-of-fold training feature values ---
        matrix = self._combine(features, encoded)
        count = len(NUMERIC) + 1
        means = matrix[:, :count].mean(axis=0)
        stds = matrix[:, :count].std(axis=0, ddof=0)
        # Constant columns become zero after centering; use a divisor of one.
        constant = np.ptp(matrix[:, :count], axis=0) == 0
        means[constant] = matrix[0, :count][constant]
        stds[constant | (stds == 0)] = 1.0
        self.state_['scaling_mean'] = means.tolist()
        self.state_['scaling_std'] = stds.tolist()
        self.state_['feature_names'] = (
            ['numeric__' + name for name in NUMERIC]
            + ['region_target__region']
            + ['category__property_type_' + value for value in self.state_['property_types']]
        )
        return self._scale(matrix)

    def transform(self, X):
        """Apply saved training state; labels are neither accepted nor accessed."""
        if not hasattr(self, 'state_'):
            raise ValueError('Fit preprocessing on training before calling transform')
        features = self.features_.transform(X)
        encoded = features['region'].map(self.state_['region_means']).fillna(
            self.state_['region_global_mean']
        ).to_numpy(dtype=float)
        return self._scale(self._combine(features, encoded))

    def get_feature_names_out(self):
        """Return the saved feature order for both output variants."""
        return np.asarray(self.state_['feature_names'], dtype=str)

    def save(self, path):
        """Write only learned labels and numeric settings, using standard JSON."""
        Path(path).write_text(json.dumps(self.state_, ensure_ascii=False, indent=2,
                                        allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        """Restore a preprocessor without importing model/persistence libraries."""
        state = json.loads(Path(path).read_text(encoding='utf-8'))
        if state.get('format') != 'numpy-pandas-preprocessor-v1':
            raise ValueError('Unrecognized preprocessing JSON format')
        result = cls(state['scale_numeric'], state['target_encoding_folds'])
        result.state_ = state
        result.features_ = PropertyFeatures()
        result.features_.city_frequencies_ = state['city_frequencies']
        result.features_.district_frequencies_ = state['district_frequencies']
        return result


def make_preprocessor(scale_numeric=True, target_cv=TARGET_ENCODING_FOLDS):
    """Create scaled MLP or unscaled XGBoost preprocessing.

    CatBoost does not use this preprocessor; see catboost_features() instead.
    """
    return FeaturePreprocessor(scale_numeric, target_cv)
