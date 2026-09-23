"""
Load the files written by src/prepare_features.py and check they line up.

Nothing here refits any preprocessing. Train and validation matrices are read
as exported. Test features do not exist on disk yet (prepare_features.py keeps
2025 untransformed), so `build_test_*` applies the *saved training-fitted*
preprocessors to the raw 2025 inputs. Only evaluate_test.py and the smoke test
(features only, no labels) call those functions.
"""
import json

import numpy as np
import pandas as pd

from src.feature_engineering import (CATBOOST_CATEGORICAL, CATBOOST_NUMERIC,
                                     FeaturePreprocessor, catboost_features, category)
from src.modeling import config as cfg

TEXT_COLUMNS = {'region': str, 'city': str, 'city_district': str, 'property_type': str}


def feature_names():
    return json.loads(cfg.FEATURE_NAMES_JSON.read_text(encoding='utf-8'))


def _read(name, **kwargs):
    return pd.read_csv(cfg.DATA_DIR / name, index_col='row_id', **kwargs)


# ---------- Raw inputs, price, metadata ----------

def load_inputs(split):
    """Raw predictors (area, year, quarter, region, city, city_district,
    property_type) for one split, indexed by row_id."""
    return _read(f'X_{split}_raw.csv.gz', dtype=TEXT_COLUMNS)


def load_price(split):
    """Original price in SAR for one split (the label; never a feature)."""
    return _read(f'y_{split}.csv.gz')['price']


def load_meta(split, inputs=None):
    """year / quarter / property_type per row, in the same row order as the features."""
    inputs = load_inputs(split) if inputs is None else inputs
    return inputs[['year', 'quarter', 'property_type']].copy()


def load_log_target(split):
    """ln(price) target exported by prepare_features.py (train and validation only).

    Already log-transformed: it is used as-is, never transformed again.
    """
    if split not in ('train', 'validation'):
        raise ValueError('Log targets are exported for train and validation only')
    return _read(f'y_{split}_log.csv.gz')['log_price']


def train_log_bounds():
    """(min, max) of the TRAINING ln(price); used to clip predictions before exp()."""
    log_train = load_log_target('train')
    return float(log_train.min()), float(log_train.max())


def check_alignment(split, inputs, price, log_target=None):
    """Same rows, same order, and log target really is ln(price)."""
    if not inputs.index.equals(price.index):
        raise ValueError(f'{split}: X_raw and y price row_ids differ')
    metadata = _read(f'{split}_metadata.csv.gz')
    if not inputs.index.equals(metadata.index) or not inputs['year'].equals(metadata['year']):
        raise ValueError(f'{split}: metadata is not aligned with X_raw')
    if log_target is not None:
        if not inputs.index.equals(log_target.index):
            raise ValueError(f'{split}: log target row_ids differ from X_raw')
        if not np.allclose(np.exp(log_target.to_numpy()), price.to_numpy(), rtol=1e-6):
            raise ValueError(f'{split}: exported target is not natural-log(price)')


# ---------- Feature matrices ----------

def load_matrix(variant, split):
    """Exported numeric matrix: variant 'unscaled' (XGBoost) or 'scaled' (MLP).

    Stored as float32 to halve memory; the values are frequencies, encodings,
    scaled numbers and 0/1 flags, none of which need float64.
    """
    if split not in ('train', 'validation'):
        raise ValueError('Matrices are exported for train and validation only')
    matrix = np.load(cfg.DATA_DIR / f'X_{split}_{variant}.npy', allow_pickle=False)
    return matrix.astype(np.float32)


def load_catboost_frame(split):
    """Native-categorical CatBoost features (categoricals kept as strings)."""
    if split not in ('train', 'validation'):
        raise ValueError('CatBoost frames are exported for train and validation only')
    frame = _read(f'X_{split}_catboost.csv.gz', dtype={c: str for c in CATBOOST_CATEGORICAL})
    return _clean_catboost_frame(frame)


def _clean_catboost_frame(frame):
    """Missing categories become the string 'Unknown' (same rule as the pipeline)."""
    frame = frame.copy()
    for col in CATBOOST_CATEGORICAL:
        frame[col] = category(frame[col])       # NaN/blank -> 'Unknown', always str
    return frame[CATBOOST_NUMERIC + CATBOOST_CATEGORICAL]


# ---------- Test features (final evaluation only) ----------

def build_test_matrix(variant):
    """Transform raw 2025 inputs with the SAVED training-fitted preprocessor.

    No test labels are read and nothing is refitted.
    """
    preprocessor = FeaturePreprocessor.load(cfg.DATA_DIR / f'preprocessor_{variant}.json')
    return preprocessor.transform(load_inputs('test')).astype(np.float32)


def build_test_catboost_frame():
    """CatBoost features for 2025 (this builder has no learned state)."""
    return _clean_catboost_frame(catboost_features(load_inputs('test')))


def transform_with_saved_preprocessor(variant, inputs):
    """Apply a saved preprocessor to any raw inputs (used by the smoke test check)."""
    preprocessor = FeaturePreprocessor.load(cfg.DATA_DIR / f'preprocessor_{variant}.json')
    return preprocessor.transform(inputs).astype(np.float32)
