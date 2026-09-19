"""
Shared evaluation helpers: metrics on the original SAR scale, target inversion,
JSON/prediction saving, and a stage wrapper that labels failures.
"""
import contextlib
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.modeling import config as cfg


# ---------- Progress messages and clear failure labels ----------

@contextlib.contextmanager
def stage(label):
    """Print progress and re-raise any error with the failing stage named."""
    print(f'[{label}] started', flush=True)
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        raise RuntimeError(f'STAGE FAILED [{label}]: {type(exc).__name__}: {exc}') from exc
    print(f'[{label}] finished in {time.perf_counter() - start:.1f}s', flush=True)


# ---------- Target inversion ----------

def count_outside_bounds(log_prediction, bounds):
    """How many ln(price) predictions fall outside the training target range."""
    log_prediction = np.asarray(log_prediction, dtype=np.float64).ravel()
    return int(((log_prediction < bounds[0]) | (log_prediction > bounds[1])).sum())


def log_to_price(log_prediction, bounds=None):
    """Convert ln(price) predictions to SAR with exp (the project's inverse).

    The pipeline uses natural log, not log1p, so expm1 would be wrong here.
    Predictions must be finite. `bounds` = (min, max) of the TRAINING ln(price);
    predictions are clipped to that range first, so an extrapolating model cannot
    return absurd prices such as exp(48). Negative prices are clipped to zero.
    """
    log_prediction = np.asarray(log_prediction, dtype=np.float64).ravel()
    if not np.isfinite(log_prediction).all():
        raise ValueError('Model produced non-finite log-price predictions')
    if bounds is not None:
        log_prediction = np.clip(log_prediction, bounds[0], bounds[1])
    price = np.exp(log_prediction)
    if not np.isfinite(price).all():
        raise ValueError('exp() overflowed while converting predictions to SAR')
    return np.maximum(price, 0.0)


# ---------- Metrics ----------

def regression_metrics(y_true, y_pred):
    """MAE, RMSE, R² on whatever scale is passed in (we always pass SAR)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    r2 = np.nan
    if n >= 2 and np.var(y_true) > 0:
        r2 = float(r2_score(y_true, y_pred))
    return {'n_rows': int(n),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': r2}


def improvement_pct(baseline_mae, model_mae):
    """Percentage MAE reduction versus the baseline (positive = better)."""
    return float((baseline_mae - model_mae) / baseline_mae * 100.0)


def per_property_type_metrics(frame, model_label):
    """Metrics per property type from a frame with property_type/actual/predicted.

    R² is left empty (NaN) with an explanation when it cannot be trusted.
    """
    rows = []
    for ptype in cfg.PROPERTY_TYPES:
        part = frame[frame['property_type'] == ptype]
        if part.empty:
            rows.append({'model': model_label, 'property_type': ptype, 'n_rows': 0,
                         'mae': np.nan, 'rmse': np.nan, 'r2': np.nan,
                         'r2_note': 'no rows for this property type'})
            continue
        m = regression_metrics(part['actual_price'], part['predicted_price'])
        note = ''
        if np.isnan(m['r2']):
            note = 'R2 undefined: fewer than 2 rows or zero target variance'
        elif m['n_rows'] < 30:
            note = 'R2 unreliable: fewer than 30 rows'
        rows.append({'model': model_label, 'property_type': ptype, **m, 'r2_note': note})
    return rows


# ---------- Saving ----------

def _jsonable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Cannot serialise {type(obj)}')


def _nan_to_none(obj):
    """JSON has no NaN; store missing numbers as null."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and np.isnan(obj):
        return None
    return obj


def save_json(obj, path):
    """Write JSON through a temp file so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(_nan_to_none(obj), indent=2, default=_jsonable,
                              ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_predictions(path, meta, actual_price, predicted_price):
    """Save actual/predicted SAR prices with year and property type per row.

    `meta` must be indexed by row_id and hold year, quarter, property_type.
    """
    if not (len(meta) == len(actual_price) == len(predicted_price)):
        raise ValueError('Predictions and metadata have different lengths')
    out = pd.DataFrame({
        'year': meta['year'].to_numpy(),
        'quarter': meta['quarter'].to_numpy(),
        'property_type': meta['property_type'].to_numpy(),
        'actual_price': np.asarray(actual_price, dtype=np.float64),
        'predicted_price': np.asarray(predicted_price, dtype=np.float64),
    }, index=meta.index)
    out.index.name = 'row_id'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_name(Path(path).name + '.tmp')
    out.to_csv(tmp, compression={'method': 'gzip'})
    tmp.replace(path)
    return out


def validation_is_complete(key):
    """A model counts as finished only when its metrics file (written last) exists."""
    return cfg.validation_metrics_path(key).exists() and cfg.MODEL_FILES[key].exists()


def save_validation_results(key, config, meta_val, y_val_price, val_price_pred,
                            train_metrics, train_seconds, best_iteration, extra=None):
    """Save predictions, then the metrics JSON. The JSON is the 'done' marker."""
    val_metrics = regression_metrics(y_val_price, val_price_pred)
    save_predictions(cfg.predictions_path(key, 'validation'), meta_val,
                     y_val_price, val_price_pred)
    save_json(config, cfg.config_path(key))
    result = {
        'model_key': key, 'model': cfg.MODEL_LABELS[key], 'version': 'Initial',
        'target_transform': cfg.TARGET_TRANSFORM,
        'validation': val_metrics,
        'train': train_metrics,
        'training_seconds': float(train_seconds),
        'best_iteration': int(best_iteration),
        **(extra or {}),
    }
    save_json(result, cfg.validation_metrics_path(key))
    return result
