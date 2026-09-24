"""
Shared settings for the modeling code.

Every path, seed, and file name lives here so that no other script has to
guess where things are. Nothing in this file reads or writes data.

Two stages, two output folders (both under outputs/, never anywhere else):

    outputs/initial_models/   Initial Modeling: baseline + Initial XGBoost /
                              CatBoost / MLP  (code: src/modeling/)
    outputs/tuning/           Hyperparameter Tuning
                              (code: src/tuning/)

Initial Modeling code writes only to outputs/initial_models/. Tuning code may
READ initial artifacts but writes only to outputs/tuning/. The output folder
for a model is chosen from its key (see TUNING_KEYS and the path helpers).

Run scripts from the repository root as modules, e.g.
    python -m src.modeling.train_xgboost
"""
from pathlib import Path

# Repository root, found relative to this file (never an absolute user path).
ROOT = Path(__file__).resolve().parents[2]

# ---------- Inputs produced by earlier pipeline stages (read-only) ----------
DATA_DIR = ROOT / 'data' / 'processed'
OUTPUTS_DIR = ROOT / 'outputs'
# Feature-pipeline / EDA outputs are written by src/prepare_features.py and the
# EDA scripts into these two folders, so they stay where those scripts put them.
INPUT_TABLES_DIR = OUTPUTS_DIR / 'tables'
FEATURE_NAMES_JSON = INPUT_TABLES_DIR / 'feature_names.json'

# ---------- Initial Modeling outputs ----------
INITIAL_DIR = OUTPUTS_DIR / 'initial_models'
MODELS_DIR = INITIAL_DIR / 'models'
TABLES_DIR = INITIAL_DIR / 'tables'
PREDICTIONS_DIR = INITIAL_DIR / 'predictions'
FIGURES_DIR = INITIAL_DIR / 'figures'
LOGS_DIR = INITIAL_DIR / 'logs'
STATUS_DIR = INITIAL_DIR / 'status'          # one small JSON per finished stage
SMOKE_DIR = INITIAL_DIR / 'smoke_test'       # diagnostic only, never final results
REPORTS_DIR = INITIAL_DIR / 'reports'
BASELINE_SUMMARY_CSV = TABLES_DIR / 'baseline_median_summary.csv'   # written by notebooks/baseline_model.ipynb

# Canonical fitted preprocessing: data/processed/preprocessor_{scaled,unscaled}.json
# (written by src/prepare_features.py on Training 2020-2023 and loaded from there
# by all code). This folder holds a checksummed reproducibility snapshot of them.
PREPROCESSING_DIR = MODELS_DIR / 'preprocessing'
FROZEN_CONFIG_JSON = MODELS_DIR / 'frozen_configs.json'

# ---------- Hyperparameter Tuning outputs ----------
TUNING_DIR = OUTPUTS_DIR / 'tuning'
TUNING_MODELS_DIR = TUNING_DIR / 'models'
TUNING_TABLES_DIR = TUNING_DIR / 'tables'
TUNING_PREDICTIONS_DIR = TUNING_DIR / 'predictions'
TUNING_FIGURES_DIR = TUNING_DIR / 'figures'
TUNING_LOGS_DIR = TUNING_DIR / 'logs'
TUNING_STATUS_DIR = TUNING_DIR / 'status'
TUNING_REPORTS_DIR = TUNING_DIR / 'reports'
FROZEN_V2_JSON = TUNING_MODELS_DIR / 'frozen_configs_v2.json'

# ---------- Experiment settings ----------
RANDOM_STATE = 42
TARGET_IMPROVEMENT_PCT = 15.0                # project success criterion (vs baseline MAE)
PROPERTY_TYPES = ['Residential', 'Commercial', 'Agricultural']

# The pipeline trains on natural log(price) and inverts with np.exp
# (see src/feature_engineering.py: log_price_target). NOT log1p / expm1.
TARGET_TRANSFORM = 'natural log: y = ln(price); inverse: price = exp(y)'

# Initial Modeling models (only these belong to the initial stage).
MODEL_KEYS = ['xgboost', 'catboost', 'mlp']
MODEL_LABELS = {'xgboost': 'Initial XGBoost', 'catboost': 'Initial CatBoost',
                'mlp': 'Initial MLP', 'baseline': 'Historical Median Baseline'}
# Names used inside result files saved BEFORE the display names were introduced.
LEGACY_LABELS = {'XGBoost': 'Initial XGBoost', 'CatBoost': 'Initial CatBoost', 'MLP': 'Initial MLP'}
MODEL_FILES = {                               # trained model artifact per model
    'xgboost': MODELS_DIR / 'xgboost' / 'xgboost_initial.json',
    'catboost': MODELS_DIR / 'catboost' / 'catboost_initial.cbm',
    'mlp': MODELS_DIR / 'mlp' / 'mlp_initial.keras',
}

# Hyperparameter Tuning experiments (NOT initial models).
V2_XGB_KEY = 'xgboost_v2'
V2_ENSEMBLE_KEY = 'ensemble_v2'
TUNING_KEYS = {V2_XGB_KEY, V2_ENSEMBLE_KEY}
MODEL_LABELS[V2_XGB_KEY] = 'XGBoost v2 (min_child_weight=200) [tuning experiment]'
MODEL_LABELS[V2_ENSEMBLE_KEY] = 'Ensemble v2 (XGBoost v2 + Initial CatBoost, 50/50) [tuning experiment]'
MODEL_FILES[V2_XGB_KEY] = TUNING_MODELS_DIR / 'xgboost_v2' / 'xgboost_v2.json'

# Hyperparameter search + stacking (src/tuning/tune_models.py, stacking.py).
TUNED_KEYS = {'xgboost': 'xgboost_tuned', 'catboost': 'catboost_tuned', 'mlp': 'mlp_tuned'}
STACK_KEY = 'stack_tuned'
TUNING_KEYS |= set(TUNED_KEYS.values()) | {STACK_KEY}
MODEL_LABELS.update({'xgboost_tuned': 'Tuned XGBoost', 'catboost_tuned': 'Tuned CatBoost',
                     'mlp_tuned': 'Tuned MLP', STACK_KEY: 'Stack (Ridge on top-2 tuned models)'})
MODEL_FILES.update({
    'xgboost_tuned': TUNING_MODELS_DIR / 'xgboost_tuned' / 'xgboost_tuned.json',
    'catboost_tuned': TUNING_MODELS_DIR / 'catboost_tuned' / 'catboost_tuned.cbm',
    'mlp_tuned': TUNING_MODELS_DIR / 'mlp_tuned' / 'mlp_tuned.keras',
    STACK_KEY: TUNING_MODELS_DIR / STACK_KEY / 'stack_meta_ridge.json',
})
FROZEN_FINAL_JSON = TUNING_MODELS_DIR / 'frozen_final_selection.json'


# ---------- Path helpers: the model key decides the stage folder ----------

def validation_metrics_path(key):
    base = TUNING_STATUS_DIR if key in TUNING_KEYS else STATUS_DIR
    return base / f'{key}_validation_metrics.json'


def test_metrics_path(key):
    base = TUNING_STATUS_DIR if key in TUNING_KEYS else STATUS_DIR
    return base / f'{key}_test_metrics.json'


def config_path(key):
    base = TUNING_MODELS_DIR if key in TUNING_KEYS else MODELS_DIR
    return base / key / f'{key}_config.json'


def predictions_path(key, split):
    base = TUNING_PREDICTIONS_DIR if key in TUNING_KEYS else PREDICTIONS_DIR
    return base / f'{key}_{split}_predictions.csv.gz'


def make_output_dirs():
    """Create the Initial Modeling output folders (never the tuning ones)."""
    initial_files = [MODEL_FILES[k].parent for k in MODEL_KEYS]
    for folder in [MODELS_DIR, PREPROCESSING_DIR, TABLES_DIR, PREDICTIONS_DIR, FIGURES_DIR,
                   LOGS_DIR, STATUS_DIR, REPORTS_DIR, *initial_files]:
        folder.mkdir(parents=True, exist_ok=True)


def make_tuning_output_dirs():
    """Create the Hyperparameter Tuning output folders (never the initial ones)."""
    for folder in [TUNING_MODELS_DIR, TUNING_TABLES_DIR, TUNING_PREDICTIONS_DIR,
                   TUNING_FIGURES_DIR, TUNING_LOGS_DIR, TUNING_STATUS_DIR,
                   TUNING_REPORTS_DIR, MODEL_FILES[V2_XGB_KEY].parent,
                   *(MODEL_FILES[k].parent for k in [*TUNED_KEYS.values(), STACK_KEY])]:
        folder.mkdir(parents=True, exist_ok=True)
