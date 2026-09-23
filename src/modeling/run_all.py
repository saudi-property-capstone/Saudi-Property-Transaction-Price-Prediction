"""
Run the whole initial-model stage, resuming safely.

Run:  python -m src.modeling.run_all

Each step is a separate script executed in its own process. Finished models
are skipped by their own script, so re-running this command after an
interruption resumes from the first unfinished step. To retrain one model
from scratch: python -m src.modeling.train_<model> --force
(then delete outputs/initial_models/models/frozen_configs.json and outputs/initial_models/status/*_test_metrics.json
if you also want to redo the frozen test evaluation).
"""
import subprocess
import sys

from src.modeling import config as cfg

STEPS = [
    ('smoke test', 'smoke_test.py'),
    ('baseline (validation)', 'baseline.py'),
    ('XGBoost', 'train_xgboost.py'),
    ('CatBoost', 'train_catboost.py'),
    ('MLP', 'train_mlp.py'),
    ('validation comparison + freeze', 'compare_validation.py'),
    ('final test evaluation', 'evaluate_test.py'),
    ('plots', 'make_plots.py'),
    ('report', 'make_report.py'),
]

if __name__ == '__main__':
    for name, script in STEPS:
        print(f'\n===== {name} ({script}) =====', flush=True)
        # Module execution from the repository root, so `src` imports resolve reliably.
        module = 'src.modeling.' + script[:-3]
        code = subprocess.run([sys.executable, '-m', module], cwd=cfg.ROOT).returncode
        if code != 0:
            print(f'\nStopped at step "{name}" (exit code {code}). Fix the cause and re-run '
                  f'`python -m src.modeling.run_all`; finished steps are skipped.')
            sys.exit(code)
    print('\nAll steps finished.')
