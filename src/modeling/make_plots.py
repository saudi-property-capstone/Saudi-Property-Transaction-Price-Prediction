"""
Create the comparison plots from the saved result tables (no models are run).

Run: python -m src.modeling.make_plots

Outputs (outputs/figures/): model_comparison_mae.png, model_comparison_rmse.png,
model_comparison_r2.png, mlp_training_history.png
Colours are the design-system categorical slots 1 and 2 (blue = Validation,
orange = Test) in fixed order; every bar also carries a value label.
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modeling import config as cfg
from src.modeling.evaluation import stage

VAL_COLOR, TEST_COLOR = '#2a78d6', '#eb6834'
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def comparison_plot(val, test, val_col, test_col, title, ylabel, filename, fmt, baseline_note):
    models = val['Model'].tolist()
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    style_axes(ax)
    for offset, frame, col, color, label in ((-width / 2, val, val_col, VAL_COLOR, 'Validation 2024'),
                                             (width / 2, test, test_col, TEST_COLOR, 'Test 2025')):
        values = frame[col].to_numpy(float)
        bars = ax.bar(x + offset, values, width - 0.04, color=color, label=label)
        for bar, v in zip(bars, values):
            # Label sits above positive bars and below negative ones.
            ax.annotate(fmt(v), (bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 3 if v >= 0 else -11), textcoords='offset points',
                        ha='center', fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('Historical Median Baseline', 'Historical Median\nBaseline')
                        for m in models], color=INK)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, loc='left', color=INK, fontsize=13, fontweight='bold', pad=30)
    # Legend above the plot area so it can never cover a bar or label.
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK)
    ax.margins(y=0.12)
    if ylabel != 'R²':
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v / 1e6:g}M'))
    fig.text(0.01, 0.01, baseline_note, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(cfg.FIGURES_DIR / filename, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def history_plot():
    hist = pd.read_csv(cfg.TABLES_DIR / 'mlp_training_history.csv')
    best = int(hist['val_loss'].idxmin())
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    style_axes(ax)
    ax.plot(hist['epoch'], hist['loss'], color=VAL_COLOR, linewidth=2, label='Training loss')
    ax.plot(hist['epoch'], hist['val_loss'], color=TEST_COLOR, linewidth=2,
            linestyle='--', label='Validation 2024 loss')
    ax.scatter([hist.loc[best, 'epoch']], [hist.loc[best, 'val_loss']], s=50, color=TEST_COLOR,
               edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"best epoch {int(hist.loc[best, 'epoch'])}",
                (hist.loc[best, 'epoch'], hist.loc[best, 'val_loss']),
                xytext=(8, 10), textcoords='offset points', fontsize=9, color=INK)
    ax.set_xlabel('Epoch', color=MUTED)
    ax.set_ylabel('MSE on ln(price)', color=MUTED)
    ax.set_title('MLP training history', loc='left', color=INK, fontsize=13, fontweight='bold')
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / 'mlp_training_history.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    cfg.make_output_dirs()
    with stage('plots / model comparison'):
        val = pd.read_csv(cfg.TABLES_DIR / 'initial_validation_comparison.csv')
        test = pd.read_csv(cfg.TABLES_DIR / 'initial_test_comparison.csv')
        note = 'Initial untuned models. Errors are on the original SAR scale.'
        comparison_plot(val, test, 'Validation MAE (SAR)', 'Test MAE (SAR)',
                        'MAE by model (lower is better)', 'MAE (SAR)',
                        'model_comparison_mae.png', lambda v: f'{v / 1e3:,.0f}k', note)
        comparison_plot(val, test, 'Validation RMSE (SAR)', 'Test RMSE (SAR)',
                        'RMSE by model (lower is better)', 'RMSE (SAR)',
                        'model_comparison_rmse.png', lambda v: f'{v / 1e6:,.2f}M', note)
        comparison_plot(val, test, 'Validation R2 (SAR)', 'Test R2 (SAR)',
                        'R² by model (higher is better)', 'R²',
                        'model_comparison_r2.png', lambda v: f'{v:.3f}', note)
    with stage('plots / MLP training history'):
        history_plot()


if __name__ == '__main__':
    main()
