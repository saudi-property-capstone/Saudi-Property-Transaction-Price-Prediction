"""Model Performance: saved evaluation results of the baseline, initial, tuned and final models."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard import charts, results
from dashboard import theme as th
from dashboard.fmt import amount_html, compact, count
from src.modeling import config as cfg

th.render_page_header('Model Performance',
                      'How the final model performs on unseen 2025 sales, how it compares with the other models and '
                      'the baseline, and what drives its predictions.',
                      icon='leaderboard')

sel, val, test = results.final_selection(), results.validation_table(), results.test_table()
if sel is None or val is None or test is None:
    th.missing('The saved evaluation tables', 'outputs/tuning/tables/', 'Run the tuning pipeline (README, section 7).')
    st.stop()

FINAL = sel['selected_model']
BASELINE = cfg.MODEL_LABELS['baseline']
final_test = test[test['Model'] == FINAL].iloc[0]
target = val.loc[val['Model'] == BASELINE, 'MAE'].iloc[0] * (1 - cfg.TARGET_IMPROVEMENT_PCT / 100)

st.success(f'**Final model: Ridge Stacking Ensemble** Tuned XGBoost + Tuned CatBoost, selected on Validation 2024 and evaluated on Test 2025.', icon=':material/verified:')

th.section('Final model on unseen 2025 sales')
th.cards([
    {'label': 'MAE', 'value': amount_html(final_test['MAE']), 'cls': 'primary',
     'sub': f'{final_test["MAE vs baseline"]:.1f}% lower than the baseline'},
    {'label': 'Median absolute error', 'value': amount_html(final_test['Median AE']),
     'sub': 'typical error, robust to very large deals'},
    {'label': 'RMSE', 'value': amount_html(final_test['RMSE']), 'sub': 'dominated by the largest errors'},
    {'label': 'R² (SAR) · R² (ln price)', 'value': f'{final_test["R²"]:.3f} · {final_test["R² (ln price)"]:.3f}',
     'help': 'R² on the SAR scale is pulled down by a few very large transactions; R² on ln(price) measures how well '
             'relative price differences are captured.'},
])

# ---------------- Comparison tables ----------------
th.section('Model comparison', 'The final model is highlighted. MAE is the primary metric; the success criterion '
                               f'was at least {cfg.TARGET_IMPROVEMENT_PCT:.0f}% lower MAE than the historical-median '
                               'baseline.')


def styled(df):
    view = df[['Model', 'Stage', 'MAE', 'RMSE', 'R²', 'R² (ln price)', 'Median AE', 'MAE vs baseline']].copy()
    fmt = {'MAE': '{:,.0f}', 'RMSE': '{:,.0f}', 'Median AE': '{:,.0f}', 'R²': '{:.3f}', 'R² (ln price)': '{:.3f}',
           'MAE vs baseline': '{:+.1f}%'}
    hl = lambda row: ([f'background-color: {th.GREEN_TINT}; font-weight: 600'] * len(row)  # noqa: E731
                      if row['Model'] == FINAL else [''] * len(row))
    return view.style.format(fmt).apply(hl, axis=1)


COLS = {'MAE': st.column_config.Column('MAE (SAR)'), 'RMSE': st.column_config.Column('RMSE (SAR)'),
        'Median AE': st.column_config.Column('Median AE (SAR)'),
        'MAE vs baseline': st.column_config.Column('MAE change vs baseline')}
tab_test, tab_val = st.tabs(['Test 2025 (held-out)', 'Validation 2024 (model selection)'])
with tab_test:
    st.dataframe(styled(test), hide_index=True, width='stretch', column_config=COLS)
    th.note('Tuned individual models have no Test 2025 results by design: only the frozen final selection was '
            'evaluated on 2025. Initial models were tested at the end of the initial-modeling stage.')
with tab_val:
    st.dataframe(styled(val), hide_index=True, width='stretch', column_config=COLS)
    th.note('Every selection decision (hyperparameters, stacking, final model) used Validation 2024 only.')

v = val.sort_values('MAE')
colors = [th.GREEN if m == FINAL else th.SAND if m == BASELINE else '#A9B4AE' for m in v['Model']]
fig = charts.hbars(v['Model'].tolist(), v['MAE'].tolist(), 'Validation 2024 MAE by model (lower is better)',
                   money=True, colors=colors, value_label='MAE')
fig.add_vline(x=target, line=dict(color=th.INK2, dash='dash', width=1.5))
fig.add_annotation(x=target, y=0, yref='paper', yanchor='top', yshift=-4, text=f'15% target: {compact(target)} SAR',
                   showarrow=False, xanchor='center', font=dict(size=11, color=th.INK2))
fig.update_layout(margin=dict(b=30))
charts.show(fig)
th.note('Green: final model · sand: historical-median baseline · grey: other models. The MLP did not beat the '
        'tree models, which is reported as a negative result.')

with st.expander('What do MAE, RMSE and R² mean here?', icon=':material/help:'):
    th.info_cards([
        ('Primary', 'MAE: mean absolute error', 'Average size of the error in SAR, whatever its direction. '
                                               ),
        ('Sensitive to large errors', 'RMSE: root mean squared error',
         'Squares errors before averaging, so a few very expensive transactions with large errors dominate it. '
         'That is why RMSE is much larger than MAE here.'),
        ('Explained variance', 'R²', 'Share of price variation explained (1 = perfect, 0 = no better than the mean). '
                                     'On the SAR scale it is pulled down by extreme deals; on ln(price) it reflects '
                                     'relative accuracy across ordinary transactions.'),
    ])
    th.note('The median absolute error is also shown: half of the transactions have an error below it, so it '
            'describes a typical sale better than the mean when a few deals are extremely large.')

# ---------------- Per property type ----------------
per_type = results.per_type_test()
if per_type is not None:
    th.section('Test 2025 MAE by property type')
    fig = go.Figure()
    for model, color in ((BASELINE, '#A9B4AE'), (FINAL, th.GREEN)):
        part = per_type[per_type['Model'] == model].set_index('Property type').reindex(th.PROPERTY_TYPES)
        fig.add_trace(go.Bar(x=part.index, y=part['MAE (SAR)'], name='Baseline' if model == BASELINE else 'Final model',
                             marker_color=color, text=[f'{compact(v)} SAR' for v in part['MAE (SAR)']],
                             textposition='outside', cliponaxis=False, textfont=dict(color=th.INK2),
                             customdata=[f'{v:,.0f}' for v in part['MAE (SAR)']],
                             hovertemplate='%{x}: %{customdata} SAR<extra>' + model + '</extra>'))
    charts.style(fig, 340, legend=True)
    fig.update_layout(bargroupgap=0.08)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    charts.show(fig)
    rows = per_type[per_type['Model'] == FINAL].set_index('Property type')
    th.note('Commercial and Agricultural properties have fewer sales ('
            + ', '.join(f'{t}: {count(rows.loc[t, "Test rows"])}' for t in ['Commercial', 'Agricultural'])
            + ' test rows) and much wider price ranges, so their absolute errors are larger.')

# ---------------- Predictions ----------------
th.section('Predictions vs actual prices (Test 2025)')
summary = results.test_predictions_summary(sel['selected_key'])
if summary is None:
    th.missing('Row-level test predictions', f"outputs/tuning/predictions/{sel['selected_key']}_test_predictions.csv.gz",
               'These files are not tracked in Git; run `python -m src.tuning.evaluate_final_test` to recreate them.')
else:
    c1, c2 = st.columns(2, gap='large')
    with c1:
        s = summary['sample']
        fig = go.Figure()
        for t, color in th.TYPE_COLORS.items():
            part = s[s['property_type'] == t]
            fig.add_trace(go.Scattergl(
                x=part['actual_price'], y=part['predicted_price'], mode='markers', name=t,
                marker=dict(size=5, color=color, opacity=0.45),
                customdata=np.column_stack([[f'{v:,.0f}' for v in part['actual_price']],
                                            [f'{v:,.0f}' for v in part['predicted_price']]]),
                hovertemplate='Actual: %{customdata[0]} SAR<br>Predicted: %{customdata[1]} SAR<extra>' + t + '</extra>'))
        lo, hi = 1e3, 1e9
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='Perfect prediction',
                                 line=dict(color=th.INK2, dash='dash', width=1.5), hoverinfo='skip'))
        charts.style(fig, 420, f'Actual vs predicted ({count(len(s))} random sales)', legend=True)
        ticks = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
        for ax in (fig.update_xaxes, fig.update_yaxes):
            ax(type='log', tickvals=ticks, ticktext=[compact(t) for t in ticks], range=[3.5, 9])
        fig.update_xaxes(title='Actual price, SAR (log scale)')
        fig.update_yaxes(title='Predicted price, SAR (log scale)')
        charts.show(fig)
    with c2:
        h = summary['pct_hist']
        colors = [th.GREEN if (lo_ >= -25 and hi_ <= 25) else '#A9B4AE' for lo_, hi_ in zip(h['lo'], h['hi'])]
        labels = [f'{lo_:+.0f}% to {hi_:+.0f}%' for lo_, hi_ in zip(h['lo'], h['hi'])]
        fig = go.Figure(go.Bar(x=(h['lo'] + h['hi']) / 2, y=h['count'], width=9, marker_color=colors,
                               customdata=labels, hovertemplate='Error %{customdata}<br>Sales: %{y:,.0f}'
                                                                '<extra></extra>'))
        charts.style(fig, 420, 'Error as % of the actual price (all test sales)')
        fig.update_layout(margin=dict(b=40))
        fig.update_xaxes(title='(predicted − actual) ÷ actual × 100',
                         ticksuffix='%', range=[-105, 305])
        fig.update_yaxes(tickformat=',.0f', title='Sales')
        charts.show(fig)
    th.cards([
        {'label': 'Within ±25% of the actual price', 'value': f'{summary["within_25"]:.1f}%', 'sub': 'of 2025 sales'},
        {'label': 'Within ±50% of the actual price', 'value': f'{summary["within_50"]:.1f}%', 'sub': 'of 2025 sales'},
       
    ])
    th.note('Just 1% of sales (rare, very large deals, mostly Commercial or Agricultural, median price 12M SAR) cause 46% of the total error. The reported MAE still includes all transactions.')

# ---------------- Interpretability ----------------
th.section('What drives the predictions (SHAP)',
           'How much each property characteristic pushes the final model\'s price predictions up or down, '
           'measured with exact SHAP values.')
by_char, by_model = results.feature_importance()
if by_char is None:
    th.missing('Saved SHAP tables', 'outputs/tuning/tables/feature_importance_by_characteristic.csv',
               'Run `python -m src.tuning.feature_importance`.')
else:
    # Share of importance on each bar, plus the typical price factor exp(mean |SHAP|) that the table used to show.
    ranked = by_char.sort_values('Stack share (%)')
    shares, factors = ranked['Stack share (%)'], ranked['Typical price factor (x)']
    fig = charts.hbars(by_char['Characteristic'].tolist(), by_char['Stack share (%)'].tolist(),
                       'Share of the final model’s importance by characteristic', value_label='Share')
    fig.update_traces(text=[f'{s:.1f}%  ·  ×{f:.2f}' for s, f in zip(shares, factors)],
                      customdata=np.column_stack([[f'{s:.1f}%' for s in shares],
                                                  [f'<br>Typical effect on price: ×{f:.2f} up or down' for f in factors]]))
    fig.update_xaxes(range=[0, shares.max() * 1.5])  # room for the longer labels
    charts.show(fig)
    loc = by_char.set_index('Characteristic')
    th.note(f'Location matters most: depending on the city and district, it typically moves a price about '
            f'{loc.loc["Location", "Typical price factor (x)"]:.1f}× up or down. Property area and the time trend come '
            f'next. Property type adds little on its own ({loc.loc["Property type", "Stack share (%)"]:.0f}%), likely '
            'because location and area already capture most of the difference between types. '
            'The ×value on each bar is the typical factor by which that characteristic raises or lowers the price.')
    with st.expander('Feature-level importance per base model'):
        st.dataframe(by_model, hide_index=True, width='stretch', column_config={
            'Mean |SHAP| (ln price)': st.column_config.NumberColumn(format='%.3f'),
            'Share of model total (%)': st.column_config.NumberColumn(format='%.1f%%')})
    figs = results.shap_figures()
    if figs:
        with st.expander('Saved SHAP figures'):
            for i in range(0, len(figs), 2):
                cols = st.columns(2)
                for col, (path, caption) in zip(cols, figs[i:i + 2]):
                    col.image(str(path), caption=caption, width='stretch')
