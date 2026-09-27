"""Home: project overview, headline figures, and entry points to the other pages."""
import streamlit as st

from dashboard import charts, data, results
from dashboard import theme as th
from dashboard.fmt import count, esc

ov = data.overview()
sel = results.final_selection()
test = results.test_table()

th.render_page_header('Saudi Property Transaction Price Prediction',
                      'A data science capstone on <bdi>1.39 million</bdi> Saudi Ministry of Justice property sales '
                      '(2020–2025), from data cleaning to a machine-learning model that estimates transaction prices.',
                      icon='home')

if ov is None:
    th.missing('The EDA summary tables', 'outputs/tables/*_summary.csv', 'Run `python src/EDA.py`.')
    st.stop()

final_row = None
if sel is not None and test is not None:
    match = test[test['Model'] == sel['selected_model']]
    final_row = match.iloc[0] if not match.empty else None

items = [
    {'label': 'Total transactions', 'value': count(ov['transactions']), 'sub': 'single-property sales after cleaning'},
    {'label': 'Study period', 'value': f"{ov['first_year']}–{ov['last_year']}", 'sub': '24 quarterly MOJ files'},
    {'label': 'Regions', 'value': str(ov['regions']), 'sub': 'administrative regions of Saudi Arabia'},
    {'label': 'Property types', 'value': str(len(ov['property_types'])), 'sub': ' · '.join(ov['property_types'])},
]
th.cards(items)
if final_row is not None:
    th.cards([
        {'label': 'Final model', 'value': '<span style="font-size:.78em">Stacked XGBoost + CatBoost</span>',
         'cls': 'primary', 'sub': 'selected on Validation 2024 only'},
        {'label': 'Test 2025 MAE', 'value': f'<span class="nw">{final_row["MAE"]:,.0f}</span><span class="unit">SAR</span>',
         'sub': 'mean absolute error on unseen 2025 sales'},
        {'label': 'Improvement vs baseline', 'value': f'{final_row["MAE vs baseline"]:.1f}%',
         'sub': 'lower MAE than the historical-median rule (target: 15%)'},
        {'label': 'Median absolute error', 'value': f'<span class="nw">{final_row["Median AE"]:,.0f}</span>'
                                                   '<span class="unit">SAR</span>',
         'sub': 'half of 2025 sales are estimated closer than this'},
    ])

yearly, types = data.summary_table('yearly_market_summary'), data.summary_table('property_type_summary')
charts.show(charts.columns(yearly['year'].astype(str).tolist(), yearly['transaction_count'].tolist(),
                           'Transactions per year', height=320))
t = types.sort_values('transaction_count', ascending=False)
th.note('Property types: ' + ' · '.join(f'{r.property_type} {r.pct_of_total:.1f}%' for r in t.itertuples())
        + ' of transactions. Full breakdown in <b>Dataset Overview</b>.')


th.section('Project workflow')
th.steps([('Data', '24 quarterly MOJ sales files merged'),
          ('Cleaning', 'invalid, bulk, duplicate and suspicious rows removed'),
          ('EDA', 'volume, price, area and regional patterns'),
          ('Features', 'location encodings, time index, seasonality'),
          ('Modeling', 'baseline, XGBoost, CatBoost, MLP, stacking'),
          ('Evaluation', 'Model selection on 2024 validation; final evaluation on unseen 2025 sales'),
          ('Prediction', 'interactive estimate with SHAP explanation')])

th.section('Explore the project')
cols = st.columns(3, gap='medium')
links = [('views/eda.py', 'Explore the market', ':material/insights:',
          'Interactive charts of transaction volume, prices, area and price per m² by region and property type.'),
         ('views/performance.py', 'View model performance', ':material/leaderboard:',
          'Compare the baseline and every model on MAE, RMSE and R², and see what drives the predictions.'),
         ('views/prediction.py', 'Predict a property price', ':material/calculate:',
          'Pick a location, property type,area and date and get the final model’s estimate with an explanation.')]
for col, (page, label, icon, body) in zip(cols, links):
    with col:
        st.markdown(f'<div class="card"><h4>{esc(label)}</h4><p>{esc(body)}</p></div>', unsafe_allow_html=True)
        st.page_link(page, label=f'Open: {label}', icon=icon, width='stretch')
