"""Dataset Overview: what the cleaned data contains, a small sample, feature roles, and data quality."""
import pandas as pd
import streamlit as st

from dashboard import charts, data
from dashboard import theme as th
from dashboard.fmt import count

th.render_page_header('Dataset Overview',
                      'The 24 quarterly MOJ sales files (2020 Q1 – 2025 Q4), merged and cleaned into one table of '
                      'single-property transactions.',
                      icon='table_chart')

if not data.dataset_available():
    th.missing('The cleaned dataset', 'data/processed/moj_sales_2020_2025_merged.csv.gz',
               'Run `python src/data_cleaning.py` (see README, section 1).')
    st.stop()

f = data.dataset_facts()
split = data.split_report()

# ---------------- Summary ----------------
th.section('Dataset summary')
d0, d1 = pd.Timestamp(f['date_min']), pd.Timestamp(f['date_max'])
th.cards([
    {'label': 'Rows (transactions)', 'value': count(f['rows']), 'sub': 'single-property sales'},
    {'label': 'Columns', 'value': str(f['columns_in_file']), 'sub': 'in the cleaned file'},
    {'label': 'Date range', 'value': f'{d0.year}–{d1.year}', 'sub': f'{d0:%b %Y} – {d1:%b %Y}'},
    {'label': 'Regions · cities · districts',
     'value': f"{f['regions']} · {f['cities']} · {count(f['districts'])}", 'sub': 'distinct locations'},
], uniform=True)

if split:
    s = split['splits']
    th.cards([
        {'label': '<span class="split-badge">Train</span>', 'value': count(s['train']['retained_rows']),
         'sub': '2020–2023 · model fitting'},
        {'label': '<span class="split-badge val">Validation</span>', 'value': count(s['validation']['retained_rows']),
         'sub': '2024 · model selection'},
        {'label': '<span class="split-badge test">Test</span>', 'value': count(s['test']['retained_rows']),
         'sub': '2025 · final held-out evaluation'},
    ], uniform=True)

th.section('Property types', 'Transactions per property type in the cleaned dataset.')
types = pd.Series(f['types']).sort_values(ascending=False)
fig = charts.hbars(types.index.tolist(), types.tolist(), 'Transactions by property type',
                   colors=[th.TYPE_COLORS.get(t, th.GREEN) for t in types.index],
                   share=(types / f['rows'] * 100).tolist(), height=250)
asc = types.sort_values()  # hbars draws bars in ascending order
fig.update_traces(text=[f'{n:,} · {n / f["rows"] * 100:.1f}%' for n in asc])
charts.show(fig)

# ---------------- Sample ----------------
th.section('Data sample', 'Twelve randomly drawn rows (fixed seed) of the cleaned dataset. The full table is not '
                          'rendered in the browser.')
sample = data.sample_rows()[['transaction_date', 'year', 'quarter', 'region', 'city', 'city_district', 'property_type',
                             'area', 'price', 'price_per_m2_calculated', 'reference_number']].copy()
sample['price_per_m2_calculated'] = sample['price_per_m2_calculated'].round(0)
sample = sample.rename(columns={
    'reference_number': 'Reference', 'transaction_date': 'Date', 'year': 'Year', 'quarter': 'Quarter',
    'region': 'Region', 'city': 'City', 'city_district': 'District', 'property_type': 'Property type',
    'area': 'Area (m²)', 'price': 'Price (SAR)', 'price_per_m2_calculated': 'Price per m² (SAR)'})
st.dataframe(sample, hide_index=True, width='stretch', column_config={
    'Reference': st.column_config.NumberColumn(format='%d'),
    'Year': st.column_config.NumberColumn(format='%d'),
    'Area (m²)': st.column_config.NumberColumn(format='localized'),
    'Price (SAR)': st.column_config.NumberColumn(format='localized'),
    'Price per m² (SAR)': st.column_config.NumberColumn(format='localized'),
})

# ---------------- Features ----------------
th.section('Feature overview', 'Columns of the cleaned file and how each one is used.')
features = pd.DataFrame([
    ('region (region_en)', 'Location', 'Administrative region (13), standardised to English', 'Model input (target-encoded for XGBoost, raw for CatBoost)'),
    ('city', 'Location', 'City name (Arabic)', 'Model input (frequency-encoded / native categorical)'),
    ('city_district', 'Location', 'District, written as “city/district”', 'Model input (frequency-encoded / native categorical)'),
    ('property_type', 'Property', 'Residential, Commercial or Agricultural', 'Model input (one-hot / native categorical)'),
    ('area', 'Property', 'Land/property area in m²', 'Model input'),
    ('year, quarter', 'Time', 'Year and quarter of the source file', 'Model input as a quarter time index + sin/cos seasonality'),
    ('transaction_date', 'Time', 'Gregorian transaction date', 'EDA and the date range only'),
    ('price', 'Target', 'Transaction price in SAR', 'Target, modelled as ln(price)'),
    ('price_per_m2_calculated', 'Derived', 'price ÷ area', 'EDA only: excluded from the model (it is computed from the target)'),
    ('reference_number', 'Identifier', 'MOJ transaction reference', 'Duplicate checks and split-overlap checks only'),
    ('region_raw, date_gregorian_raw, n_properties, source_file', 'Provenance', 'Original values kept for traceability',
     'Not used by the model'),
], columns=['Column', 'Group', 'Meaning', 'Role in the project'])
st.dataframe(features, hide_index=True, width='stretch')

# ---------------- Quality ----------------
th.section('Data quality', 'Cleaning rules applied to the raw transaction data and verified on the final cleaned dataset.')
q = data.quality_checks()
c1, c2 = st.columns([1.1, 1], gap='large')
with c1:
    st.markdown('**Rows removed during cleaning (in order)**')
    st.markdown(
        '1. Bulk transactions (`n_properties > 1`): the scope is single-property sales.\n'
        '2. Invalid values: `price ≤ 0` or `area ≤ 0`.\n'
        '3. Missing required fields: price, area, date, region or property type.\n'
        '4. Suspicious repeated high-value transaction groups were reviewed and removed when they matched predefined data-error rules.\n'
        '5. Duplicates: repeated business key or `reference_number` (first copy kept).\n'
        '6. Rare property types (Industrial, Mixed Use, Other): too few rows for reliable analysis.')
   
with c2:
    st.markdown('**Checks on the final cleaned file**')
    checks = [('Missing values in the loaded columns', count(q['missing_cells'])),
              ('Duplicate reference numbers', count(q['duplicate_references'])),
              ('Rows with price ≤ 0 or area ≤ 0', count(q['non_positive_price_or_area']))]
    if q['suspicious_removed'] is not None:
        checks.append(('Suspicious rows removed',
                       f"{count(q['suspicious_removed'])} ({q['suspicious_groups']} groups)"))
    if q['feature_target_issues'] is not None:
        checks.append(('Invalid targets found when building features', count(q['feature_target_issues'])))
    if q['outliers_flagged']:
        flagged = sum(q['outliers_flagged'].values())
        checks.append(('Extreme transactions inspected (kept)', count(flagged)))
    st.dataframe(pd.DataFrame(checks, columns=['Check', 'Result']), hide_index=True, width='stretch')
    th.note('Cleaning rules applied to the raw transaction data and verified on the final cleaned dataset.')

if q['outliers_flagged']:
    with st.expander('Extreme transactions inspected by the outlier analysis'):
        out = data.summary_table('outlier_inspection')
        view = out[['price', 'area', 'property_type', 'region', 'city', 'year', 'quarter', 'flagged_as']].head(40)
        st.dataframe(view, hide_index=True, width='stretch', column_config={
            'price': st.column_config.NumberColumn('Price (SAR)', format='localized'),
            'area': st.column_config.NumberColumn('Area (m²)', format='localized'),
            'year': st.column_config.NumberColumn('Year', format='%d')})
        st.caption(f'Top 40 of {len(out):,} rows sorted by price.')
