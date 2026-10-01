"""Price Prediction: the frozen final model through src/demo/predictor.py (prediction logic unchanged)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import charts, data
from dashboard import theme as th
from dashboard.fmt import amount_html, compact, esc, sar, district_label, district_labels
from src.modeling import config as cfg


@st.cache_resource(show_spinner='Loading the final model…')
def get_predictor():
    # Imported here so the page header renders while the model libraries load.
    from src.demo.predictor import PricePredictor
    return PricePredictor()


@st.cache_data(show_spinner=False)
def model_summary():
    test = pd.read_csv(cfg.TUNING_TABLES_DIR / 'final_test_comparison.csv').set_index('Model')
    row = test.loc[cfg.MODEL_LABELS[cfg.STACK_KEY]]
    return row['Test MAE (SAR)'], row['MAE improvement vs test baseline (%)'], row['Median abs error (SAR)']


th.render_page_header('Price Prediction',
                      "Estimate a property's transaction price with the final model, or compare the model with a real "
                      '2025 sale it never saw.',
                      icon='calculate')

try:
    p = get_predictor()
except FileNotFoundError as e:
    th.missing('The trained model files', getattr(e, 'filename', None) or 'outputs/tuning/models/',
               'Model binaries are not tracked in Git; recreate them with the tuning pipeline (README, section 7).')
    st.stop()
from src.demo.predictor import LAST_TRAINING_YEAR  # noqa: E402  (already loaded by get_predictor)

mae, _, med = model_summary()
ANY = 'Any'


def place(property_type, area, district, city, region, year, quarter):
    return (f'{esc(property_type)} · {area:,.0f} m² · <bdi>{esc(district_label(district))}</bdi>, '
            f'<bdi>{esc(city)}</bdi> ({esc(region)}) · {year} Q{quarter}')


def explanation(r):
    th.section('Why this estimate?',
               f"The model starts from its average estimate, <b>{sar(r['base_price'])}</b>, and each characteristic "
               'multiplies it up or down (exact SHAP values of the final model).')
    f = r['factors']
    df = pd.DataFrame({'Characteristic': f.index, 'Factor': f.values}).iloc[::-1]
    colors = [th.GREEN if v >= 1 else th.SAND for v in df['Factor']]
    lim = float(max(0.35, np.abs(np.log10(df['Factor'])).max() * 1.3))
    fig = go.Figure(go.Bar(
        y=df['Characteristic'], x=np.log10(df['Factor']), orientation='h', marker_color=colors,
        text=[f'×{v:.2f}' for v in df['Factor']], textposition='outside', cliponaxis=False,
        textfont=dict(color=th.INK, size=13),
        customdata=np.where(df['Factor'] >= 1, 'raises the price', 'lowers the price'),
        hovertemplate='%{y}: %{text} (%{customdata})<extra></extra>'))
    charts.style(fig, 290)
    ticks = [t for t in (0.25, 0.5, 1, 2, 4) if abs(np.log10(t)) <= lim]
    fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[f'×{t:g}' for t in ticks], range=[-lim, lim],
                     title='Effect on the price (×1 = no effect) · green raises, sand lowers')
    fig.add_vline(x=0, line=dict(color=th.INK2, dash='dash', width=1.2))
    fig.update_yaxes(ticksuffix='  ')
    charts.show(fig)
    st.caption(f"{sar(r['base_price'])} × " + ' × '.join(f'{v:.2f}' for v in f.values) + f" ≈ {sar(r['price'])}")


def area_warning(property_type, area):
    if not data.dataset_available():
        return
    lo, hi = data.training_area_range()[property_type][0.01], data.training_area_range()[property_type][0.99]
    if area < lo or area > hi:
        st.warning(f'{area:,.0f} m² is unusual for a {property_type.lower()} property: 98% of {property_type.lower()} '
                   f'training sales were between {lo:,.0f} and {hi:,.0f} m². The estimate is less reliable outside '
                   'that range.', icon=':material/warning:')


mode = st.segmented_control('Mode', ['Estimate a property', 'Check a real 2025 sale'],
                            default='Estimate a property', label_visibility='collapsed')

if mode != 'Check a real 2025 sale':
    # ================= Estimate a property =================
    left, right = st.columns([1, 1.55], gap='large')
    with left:
        with st.container(border=True):
            st.markdown('**Property details**')
            regions = p.regions()
            region = st.selectbox('Region', regions, index=regions.index('Riyadh') if 'Riyadh' in regions else 0)
            city = st.selectbox('City', p.cities(region), help='Cities with training sales in the selected region.')
            districts = p.districts(region, city)
            district = st.selectbox('District', districts, format_func=district_labels(districts).get,
                                    help='Districts with training sales in the selected city, most active first.')
            property_type = st.radio('Property type', th.PROPERTY_TYPES, horizontal=True)
            area = st.number_input('Area (m²)', min_value=10.0, max_value=5_000_000.0, value=500.0, step=50.0,
                                   format='%.0f')
            c1, c2 = st.columns(2)
            year = c1.selectbox('Year', list(range(2020, 2026)), index=4)
            quarter = c2.selectbox('Quarter', [1, 2, 3, 4], index=3)
            clicked = st.button('Predict price', type='primary', width='stretch', icon=':material/calculate:')

    inputs = (region, city, district, property_type, float(area), int(year), int(quarter))
    if clicked:
        st.session_state['estimate'] = (inputs, p.estimate(*inputs))
    saved = st.session_state.get('estimate')

    with right:
        if saved is None:
            st.markdown('<div class="result"><div class="label">Estimated property price</div>'
                        '<div class="meta">Choose the property details and press <b>Predict price</b>.</div></div>',
                        unsafe_allow_html=True)
        else:
            (region_s, city_s, district_s, type_s, area_s, year_s, quarter_s), r = saved
            if saved[0] != inputs:
                st.caption('The inputs changed. Press **Predict price** to update the estimate.')
            st.markdown(
                f'<div class="result"><div class="label">Estimated property price</div>'
                f'<div class="value">{r["price"]:,.0f}<span class="unit">SAR</span></div>'
                f'<div class="meta">{place(type_s, area_s, district_s, city_s, region_s, year_s, quarter_s)}<br>'
                f'≈ {compact(r["price"], 2)} SAR · {r["price"] / area_s:,.0f} SAR per m² of the estimate</div></div>',
                unsafe_allow_html=True)
            st.write('')
            th.cards([
                {'label': 'Typical range', 'value': f'<span class="nw">{r["low"]:,.0f}</span> – {amount_html(r["high"])}',
                 'help': "Half of comparable 2024 sales had an actual price within this range of the model's estimate "
                         '(from validation errors of the final model; not a confidence interval).'},
            ])
            area_warning(type_s, area_s)

    if saved is not None:
        (region_s, city_s, district_s, type_s, area_s, year_s, quarter_s), r = saved
        if data.dataset_available():
            hist = data.district_history(region_s, city_s, district_s, type_s)
            th.section('Historical sales in this district',
                       f'Actual {type_s.lower()} sales in <bdi>{esc(district_label(district_s))}</bdi> during '
                       f'2020–{LAST_TRAINING_YEAR}. These are descriptive statistics of past sales, not model output.')
            if hist is None:
                th.note('No sales of this property type in this district during the training period. The historical '
                        f'median (baseline) falls back to the {r["historical_source"]}: '
                        f'{r["historical_median"]:,.0f} SAR.')
            else:
                th.cards([
                    {'label': 'Sales recorded', 'value': f'{hist["n"]:,}'},
                    {'label': 'Historical median (baseline)', 'value': amount_html(r['historical_median']),
                     'help': f'Median sale price of these {hist["n"]:,} sales; the simple rule the final model is '
                             'compared against.'},
                    {'label': 'Median area', 'value': f'<span class="nw">{hist["median_area"]:,.0f}</span>'
                                                      '<span class="unit">m²</span>'},
                    {'label': 'Median price per m²', 'value': amount_html(hist['median_ppm2'])},
                ])
        explanation(r)

else:
    # ================= Check a real 2025 sale =================
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1])
        region = c1.selectbox('Region', [ANY] + p.sale_choices('region'))
        f_region = None if region == ANY else region
        f_city = f_district = None
        city = c2.selectbox('City', [ANY] + (p.sale_choices('city', region=f_region) if f_region else []),
                            disabled=f_region is None)
        f_city = None if city == ANY else city
        sale_districts = p.sale_choices('city_district', region=f_region, city=f_city) if f_city else []
        sale_labels = district_labels(sale_districts)
        district = c3.selectbox('District', [ANY] + sale_districts,
                                format_func=lambda d: sale_labels.get(d, d), disabled=f_city is None)
        f_district = None if district == ANY else district
        ptype = c4.selectbox('Property type', [ANY] + th.PROPERTY_TYPES)
        f_ptype = None if ptype == ANY else ptype
        if 'draw' not in st.session_state:
            st.session_state.draw = 0
        b1, b2 = st.columns([1, 3], vertical_alignment='center')
        if b1.button('Show another sale', type='primary', width='stretch', icon=':material/shuffle:'):
            st.session_state.draw += 1
        pool = p.filter_sales(region=f_region, city=f_city, district=f_district, property_type=f_ptype)
        b2.caption(f'{len(pool):,} matching sales in 2025. Each sale is drawn at random from them (not hand-picked), '
                   'so some estimates are close and some are far off.')

    if pool.empty:
        st.info('No 2025 sales match these filters.')
        st.stop()
    sale = pool.iloc[int(np.random.default_rng(st.session_state.draw).integers(len(pool)))]
    r = p.estimate(sale['region'], sale['city'], sale['city_district'], sale['property_type'],
                   sale['area'], sale['year'], sale['quarter'])
    actual, est = float(sale['actual_price']), r['price']
    diff, pct = est - actual, (est - actual) / actual * 100
    direction = 'above' if diff > 0 else 'below'
    th.note('Real 2025 sale (not seen by the model): ' +
            place(sale['property_type'], sale['area'], sale['city_district'], sale['city'], sale['region'],
                  sale['year'], sale['quarter']))
    th.cards([{'label': 'Actual sale price', 'value': amount_html(actual), 'cls': 'actual'},
              {'label': "Model's estimate", 'value': amount_html(est), 'cls': 'primary'},
              {'label': 'Difference', 'value': f'<span class="nw">{abs(diff):,.0f}</span><span class="unit">SAR</span> '
                                               f'<span class="diff-pct">({abs(pct):.1f}% {direction} the actual '
                                               'price)</span>'}])
    th.note(f'For comparison, across all 2025 sales the average error is {mae:,.0f} SAR and half of the sales have an '
            f'error below {med:,.0f} SAR. The model was frozen before any 2025 sale was used.')
    explanation(r)
