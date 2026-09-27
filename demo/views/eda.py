"""Exploratory Data Analysis: interactive, aggregated views of the cleaned transactions."""
import streamlit as st

from dashboard import charts, data
from dashboard import theme as th
from dashboard.fmt import amount_html, compact, count

th.render_page_header('Exploratory Data Analysis',
                      'Transaction volume, prices and areas across time, regions and property types, updated with the '
                      'filters below.',
                      icon='insights')

if not data.dataset_available():
    th.missing('The cleaned dataset', 'data/processed/moj_sales_2020_2025_merged.csv.gz',
               'Run `python src/data_cleaning.py`. The static EDA figures are in `outputs/figures/`.')
    st.stop()

regions_all, (y0, y1) = data.filter_options()
f1, f2, f3 = st.columns([1.4, 1.2, 1])
regions = f1.multiselect('Region', regions_all, placeholder='All regions')
types = f2.multiselect('Property type', th.PROPERTY_TYPES, placeholder='All property types')
years = f3.slider('Years', y0, y1, (y0, y1))

agg = data.eda_aggregates(tuple(sorted(regions)), tuple(sorted(types)), tuple(years))
if agg is None:
    st.info('No transactions match these filters.')
    st.stop()

th.cards([{'label': 'Transactions', 'value': count(agg['n'])},
          {'label': 'Median price', 'value': amount_html(agg['median_price'])},
          {'label': 'Median area', 'value': f'<span class="nw">{agg["median_area"]:,.0f}</span><span class="unit">m²</span>'},
          {'label': 'Median price per m²', 'value': amount_html(agg['median_ppm2']),
           'help': 'Descriptive only: price per m² is derived from the price, so it is never a model input.'}])

t_act, t_type, t_price, t_area, t_ppm2 = st.tabs(['Transaction activity', 'Property types', 'Prices', 'Area',
                                                  'Price per m²'])

with t_act:
    by_year, q = agg['by_year'], agg['by_quarter']
    c1, c2 = st.columns(2, gap='large')
    with c1:
        charts.show(charts.columns(by_year['year'].astype(str).tolist(), by_year['transactions'].tolist(),
                                   'Transactions per year'))
    with c2:
        fig = charts.line(q['period'].tolist(), q['transactions'].tolist(), 'Transactions per quarter',
                          hover_label='Transactions')
        q1 = [p for p in q['period'] if p.endswith('Q1')]
        fig.update_xaxes(type='category', tickvals=q1, ticktext=[p[:4] for p in q1], tickangle=0)
        fig.update_yaxes(tickformat=',.0f')
        charts.show(fig)
    r = agg['by_region']
    charts.show(charts.hbars(r['region'].tolist(), r['transactions'].tolist(), 'Transactions by region',
                             share=(r['transactions'] / r['transactions'].sum() * 100).tolist()))

    c = agg['top_cities']
    charts.show(charts.hbars(c['city'].tolist(), c['transactions'].tolist(), 'Top 15 cities by transactions',
                             extra=[f' · {rg}' for rg in c['region']]))

with t_type:
    bt = agg['by_type']
    colors = [th.TYPE_COLORS[t] for t in bt['property_type']]
    c1, c2 = st.columns(2, gap='large')
    with c1:
        charts.show(charts.hbars(bt['property_type'].tolist(), bt['transactions'].tolist(),
                                 'Transactions by property type', colors=colors,
                                 share=(bt['transactions'] / bt['transactions'].sum() * 100).tolist(), height=300))
    with c2:
        charts.show(charts.hbars(bt['property_type'].tolist(), bt['median_price'].tolist(),
                                 'Median price by property type', money=True, colors=colors,
                                 value_label='Median price', height=300))
    shown = {k: v for k, v in th.TYPE_COLORS.items() if k in set(bt['property_type'])}
    charts.show(charts.multi_line(agg['by_year_type'], 'year', 'transactions', 'property_type',
                                  'Transactions per year by property type', shown))

with t_price:
    c1, c2 = st.columns(2, gap='large')
    with c1:
        charts.show(charts.log_histogram(agg['price_hist'], 'Distribution of transaction prices',
                                         x_title='Price, SAR (log scale)'))
        th.note('Transaction prices are strongly right-skewed, so a logarithmic x-axis is used to make the distribution easier to read.')
    with c2:
        charts.show(charts.line(by_year['year'].astype(str).tolist(), by_year['median_price'].tolist(),
                                'Median price per year', money=True, hover_label='Median price'))
    r = agg['by_region']
    charts.show(charts.hbars(r['region'].tolist(), r['median_price'].tolist(), 'Median price by region', money=True,
                             value_label='Median price', extra=[f' · {n:,} sales' for n in r['transactions']]))
    charts.show(charts.heatmap_year_quarter(q, 'Median price by year and quarter (SAR)'))

with t_area:
    c1, c2 = st.columns([1, 1.25], gap='large')
    with c1:
        charts.show(charts.log_histogram(agg['area_hist'], 'Distribution of property area', money=False,
                                         color=th.BLUE, x_title='Area, m² (log scale)'))
        th.note(f'Median area: {agg["median_area"]:,.0f} m². Agricultural plots stretch the long right tail.')
    with c2:
        charts.show(charts.density(agg['density'], agg['area_price_median'], 'Area vs transaction price'))
        th.note('Shading shows how many transactions fall in each area × price cell (darker = more); lines show the '
                'median price per area band for each property type. Both axes are logarithmic.')

with t_ppm2:
    st.info('Price per m² (price ÷ area) is used for market analysis only. It is derived from the target price, so it '
            'was deliberately excluded from the model inputs.', icon=':material/info:')
    r = agg['by_region']
    c1, c2 = st.columns([1.3, 1], gap='large')
    with c1:
        charts.show(charts.hbars(r['region'].tolist(), r['median_ppm2'].tolist(), 'Median price per m² by region',
                                 money=True, value_label='Median price per m²'))
    with c2:
        bt = agg['by_type']
        charts.show(charts.hbars(bt['property_type'].tolist(), bt['median_ppm2'].tolist(),
                                 'Median price per m² by property type', money=True,
                                 colors=[th.TYPE_COLORS[t] for t in bt['property_type']],
                                 value_label='Median price per m²', height=260))
        charts.show(charts.line(by_year['year'].astype(str).tolist(), by_year['median_ppm2'].tolist(),
                                'Median price per m² per year', money=True, hover_label='Median price per m²',
                                height=300))

