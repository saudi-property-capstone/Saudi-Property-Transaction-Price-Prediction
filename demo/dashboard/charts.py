"""Plotly chart builders with one consistent look (all inputs are small aggregates)."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme as th
from dashboard.fmt import compact, log_ticks

CONFIG = {'displayModeBar': False, 'responsive': True}


def style(fig, height=360, title=None, legend=False):
    fig.update_layout(
        height=height, template='plotly_white', paper_bgcolor=th.SURFACE, plot_bgcolor=th.SURFACE,
        font=dict(family="IBM Plex Sans Arabic, Segoe UI, sans-serif", size=13, color=th.INK2),
        title=dict(text=f'<b>{title}</b>' if title else None, font=dict(size=15, color=th.INK), x=0, xanchor='left',
                   y=0.98, yanchor='top'),
        margin=dict(l=8, r=16, t=(88 if legend else 48) if title else (36 if legend else 12), b=8), barcornerradius=4, bargap=0.28,
        hoverlabel=dict(bgcolor='#FFFFFF', bordercolor=th.LINE, font=dict(color=th.INK, size=13)),
        showlegend=legend,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, title=None,
                    font=dict(color=th.INK, size=12)),
        separators='.,')
    fig.update_xaxes(gridcolor=th.GRID, linecolor=th.LINE, zeroline=False, tickfont=dict(color=th.INK2),
                     title_font=dict(size=12, color=th.INK2))
    fig.update_yaxes(gridcolor=th.GRID, linecolor=th.LINE, zeroline=False, tickfont=dict(color=th.INK2),
                     title_font=dict(size=12, color=th.INK2))
    return fig


def show(fig):
    st.plotly_chart(fig, width='stretch', config=CONFIG)


def _money_axis(axis, values):
    """Readable SAR ticks (e.g. 250K, 1.5M) instead of Plotly's SI/scientific defaults."""
    vmax = float(np.nanmax(values)) if len(values) else 1
    step = _nice_step(vmax / 4)
    ticks = np.arange(0, vmax * 1.12 + step, step)
    axis(tickvals=ticks, ticktext=[compact(t) for t in ticks])


def _nice_step(x):
    if x <= 0:
        return 1
    mag = 10 ** np.floor(np.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if x <= m * mag:
            return m * mag
    return 10 * mag


def columns(x, y, title, money=False, color=th.GREEN, height=340, hover_label='Transactions', labels=True,
            x_title=None):
    """Vertical bars (time on x). Direct labels on the bars when few enough to stay readable."""
    text = [compact(v) for v in y] if labels else None
    hover = '%{x}<br>' + hover_label + ': ' + ('%{customdata} SAR' if money else '%{y:,.0f}') + '<extra></extra>'
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=color, text=text, textposition='outside', cliponaxis=False,
                           textfont=dict(color=th.INK2, size=12), customdata=[f'{v:,.0f}' for v in y],
                           hovertemplate=hover))
    style(fig, height, title)
    fig.update_xaxes(type='category', title=x_title)
    fig.update_yaxes(showticklabels=not labels or money, showgrid=not labels)
    if money:
        _money_axis(fig.update_yaxes, y)
    return fig


def hbars(categories, values, title, money=False, colors=None, height=None, value_label='Transactions',
          extra=None, share=None):
    """Horizontal bars sorted by value, largest on top, with direct value labels."""
    order = np.argsort(values)
    cats = [categories[i] for i in order]
    vals = [values[i] for i in order]
    cols = [colors[i] for i in order] if isinstance(colors, list) else (colors or th.GREEN)
    if money:
        text = [f'{compact(v)} SAR' for v in vals]
    elif share is not None:
        shares = [share[i] for i in order]
        text = [f'{compact(v)} · {s:.1f}%' for v, s in zip(vals, shares)]
    else:
        text = [compact(v) for v in vals]
    custom = [f'{v:,.0f}' + (' SAR' if money else '') for v in vals]
    extra_text = [extra[i] for i in order] if extra is not None else [''] * len(vals)
    fig = go.Figure(go.Bar(y=cats, x=vals, orientation='h', marker_color=cols, text=text, textposition='outside',
                           cliponaxis=False, textfont=dict(color=th.INK2, size=12),
                           customdata=np.column_stack([custom, extra_text]),
                           hovertemplate='<b>%{y}</b><br>' + value_label + ': %{customdata[0]}%{customdata[1]}'
                                         '<extra></extra>'))
    style(fig, height or max(220, 34 * len(cats) + 70), title)
    pad = 1.5 if (share is not None or money) else 1.3
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, max(vals) * pad])
    fig.update_yaxes(ticksuffix='  ', automargin=True)
    return fig


def line(x, y, title, money=False, color=th.GREEN, height=340, hover_label='Value', markers=True):
    custom = [f'{v:,.0f}' + (' SAR' if money else '') for v in y]
    fig = go.Figure(go.Scatter(x=x, y=y, mode='lines+markers' if markers else 'lines',
                               line=dict(color=color, width=2.5), marker=dict(size=8, color=color,
                                                                               line=dict(color='#fff', width=2)),
                               customdata=custom, hovertemplate='%{x}<br>' + hover_label + ': %{customdata}'
                                                                                          '<extra></extra>'))
    style(fig, height, title)
    fig.update_layout(hovermode='x')
    if money:
        _money_axis(fig.update_yaxes, y)
    fig.update_yaxes(rangemode='tozero')
    return fig


def multi_line(df, x, y, series, title, colors, height=340, hover_label='Transactions'):
    fig = go.Figure()
    for name, color in colors.items():
        part = df[df[series] == name]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(x=part[x], y=part[y], name=name, mode='lines+markers',
                                 line=dict(color=color, width=2.5),
                                 marker=dict(size=8, color=color, line=dict(color='#fff', width=2)),
                                 hovertemplate=name + ': %{y:,.0f}<extra></extra>'))
    style(fig, height, title, legend=True)
    fig.update_layout(hovermode='x unified')
    fig.update_xaxes(type='category')
    fig.update_yaxes(rangemode='tozero', tickformat=',.0f')
    return fig


def log_histogram(hist, title, money=True, color=th.GREEN, height=330, x_title=None):
    """Histogram computed on log10 bins; ticks read as plain amounts (1K, 10K, 1M…)."""
    h = hist[hist['count'] > 0]
    lo, hi = h['x'].min() - 0.05, h['x'].max() + 0.05
    unit = ' SAR' if money else ' m²'
    custom = np.column_stack([[f'{v:,.0f}' for v in h['lo']], [f'{v:,.0f}' for v in h['hi']]])
    fig = go.Figure(go.Bar(x=h['x'], y=h['count'], width=0.092, marker_color=color, customdata=custom,
                           hovertemplate='%{customdata[0]} – %{customdata[1]}' + unit +
                                         '<br>Transactions: %{y:,.0f}<extra></extra>'))
    style(fig, height, title)
    vals, labels = log_ticks(lo, hi, money)
    fig.update_xaxes(tickvals=vals, ticktext=labels, range=[lo, hi], title=x_title)
    fig.update_yaxes(tickformat=',.0f', title='Transactions')
    return fig


def density(dens, medians, title, height=440):
    """Area vs price as a 2D count heatmap (log-log) with the median price per area bin overlaid."""
    z = dens['z'].astype(float)
    z[z == 0] = np.nan
    fig = go.Figure(go.Heatmap(
        x=dens['area_centers'], y=dens['price_centers'], z=np.log10(z),
        customdata=np.nan_to_num(z).astype(int),
        colorscale=[[0, '#EAF3ED'], [0.5, '#5FA77C'], [1, th.GREEN_DARK]], showscale=False,
        hovertemplate='Area ≈ %{x:.1f} (log₁₀ m²)<br>Price ≈ %{y:.1f} (log₁₀ SAR)<br>'
                      'Transactions: %{customdata:,}<extra></extra>'))
    for name, color in th.TYPE_COLORS.items():
        m = medians[medians['type'] == name]
        if m.empty:
            continue
        fig.add_trace(go.Scatter(
            x=m['area_center'], y=m['median'], name=f'Median price · {name}', mode='lines',
            line=dict(color=color if name != 'Residential' else th.INK, width=2.5),
            customdata=[f'{10 ** v:,.0f}' for v in m['median']],
            hovertemplate=name + ' median: %{customdata} SAR<extra></extra>'))
    style(fig, height, title, legend=True)
    xv, xl = log_ticks(1, 6.5, money=False)
    yv, yl = log_ticks(3.5, 9, money=True)
    fig.update_xaxes(tickvals=xv, ticktext=[f'{t} m²' for t in xl], title='Area (log scale)', range=[1, 6.5])
    fig.update_yaxes(tickvals=yv, ticktext=yl, title='Price, SAR (log scale)', range=[3.5, 9])
    return fig


def heatmap_year_quarter(q, title, height=300):
    pivot = q.pivot(index='year', columns='quarter', values='median_price').sort_index()
    text = [[compact(v) if not np.isnan(v) else '' for v in row] for row in pivot.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(), x=[f'Q{c}' for c in pivot.columns], y=[str(i) for i in pivot.index], text=text,
        texttemplate='%{text}', textfont=dict(size=12),
        colorscale=[[0, '#EAF3ED'], [1, th.GREEN]], showscale=False, xgap=2, ygap=2,
        hovertemplate='%{y} %{x}<br>Median price: %{z:,.0f} SAR<extra></extra>'))
    style(fig, height, title)
    fig.update_yaxes(autorange='reversed', showgrid=False)
    fig.update_xaxes(showgrid=False, side='top')
    return fig
