"""Number formatting for user-facing values: full amounts with separators, or compact K/M/B."""
import html
from collections import Counter

import numpy as np


def sar(v):
    """1450000 -> '1,450,000 SAR'."""
    return f'{float(v):,.0f} SAR'


def compact(v, digits=1):
    """1450000 -> '1.5M'; keeps short values readable without scientific notation."""
    v = float(v)
    for size, suffix in ((1e9, 'B'), (1e6, 'M'), (1e3, 'K')):
        if abs(v) >= size:
            text = f'{v / size:,.{digits}f}'
            if '.' in text:  # strip only decimal zeros: '1.50' -> '1.5', never '10' -> '1'
                text = text.rstrip('0').rstrip('.')
            return f'{text}{suffix}'
    return f'{v:,.0f}'


def compact_sar(v, digits=1):
    return f'{compact(v, digits)} SAR'


def count(v):
    return f'{int(v):,}'


def pct(v, digits=1):
    return f'{float(v):.{digits}f}%'


def esc(v):
    return html.escape(str(v))


def amount_html(v):
    """Full amount with thousands separators and a small SAR label, never truncated."""
    return f'<span class="nw">{float(v):,.0f}</span><span class="unit">SAR</span>'


def district_label(d):
    """'الرياض/النرجس' -> 'النرجس' (city prefix removed for display)."""
    return d.split('/', 1)[-1].strip() or d


def district_labels(districts):
    """Unique display label per district: 'الرياض/الخير' -> 'الخير (الرياض)' when another district is also 'الخير'.

    st.selectbox maps the chosen label back to an option by label, so duplicate labels select the wrong district.
    """
    short = Counter(district_label(d) for d in districts)
    labels = {d: district_label(d) if short[district_label(d)] == 1 else
              f'{district_label(d)} ({d.split("/", 1)[0].strip()})' for d in districts}
    seen = Counter(labels.values())
    return {d: d if seen[lab] > 1 else lab for d, lab in labels.items()}


def log_ticks(lo, hi, money=True):
    """Major ticks (powers of ten, in log10 units) with unique labels between lo and hi: 1K, 10K, 100K, 1M…

    `money` is kept for call-site compatibility; SAR and m² axes use the same compact labels.
    """
    vals = [k for k in range(int(np.ceil(lo - 0.05)), int(np.floor(hi + 0.05)) + 1)]
    return vals, [compact(10 ** k, 0) for k in vals]
