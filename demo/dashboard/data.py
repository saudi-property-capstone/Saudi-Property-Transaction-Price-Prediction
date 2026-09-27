"""Cached, read-only access to the cleaned dataset and the saved EDA tables.

The cleaned dataset (~1.39M rows) is read once per server process with only the
columns the dashboard needs; every chart receives a small aggregate, never raw rows.
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

from src.data_io import PROCESSED_PATH
from src.modeling import config as cfg

TABLES = cfg.INPUT_TABLES_DIR
FIGURES = cfg.OUTPUTS_DIR / 'figures'
TRAIN_YEARS = (2020, 2023)          # chronological split, see src/prepare_features.py

COLUMNS = ['reference_number', 'transaction_date', 'year', 'quarter', 'region_en', 'city', 'city_district',
           'property_type', 'area', 'price', 'price_per_m2_calculated']
CATEGORICAL = ['region_en', 'city', 'city_district', 'property_type']


# ---------- saved EDA outputs (small, tracked in Git) ----------

@st.cache_data(show_spinner=False)
def summary_table(name):
    path = TABLES / f'{name}.csv'
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def split_report():
    path = TABLES / 'feature_split_report.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


@st.cache_data(show_spinner=False)
def overview():
    """Headline facts from the saved EDA tables, so the Home page works without the full dataset."""
    yearly, regional, types = summary_table('yearly_market_summary'), summary_table('regional_summary'), \
        summary_table('property_type_summary')
    if yearly is None or regional is None or types is None:
        return None
    return {'transactions': int(yearly['transaction_count'].sum()),
            'first_year': int(yearly['year'].min()), 'last_year': int(yearly['year'].max()),
            'regions': int(regional['region'].nunique()),
            'property_types': types.sort_values('transaction_count', ascending=False)['property_type'].tolist(),
            'total_value': float(yearly['total_value'].sum())}


# ---------- the cleaned dataset ----------

def dataset_available():
    return PROCESSED_PATH.exists()


@st.cache_resource(show_spinner='Loading the cleaned dataset (1.39M transactions)…')
def dataset():
    """Cleaned MOJ transactions, needed columns only. Shared read-only across sessions: never mutate it."""
    if not dataset_available():
        return None
    df = pd.read_csv(PROCESSED_PATH, usecols=COLUMNS,
                     dtype={**{c: 'category' for c in CATEGORICAL}, 'transaction_date': 'string'})
    return df.rename(columns={'region_en': 'region'})


@st.cache_data(show_spinner=False)
def dataset_facts():
    df = dataset()
    return {'rows': len(df), 'columns_in_file': len(pd.read_csv(PROCESSED_PATH, nrows=0).columns),
            'date_min': str(df['transaction_date'].min()), 'date_max': str(df['transaction_date'].max()),
            'regions': df['region'].nunique(), 'cities': df['city'].nunique(),
            'districts': df['city_district'].nunique(),
            'types': df['property_type'].value_counts().to_dict()}


@st.cache_data(show_spinner=False)
def sample_rows(n=12, seed=42):
    df = dataset()
    out = df.sample(n, random_state=seed).sort_values('transaction_date')
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def quality_checks():
    """Checks computed on the final cleaned file (what remains after cleaning, not the removed counts)."""
    df = dataset()
    removed = summary_table('suspicious_transactions_removed')
    outliers = summary_table('outlier_inspection')
    issues = summary_table('feature_target_issues')
    return {
        'missing_cells': int(df.isna().sum().sum()),
        'duplicate_references': int(df['reference_number'].duplicated().sum()),
        'non_positive_price_or_area': int(((df['price'] <= 0) | (df['area'] <= 0)).sum()),
        'suspicious_removed': None if removed is None else len(removed),
        'suspicious_groups': None if removed is None else
        removed.groupby(['price', 'transaction_date', 'region_en', 'city']).ngroups,
        'outliers_flagged': None if outliers is None else outliers['flagged_as'].value_counts().to_dict(),
        'feature_target_issues': None if issues is None else len(issues),
    }


# ---------- EDA aggregates (keyed by filters; each result is a few hundred rows at most) ----------

def _filter(regions, types, years):
    df = dataset()
    mask = df['year'].between(*years)
    if regions:
        mask &= df['region'].isin(regions)
    if types:
        mask &= df['property_type'].isin(types)
    return df[mask]


def _log_hist(values, lo, hi, step=0.1):
    edges = np.arange(lo, hi + step, step)
    counts, edges = np.histogram(np.log10(values[values > 0]), bins=edges)
    return pd.DataFrame({'x': (edges[:-1] + edges[1:]) / 2, 'lo': 10 ** edges[:-1], 'hi': 10 ** edges[1:],
                         'count': counts})


@st.cache_data(show_spinner='Aggregating transactions…')
def eda_aggregates(regions, types, years):
    """All EDA chart inputs for one filter selection (tuples, so the cache key is hashable)."""
    d = _filter(regions, types, years)
    if d.empty:
        return None
    by = lambda cols: d.groupby(cols, observed=True)  # noqa: E731
    out = {'n': len(d), 'median_price': float(d['price'].median()), 'median_area': float(d['area'].median()),
           'median_ppm2': float(d['price_per_m2_calculated'].median())}

    out['by_year'] = by('year').agg(transactions=('price', 'size'), median_price=('price', 'median'),
                                    median_ppm2=('price_per_m2_calculated', 'median')).reset_index()
    q = by(['year', 'quarter']).agg(transactions=('price', 'size'), median_price=('price', 'median')).reset_index()
    q['period'] = q['year'].astype(str) + '-Q' + q['quarter'].astype(str)
    out['by_quarter'] = q
    out['by_year_type'] = by(['year', 'property_type']).size().rename('transactions').reset_index()
    out['by_region'] = by('region').agg(transactions=('price', 'size'), median_price=('price', 'median'),
                                        median_ppm2=('price_per_m2_calculated', 'median')).reset_index()
    out['by_type'] = by('property_type').agg(transactions=('price', 'size'), median_price=('price', 'median'),
                                             median_area=('area', 'median'),
                                             median_ppm2=('price_per_m2_calculated', 'median')).reset_index()
    cities = by(['city', 'region']).agg(transactions=('price', 'size'), median_price=('price', 'median')).reset_index()
    out['top_cities'] = cities.nlargest(15, 'transactions')

    out['price_hist'] = _log_hist(d['price'].to_numpy(), 3, 9.5)
    out['area_hist'] = _log_hist(d['area'].to_numpy(), 0, 7.5)

    # Area vs price as a 2D density on log-log bins (aggregated, instead of 1M scatter points).
    lp, la = np.log10(d['price'].to_numpy()), np.log10(d['area'].to_numpy())
    h, pe, ae = np.histogram2d(lp, la, bins=[np.arange(3.5, 9.01, 0.1), np.arange(1, 6.51, 0.1)])
    out['density'] = {'z': h, 'price_centers': (pe[:-1] + pe[1:]) / 2, 'area_centers': (ae[:-1] + ae[1:]) / 2}
    area_bin = pd.cut(la, bins=np.arange(1, 6.51, 0.25))
    med = pd.DataFrame({'bin': area_bin, 'lp': lp, 'type': d['property_type'].to_numpy()})
    med = med.groupby(['type', 'bin'], observed=True)['lp'].agg(['median', 'size']).reset_index()
    med = med[med['size'] >= 50]
    med['area_center'] = med['bin'].map(lambda b: (b.left + b.right) / 2).astype(float)
    out['area_price_median'] = med.drop(columns='bin')
    return out


def filter_options():
    df = dataset()
    regions = df['region'].value_counts().index.tolist()
    return regions, (int(df['year'].min()), int(df['year'].max()))


# ---------- historical context for the prediction page (training years only) ----------

@st.cache_data(show_spinner=False)
def training_area_range():
    """1st-99th percentile of area per property type in Training 2020-2023: used only to warn about unusual inputs."""
    df = dataset()
    t = df[df['year'].between(*TRAIN_YEARS)]
    return t.groupby('property_type', observed=True)['area'].quantile([0.01, 0.99]).unstack().to_dict('index')


@st.cache_data(show_spinner=False)
def district_history(region, city, district, property_type):
    """Actual 2020-2023 sales of this property type in this district (descriptive statistics, not the model)."""
    df = dataset()
    m = ((df['city_district'] == district) & (df['property_type'] == property_type) & (df['city'] == city)
         & (df['region'] == region) & df['year'].between(*TRAIN_YEARS))
    s = df.loc[m, ['price', 'area', 'price_per_m2_calculated']]
    if s.empty:
        return None
    return {'n': len(s), 'p25': float(s['price'].quantile(0.25)), 'median': float(s['price'].median()),
            'p75': float(s['price'].quantile(0.75)), 'median_area': float(s['area'].median()),
            'median_ppm2': float(s['price_per_m2_calculated'].median())}
