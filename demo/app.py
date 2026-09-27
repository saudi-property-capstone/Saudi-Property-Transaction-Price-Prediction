"""
Saudi Property Transaction Price Prediction - multi-page Streamlit dashboard.

Run from the repository root:
    streamlit run demo/app.py

Pages live in demo/views/, shared helpers in demo/dashboard/. Every page reads saved
project outputs; nothing is trained or tuned here. Prediction uses the frozen final
model through src/demo/predictor.py (unchanged).
Visual theme: .streamlit/config.toml (colors) + demo/dashboard/theme.py (layout, typography).
"""
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
for p in (ROOT, DEMO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st

from dashboard import theme

st.set_page_config(page_title='Saudi Property Price Prediction', page_icon=':material/apartment:', layout='wide')
theme.apply()

VIEWS = DEMO / 'views'
PAGES = [
    st.Page(VIEWS / 'home.py', title='Home', icon=':material/home:', default=True),
    st.Page(VIEWS / 'dataset.py', title='Dataset Overview', icon=':material/table_chart:'),
    st.Page(VIEWS / 'eda.py', title='Exploratory Analysis', icon=':material/insights:'),
    st.Page(VIEWS / 'performance.py', title='Model Performance', icon=':material/leaderboard:'),
    st.Page(VIEWS / 'prediction.py', title='Price Prediction', icon=':material/calculate:'),
    st.Page(VIEWS / 'methodology.py', title='Methodology', icon=':material/account_tree:'),
]

TEAM = ['Haya Alharthi', 'Dana Alrabea', 'Omar Alhejaili', 'Mshari Yehya']

with st.sidebar:
    st.markdown('<div class="sb-brand">Saudi Property Transaction Price Prediction'
                '<small>MOJ sales 2020–2025 · Data Science capstone</small></div>', unsafe_allow_html=True)
    members = ''.join(f'<div class="member"><span class="avatar">{"".join(w[0] for w in name.split()[:2])}</span>'
                      f'<span class="name">{name}</span></div>' for name in TEAM)
    st.markdown('<div class="sb-team"><div class="lbl"><span class="material-symbols-rounded" aria-hidden="true">'
                f'group</span>Project team</div><div class="members">{members}</div></div>', unsafe_allow_html=True)

nav = st.navigation(PAGES)
nav.run()
