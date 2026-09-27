"""Visual identity: Saudi green on white, sand accent. Colors live here and in .streamlit/config.toml."""
import base64
from pathlib import Path

import streamlit as st

from dashboard.fmt import esc

GREEN, GREEN_DARK, GREEN_TINT = '#006C35', '#004D26', '#EAF3ED'
SAND, SAND_LIGHT = '#B7791F', '#F6F1E7'
BLUE = '#3F6FB5'
INK, INK2, MUTED = '#1B2A22', '#52514E', '#8A8F8B'
LINE, GRID = '#E3E8E4', '#EEF1EE'
SURFACE = '#FFFFFF'
FONT = "'IBM Plex Sans Arabic', 'Segoe UI', Tahoma, sans-serif"

# Fixed identity colors (validated for color-vision deficiency, all pairs ΔE >= 8).
TYPE_COLORS = {'Residential': GREEN, 'Commercial': SAND, 'Agricultural': BLUE}
PROPERTY_TYPES = list(TYPE_COLORS)

OUTLINE_SVG = (Path(__file__).resolve().parents[1] / 'assets' / 'saudi_outline.svg').read_text(encoding='utf-8')

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
html, body, p, li, label, input, textarea, button, h1, h2, h3, h4, td, th,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-baseweb="select"], [data-baseweb="popover"] {{ font-family: {FONT}; }}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: {SURFACE}; }}
.block-container {{ padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1200px; }}
h1, h2, h3, h4 {{ color: {INK}; }}

/* Landing hero with the outline map */
.sa-hero {{ position: relative; overflow: hidden; border-radius: 18px; padding: 30px 32px;
  background: linear-gradient(120deg, {GREEN_DARK} 0%, {GREEN} 70%, #0b7a41 100%); color: #fff;
  display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 20px; }}
.sa-hero::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 5px; background: {SAND}; opacity: .85; }}
.sa-hero .eyebrow {{ font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; color: rgba(255,255,255,.75); font-weight: 600; }}
.sa-hero h1 {{ margin: 6px 0 0; padding: 0; color: #fff; font-size: clamp(1.5rem, 2.8vw, 2.2rem); font-weight: 700; line-height: 1.25; }}
.sa-hero .sub {{ margin-top: 10px; font-size: 1rem; color: rgba(255,255,255,.9); line-height: 1.65; max-width: 720px; }}
.sa-hero .sub bdi {{ font-weight: 600; color: #fff; }}
.sa-map {{ flex: 0 0 auto; width: 150px; }}
.sa-map img {{ width: 100%; height: auto; display: block; opacity: .6; }}

/* Tinted banner header at the top of every page (render_page_header) */
.sa-banner {{ background: {GREEN_TINT}; border: 1px solid #CFE3D6; border-radius: 12px;
  padding: 1rem 1.4rem; margin: 0 0 1.5rem; }}
.sa-banner .ttl {{ display: flex; align-items: center; gap: .55rem; color: {GREEN}; }}
.sa-banner .ttl .material-symbols-rounded {{ font-size: 20px; line-height: 1; font-weight: 400; font-style: normal;
  color: {GREEN}; flex: 0 0 auto; }}
.sa-banner h1 {{ margin: 0; padding: 0; color: {GREEN}; font-size: clamp(1.3rem, 2.2vw, 1.7rem); font-weight: 700;
  line-height: 1.3; }}
.sa-banner .sub {{ margin-top: .35rem; color: {INK2}; font-weight: 400; font-size: .97rem; line-height: 1.55;
  max-width: 860px; }}

/* KPI cards: full values, never truncated */
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 4px 0 12px; }}
.kpi {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 16px 18px; min-width: 0;
  box-shadow: 0 1px 2px rgba(27,42,34,.04); }}
.kpi.primary {{ border-color: {GREEN}; box-shadow: inset 4px 0 0 {GREEN}, 0 1px 2px rgba(27,42,34,.04); }}
.kpi.actual {{ border-color: {SAND}; box-shadow: inset 4px 0 0 {SAND}, 0 1px 2px rgba(27,42,34,.04); }}
.kpi-label {{ font-size: .85rem; color: {INK2}; font-weight: 500; display: flex; align-items: center; gap: 6px; }}
.kpi-help {{ cursor: help; color: {SAND}; font-size: .9rem; }}
.kpi-value {{ margin-top: 6px; color: {INK}; font-weight: 700; font-size: clamp(1.25rem, 1.9vw, 1.6rem);
  line-height: 1.3; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
.kpi.primary .kpi-value {{ color: {GREEN_DARK}; }}
.kpi-value .nw {{ white-space: nowrap; }}
.kpi-value .unit {{ font-size: .62em; font-weight: 600; color: {INK2}; margin-left: 5px; letter-spacing: .03em; white-space: nowrap; }}
.kpi-sub {{ margin-top: 4px; font-size: .8rem; color: {MUTED}; }}
/* Uniform cards: identical title/value sizes, padding and height across the row */
.kpi-grid.uniform {{ grid-template-columns: repeat(auto-fit, minmax(235px, 1fr)); }}
.kpi-grid.uniform .kpi {{ display: flex; flex-direction: column; justify-content: flex-start; min-height: 118px; padding: 16px 18px; }}
.kpi-grid.uniform .kpi-label {{ font-size: .85rem; font-weight: 500; }}
.kpi-grid.uniform .kpi-value {{ font-size: 1.5rem; font-weight: 700; white-space: nowrap; overflow-wrap: normal; }}
.kpi-grid.uniform .kpi-sub {{ margin-top: auto; padding-top: 4px; font-size: .8rem; }}
.split-badge {{ display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  padding: 2px 9px; border-radius: 999px; background: {GREEN_TINT}; color: {GREEN_DARK}; border: 1px solid #CFE3D6; }}
.split-badge.val {{ background: {SAND_LIGHT}; color: #7A5214; border-color: #E6D6B8; }}
.split-badge.test {{ background: #EDF2FA; color: #2C5190; border-color: #D3DEF0; }}
.diff-pct {{ display: block; margin-top: 2px; font-size: .58em; font-weight: 600; color: {INK2}; }}

/* Large prediction result */
.result {{ border-radius: 16px; padding: 22px 24px; background: {GREEN_TINT}; border: 1px solid #CFE3D6; }}
.result .label {{ font-size: .9rem; color: {GREEN_DARK}; font-weight: 600; letter-spacing: .02em; }}
.result .value {{ margin-top: 6px; font-size: clamp(1.9rem, 3.6vw, 2.6rem); font-weight: 700; color: {GREEN_DARK};
  font-variant-numeric: tabular-nums; line-height: 1.15; }}
.result .value .unit {{ font-size: .45em; margin-left: 8px; color: {INK2}; }}
.result .meta {{ margin-top: 8px; color: {INK2}; font-size: .9rem; line-height: 1.55; }}
.result .meta bdi {{ font-weight: 600; color: {INK}; }}

/* Section headings */
.sa-section {{ display: flex; align-items: center; gap: 10px; margin: 26px 0 8px; }}
.sa-section::before {{ content: ""; width: 6px; height: 22px; border-radius: 3px; background: {GREEN}; flex: 0 0 auto; }}
.sa-section h3 {{ margin: 0; padding: 0; font-size: 1.22rem; }}
.sa-lead {{ color: {INK2}; font-size: .93rem; margin: -2px 0 10px; line-height: 1.6; }}
.sa-note {{ color: {INK2}; font-size: .87rem; margin: 2px 0 10px; line-height: 1.55; }}

/* Callout cards (navigation, explanations) */
.card {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 16px 18px; height: 100%; }}
.card h4 {{ margin: 0 0 6px; font-size: 1.02rem; color: {INK}; }}
.card p {{ margin: 0; color: {INK2}; font-size: .9rem; line-height: 1.6; }}
.card .tag {{ display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  color: {GREEN}; margin-bottom: 6px; }}

/* Pipeline stepper */
.steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 6px 0 14px; }}
.step {{ border: 1px solid {LINE}; border-radius: 12px; padding: 12px 14px; background: #fff; }}
.step .n {{ display: inline-flex; width: 24px; height: 24px; border-radius: 12px; background: {GREEN}; color: #fff;
  font-size: .78rem; font-weight: 700; align-items: center; justify-content: center; }}
.step .t {{ margin-top: 8px; font-weight: 600; color: {INK}; font-size: .92rem; line-height: 1.35; }}
.step .d {{ margin-top: 4px; color: {INK2}; font-size: .8rem; line-height: 1.45; }}

/* Chips */
.chip {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600;
  background: {GREEN_TINT}; color: {GREEN_DARK}; margin: 0 6px 6px 0; border: 1px solid #CFE3D6; }}
.chip.neutral {{ background: #F4F5F4; color: {INK2}; border-color: {LINE}; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: #F5F8F6; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] label p {{ font-weight: 600; color: {INK}; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: {GREEN_TINT}; }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: {GREEN_DARK}; font-weight: 600; }}
.sb-brand {{ font-weight: 700; color: {GREEN_DARK}; font-size: 1rem; line-height: 1.3; }}
.sb-brand small {{ display: block; color: {INK2}; font-weight: 500; font-size: .78rem; margin-top: 2px; }}
.sb-team {{ margin-top: 14px; padding-top: 12px; border-top: 1px solid {LINE}; }}
.sb-team .lbl {{ display: flex; align-items: center; gap: 6px; font-size: .7rem; font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase; color: {GREEN}; margin-bottom: 8px; }}
.sb-team .lbl .material-symbols-rounded {{ font-size: 16px; line-height: 1; font-weight: 400; }}
.sb-team .members {{ display: flex; flex-direction: column; gap: 9px; }}
.sb-team .member {{ display: flex; align-items: center; gap: 10px; }}
.sb-team .avatar {{ flex: 0 0 auto; width: 32px; height: 32px; border-radius: 50%; background: {GREEN_TINT};
  border: 1px solid #CFE3D6; color: {GREEN_DARK}; font-family: {FONT}; font-size: .72rem; font-weight: 700;
  letter-spacing: .02em; line-height: 1; display: inline-flex; align-items: center; justify-content: center; }}
/* Same font, size and weight as the brand subtitle (.sb-brand small) */
.sb-team .name {{ color: {INK}; font-family: {FONT}; font-size: .78rem; font-weight: 500; line-height: 1.3; }}
[data-baseweb="select"] div, [data-baseweb="popover"] li {{ unicode-bidi: plaintext; }}
div[data-testid="stAlert"] {{ border-radius: 12px; }}
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 10px; }}
[data-testid="stPageLink"] a {{ border: 1px solid #CFE3D6; background: {GREEN_TINT}; border-radius: 10px; }}
[data-testid="stPageLink"] a:hover {{ background: #DCEBE1; }}

@media (max-width: 640px) {{
  .sa-hero {{ padding: 22px 18px; }}
  .sa-map {{ width: 84px; }}
}}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle_html, eyebrow=''):
    # Embedded as an image: Streamlit strips inline <svg> from markdown.
    outline_uri = 'data:image/svg+xml;base64,' + base64.b64encode(
        OUTLINE_SVG.replace('currentColor', '#FFFFFF').encode('utf-8')).decode('ascii')
    eb = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ''
    st.markdown(
        f'<div class="sa-hero"><div>{eb}<h1>{title}</h1><div class="sub">{subtitle_html}</div></div>'
        f'<div class="sa-map"><img src="{outline_uri}" alt="Outline map of Saudi Arabia"></div></div>',
        unsafe_allow_html=True)


def render_page_header(title, subtitle, icon):
    """Tinted banner at the top of a page: accent icon + title on one line, one-sentence subtitle below.

    icon: a Material Symbols name without the ':material/' prefix (e.g. 'home'), matching the sidebar icons.
    subtitle may contain inline HTML such as <bdi>.
    """
    st.markdown(f'<div class="sa-banner"><div class="ttl"><span class="material-symbols-rounded" aria-hidden="true">'
                f'{esc(icon)}</span><h1>{esc(title)}</h1></div><div class="sub">{subtitle}</div></div>',
                unsafe_allow_html=True)


def section(title, lead=None):
    st.markdown(f'<div class="sa-section"><h3>{title}</h3></div>', unsafe_allow_html=True)
    if lead:
        st.markdown(f'<div class="sa-lead">{lead}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="sa-note">{text}</div>', unsafe_allow_html=True)


def cards(items, uniform=False):
    """items: dicts with label, value (HTML), and optional cls, help, sub.

    uniform=True gives every card the same title/value size, padding and height.
    """
    parts = []
    for it in items:
        help_html = (f' <span class="kpi-help" title="{esc(it["help"])}">ⓘ</span>' if it.get('help') else '')
        sub = f'<div class="kpi-sub">{it["sub"]}</div>' if it.get('sub') else ''
        parts.append(f'<div class="kpi {it.get("cls", "")}"><div class="kpi-label">{it["label"]}{help_html}</div>'
                     f'<div class="kpi-value">{it["value"]}</div>{sub}</div>')
    grid = 'kpi-grid uniform' if uniform else 'kpi-grid'
    st.markdown(f'<div class="{grid}">' + ''.join(parts) + '</div>', unsafe_allow_html=True)


def info_cards(items, columns=3):
    """items: (tag, title, body_html). Rendered as equal-height cards in a responsive grid."""
    cols = st.columns(columns)
    for i, (tag, title, body) in enumerate(items):
        tag_html = f'<div class="tag">{tag}</div>' if tag else ''
        cols[i % columns].markdown(f'<div class="card">{tag_html}<h4>{title}</h4><p>{body}</p></div>',
                                   unsafe_allow_html=True)


def steps(items):
    """items: (title, description). Numbered pipeline stepper."""
    html_parts = [f'<div class="step"><span class="n">{i}</span><div class="t">{t}</div><div class="d">{d}</div></div>'
                  for i, (t, d) in enumerate(items, 1)]
    st.markdown('<div class="steps">' + ''.join(html_parts) + '</div>', unsafe_allow_html=True)


def chips(values, neutral=False):
    cls = 'chip neutral' if neutral else 'chip'
    st.markdown(''.join(f'<span class="{cls}">{esc(v)}</span>' for v in values), unsafe_allow_html=True)


def missing(what, path, how):
    st.warning(f'**{what} is not available.** Expected file: `{path}`. {how}', icon=':material/folder_off:')
