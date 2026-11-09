"""
All CSS and HTML cards live here, so app.py only has logic.
"""

import streamlit as st

BG = "#b7a07f"
CARD = "#D9CBB5"
TAB = "#C9B79C"
ACCENT = "#5E4B3C"
ACCENT_DARK = "#7A6350"
INK = "#3D3A4B"
SOFT_INK = "#5A5767"
GENUINE_BG = "#CFE3D6"
FAKE_BG = "#EBCBC6"
FAKE_LINE = "#C47A7A"
GENUINE_LINE = "#5F8F7E"  


# add CSS to the page (call this once at the top of app.py)
def load_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Nunito', sans-serif; }}

    .block-container {{ max-width: 960px; padding-top: 2rem; }}

    #MainMenu, footer, .stAppDeployButton {{ display: none; }}
    
    h1, h2, h3, h4 {{ color: {INK}; font-weight: 700; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
        .stTabs [data-baseweb="tab"] {{
        background: {TAB}; border-radius: 12px; padding: 8px 18px;
        color: {INK};
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background: {CARD}; color: {INK};
    }}
    .stTabs [aria-selected="true"] {{ background: {ACCENT}; color: white; }}
    
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{ display: none; }}

    .stTextArea textarea {{
        background: {CARD}; color: {INK};
        border-radius: 14px; border: 1px solid #A89572;
    }}

    /* buttons */
    .stButton > button {{
        background: {ACCENT}; color: white; border: none;
        border-radius: 12px; padding: 0.6rem 1.6rem; font-weight: 600;
    }}
    .stButton > button:hover {{ background: {ACCENT_DARK}; color: white; }}

    .hero {{
        background: {CARD}; border-radius: 20px;
        padding: 28px 32px; margin-bottom: 24px;
    }}
    .hero h1 {{ margin: 0; font-size: 2rem; }}
    .hero p {{ margin: 6px 0 0 0; color: {SOFT_INK}; }}

    .stat-card {{
        background: {CARD}; border-radius: 16px; padding: 16px 18px;
        text-align: center;
    }}
    .stat-card .value {{ font-size: 1.6rem; font-weight: 700; color: {INK}; }}
    .stat-card .label {{ font-size: 0.85rem; color: {SOFT_INK}; }}

    .result-card {{ border-radius: 16px; padding: 20px 24px; margin-top: 16px; }}
    .result-genuine {{ background: {GENUINE_BG}; border-left: 6px solid {GENUINE_LINE}; }}
    .result-fake {{ background: {FAKE_BG}; border-left: 6px solid {FAKE_LINE}; }}
    .result-card h3 {{ margin: 0 0 6px 0; }}
    .result-card p {{ margin: 0; color: {SOFT_INK}; }}
    
    /* success box ("Best model...")*/
    div[data-testid="stAlert"] {{
        background: {CARD}; color: {INK}; border-radius: 14px;
    }}
    div[data-testid="stAlert"] p {{ color: {INK}; }}

    /* dropdown*/
    div[data-baseweb="select"] > div {{
        background: {CARD}; color: {INK}; border-radius: 12px;
    }}
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {{ color: {INK}; }}

    /* misclassified articles */
    div[data-testid="stExpander"] {{ background: {CARD}; border-radius: 12px; }}
    </style>
    """, unsafe_allow_html=True)

# top banner
def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)

# small card 
def stat_card(label, value):
    st.markdown(f'<div class="stat-card"><div class="value">{value}</div>'
                f'<div class="label">{label}</div></div>', unsafe_allow_html=True)

# prediction result card; kind is "genuine" or "fake"
def result_card(kind, title, message):
    st.markdown(f'<div class="result-card result-{kind}"><h3>{title}</h3>'
                f'<p>{message}</p></div>', unsafe_allow_html=True)