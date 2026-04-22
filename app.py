"""
app.py - University Knowledge Base: Streamlit Application Entry Point.

Tabs:
  1. 🎓 Student Search Panel
  2. 🛠️  Admin Panel
  3. 🤖  Admin AI Agent (optional, requires GROQ_API_KEY)
"""

from __future__ import annotations
import io
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Path setup so src/ imports work whether launched from project root or not
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.admin_agent import AdminAgent
from src.chroma_store import ChromaStore
from src.retriever import UniversityRetriever
from src.schemas import DocMetadata, SearchFilter
from src.utils import suggest_metadata_from_filename
from src.student_agent import StudentAgent
from src.ai_classifier import classify_uploaded_file

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="University Knowledge Base",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global Styles — Premium Dark Academic Aesthetic
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ── Google Fonts ─────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── CSS Variables ────────────────────────────────────────────────── */
    :root {
        --bg-primary:      #080c14;
        --bg-secondary:    #0d1320;
        --bg-card:         #111827;
        --bg-elevated:     #162032;
        --border:          #1e2d45;
        --border-light:    #243347;
        --accent-gold:     #c9a84c;
        --accent-gold-dim: #c9a84c33;
        --accent-blue:     #3b82f6;
        --accent-blue-dim: #3b82f622;
        --accent-teal:     #14b8a6;
        --accent-red:      #f43f5e;
        --text-primary:    #e8edf5;
        --text-secondary:  #7a8fa8;
        --text-muted:      #4a5c70;
        --font-display:    'Playfair Display', Georgia, serif;
        --font-body:       'DM Sans', sans-serif;
        --font-mono:       'JetBrains Mono', monospace;
        --radius-sm:       6px;
        --radius-md:       10px;
        --radius-lg:       16px;
        --shadow-card:     0 4px 24px rgba(0,0,0,0.4);
        --shadow-glow:     0 0 32px rgba(201,168,76,0.08);
    }

    /* ── Base Reset ───────────────────────────────────────────────────── */
    html, body, .stApp {
        background-color: var(--bg-primary) !important;
        font-family: var(--font-body);
        color: var(--text-primary);
    }

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ── Streamlit base overrides ─────────────────────────────────────── */
    .block-container { padding-top: 1.5rem !important; }

    h1, h2, h3 {
        font-family: var(--font-display) !important;
        color: var(--text-primary) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary) !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 0;
        padding: 0 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-body) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        color: var(--text-secondary) !important;
        border-bottom: 2px solid transparent !important;
        padding: 12px 20px !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-gold) !important;
        border-bottom: 2px solid var(--accent-gold) !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem !important;
    }

    /* ── Buttons ──────────────────────────────────────────────────────── */
    .stButton > button {
        font-family: var(--font-body) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        border-radius: var(--radius-sm) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #b8941f, var(--accent-gold)) !important;
        border: none !important;
        color: #0a0a0a !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 12px rgba(201,168,76,0.25) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 20px rgba(201,168,76,0.4) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-light) !important;
        color: var(--text-primary) !important;
    }

    /* ── Inputs ───────────────────────────────────────────────────────── */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        font-size: 0.875rem !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--accent-gold) !important;
        box-shadow: 0 0 0 2px var(--accent-gold-dim) !important;
    }

    /* ── Expanders ────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: var(--text-primary) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
        padding: 1.25rem !important;
    }

    /* ── Alerts ──────────────────────────────────────────────────────── */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: var(--radius-sm) !important;
        font-family: var(--font-body) !important;
        font-size: 0.875rem !important;
    }

    /* ── Dataframe ────────────────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden;
    }

    /* ── Chat ─────────────────────────────────────────────────────────── */
    .stChatMessage {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        margin-bottom: 0.75rem !important;
        padding: 1rem !important;
    }
    .stChatMessage [data-testid="stChatMessageContent"] {
        margin-top: 0 !important;
    }
    .stChatMessage [data-testid="stChatMessageAvatar"] {
        width: 38px !important;
        height: 38px !important;
    }

    /* ── Hero Header ──────────────────────────────────────────────────── */
    .hero-header {
        position: relative;
        background: linear-gradient(135deg, #0d1727 0%, #111e33 50%, #0d1727 100%);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 36px 40px;
        margin-bottom: 28px;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(201,168,76,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-header::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 30%;
        width: 140px; height: 140px;
        background: radial-gradient(circle, rgba(59,130,246,0.07) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: var(--font-display);
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 6px;
        letter-spacing: -0.01em;
        line-height: 1.2;
        position: relative;
        z-index: 1;
    }
    .hero-title span { color: var(--accent-gold); }
    .hero-subtitle {
        font-family: var(--font-body);
        font-size: 0.9rem;
        font-weight: 400;
        color: var(--text-secondary);
        margin: 0;
        position: relative;
        z-index: 1;
        letter-spacing: 0.02em;
    }
    .hero-pills {
        display: flex;
        gap: 8px;
        margin-top: 16px;
        position: relative;
        z-index: 1;
    }
    .hero-pill {
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border-light);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.72rem;
        font-family: var(--font-mono);
        color: var(--text-muted);
        letter-spacing: 0.05em;
    }

    /* ── Section Heading ──────────────────────────────────────────────── */
    .section-heading {
        font-family: var(--font-display);
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 4px;
    }
    .section-caption {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin: 0 0 20px;
        font-family: var(--font-body);
    }

    /* ── Result Card ──────────────────────────────────────────────────── */
    .result-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent-gold);
        border-radius: var(--radius-md);
        padding: 18px 22px;
        margin-bottom: 12px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        border-color: var(--border-light);
        box-shadow: var(--shadow-card);
    }
    .result-card.internal { border-left-color: var(--accent-red); }

    .result-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 10px;
        gap: 12px;
    }
    .result-card-title {
        font-family: var(--font-body);
        font-weight: 600;
        font-size: 0.9rem;
        color: var(--text-primary);
        flex: 1;
    }
    .result-card-badges { display: flex; gap: 6px; align-items: center; flex-shrink: 0; }

    .result-card-body {
        font-size: 0.875rem;
        color: #9bafc4;
        line-height: 1.7;
        margin: 0 0 12px;
        font-family: var(--font-body);
    }
    .result-card-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: var(--font-mono);
        border-top: 1px solid var(--border);
        padding-top: 10px;
    }
    .result-meta-item { display: flex; align-items: center; gap: 4px; }
    .result-meta-label { color: var(--text-muted); }
    .result-meta-value { color: var(--text-secondary); }

    /* ── Badges ───────────────────────────────────────────────────────── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: var(--font-mono);
        letter-spacing: 0.03em;
        white-space: nowrap;
    }
    .badge-score {
        background: var(--accent-gold-dim);
        color: var(--accent-gold);
        border: 1px solid #c9a84c44;
    }
    .badge-public {
        background: rgba(20,184,166,0.12);
        color: var(--accent-teal);
        border: 1px solid rgba(20,184,166,0.3);
    }
    .badge-internal {
        background: rgba(244,63,94,0.1);
        color: var(--accent-red);
        border: 1px solid rgba(244,63,94,0.25);
    }
    .badge-rank {
        background: rgba(201,168,76,0.1);
        color: var(--accent-gold);
        border: 1px solid rgba(201,168,76,0.2);
        font-size: 0.7rem;
        padding: 2px 8px;
    }

    /* ── Stat Card ────────────────────────────────────────────────────── */
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 22px 18px;
        text-align: center;
        transition: border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .stat-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
        opacity: 0.6;
    }
    .stat-card:hover { border-color: var(--border-light); }
    .stat-number {
        font-family: var(--font-display);
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--accent-gold);
        line-height: 1;
        margin-bottom: 6px;
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: var(--font-body);
    }

    /* ── Sidebar Custom ───────────────────────────────────────────────── */
    .sidebar-title {
        font-family: var(--font-display);
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.01em;
    }
    .sidebar-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 16px 0;
    }
    .sidebar-stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        font-size: 0.82rem;
    }
    .sidebar-stat-key { color: var(--text-muted); font-family: var(--font-body); }
    .sidebar-stat-val {
        color: var(--text-secondary);
        font-family: var(--font-mono);
        font-size: 0.78rem;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1px 8px;
    }
    .sidebar-topic-chip {
        display: inline-block;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-family: var(--font-mono);
        color: var(--text-muted);
        margin: 2px 2px 2px 0;
    }
    .sidebar-key-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        padding: 10px 12px;
        border-radius: var(--radius-sm);
        font-family: var(--font-body);
    }
    .sidebar-key-ok {
        background: rgba(20,184,166,0.1);
        border: 1px solid rgba(20,184,166,0.25);
        color: var(--accent-teal);
    }
    .sidebar-key-missing {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.2);
        color: #7ab3f0;
    }

    /* ── Empty State ──────────────────────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: var(--text-muted);
    }
    .empty-state-icon { font-size: 2.5rem; margin-bottom: 12px; }
    .empty-state-title {
        font-family: var(--font-display);
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }
    .empty-state-body { font-size: 0.83rem; line-height: 1.6; }

    /* ── Welcome Dashboard ────────────────────────────────────────────── */
    .welcome-container {
        padding: 40px;
        background: rgba(201,168,76,0.03);
        border: 1px solid rgba(201,168,76,0.1);
        border-radius: var(--radius-lg);
        margin: 20px 0 40px 0;
    }
    .welcome-greeting {
        font-family: var(--font-display);
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--accent-gold);
        margin-bottom: 8px;
    }
    .welcome-subtext {
        font-size: 1rem;
        color: var(--text-muted);
        margin-bottom: 30px;
    }
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
    }
    .dashboard-card {
        padding: 24px;
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .dashboard-card:hover {
        transform: translateY(-4px);
        border-color: var(--accent-gold);
    }
    .card-icon { font-size: 2rem; margin-bottom: 12px; }
    .card-title { font-weight: 700; font-size: 1.1rem; margin-bottom: 8px; }
    .card-desc { font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; }

    /* ── Clickable Suggestion Chips ────────────────────────────────────── */
    .suggestion-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 20px 0;
    }
    .suggestion-chip {
        background: var(--bg-elevated);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-sm);
        padding: 10px 18px;
        font-size: 0.88rem;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .suggestion-chip:hover {
        border-color: var(--accent-gold);
        background: rgba(201,168,76,0.05);
        transform: translateY(-2px);
    }

    /* ── Example Prompts ──────────────────────────────────────────────── */
    .prompt-chip {
        display: inline-block;
        background: var(--bg-elevated);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-sm);
        padding: 6px 14px;
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-family: var(--font-body);
        margin: 4px 4px 4px 0;
        cursor: default;
        font-style: italic;
    }

    /* ── Filter Strip ─────────────────────────────────────────────────── */
    .filter-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-family: var(--font-body);
        margin-bottom: 6px;
    }

    /* ── Results Summary Banner ───────────────────────────────────────── */
    .results-summary {
        background: rgba(201,168,76,0.07);
        border: 1px solid rgba(201,168,76,0.2);
        border-radius: var(--radius-sm);
        padding: 10px 16px;
        font-size: 0.85rem;
        color: var(--accent-gold);
        font-family: var(--font-body);
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _access_badge(access: str) -> str:
    if access == "internal":
        return '<span class="badge badge-internal">⬤ internal</span>'
    return '<span class="badge badge-public">⬥ public</span>'


def _score_badge(score: float) -> str:
    return f'<span class="badge badge-score">↑ {score * 100:.1f}%</span>'


def _rank_badge(n: int) -> str:
    return f'<span class="badge badge-rank">#{n}</span>'


def _meta_item(icon: str, label: str, value: str) -> str:
    if not value:
        return ""
    return (
        f'<span class="result-meta-item">'
        f'<span class="result-meta-label">{icon} {label}</span>'
        f'<span class="result-meta-value">&nbsp;{value}</span>'
        f"</span>"
    )


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS & CACHED RESOURCES
# ═══════════════════════════════════════════════════════════════════════════

CHROMA_DIR = str(Path(__file__).parent / "chroma_db")


@st.cache_resource
def get_store() -> ChromaStore:
    use_groq = bool(os.getenv("GROQ_API_KEY"))
    return ChromaStore(persist_dir=CHROMA_DIR, use_openai=use_groq)


@st.cache_resource
def get_retriever() -> UniversityRetriever:
    return UniversityRetriever(store=get_store())


def get_agent():
    from src.admin_agent import AdminAgent
    return AdminAgent(get_store())


@st.cache_resource
def get_student_agent() -> StudentAgent:
    return StudentAgent(store=get_store())


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

store = get_store()
stats = store.get_stats()

# ── Initialize Session State ─────────────────────────────────────────────
if "user_profile" not in st.session_state:
    from src.schemas import UserProfile
    st.session_state.user_profile = UserProfile(user_id="anonymous", interests=[], privacy_opt_out=False)

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if "is_contributor" not in st.session_state:
    st.session_state.is_contributor = False

if "contributor_id" not in st.session_state:
    st.session_state.contributor_id = None

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">🎓 University KB</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Quick stats
    for key, val in [
        ("Total Chunks", stats.total_chunks),
        ("Unique Sources", stats.unique_sources),
    ]:
        st.markdown(
            f'<div class="sidebar-stat-row">'
            f'<span class="sidebar-stat-key">{key}</span>'
            f'<span class="sidebar-stat-val">{val}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    if stats.year_range[0]:
        st.markdown(
            f'<div class="sidebar-stat-row">'
            f'<span class="sidebar-stat-key">Year Range</span>'
            f'<span class="sidebar-stat-val">{stats.year_range[0]} – {stats.year_range[1]}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    # ── Student Contributor Login ────────────────────────────────────────
    # ── Role Selection: Exclusive Locking ──
    is_admin = st.session_state.get("is_admin", False)
    is_contributor = st.session_state.get("is_contributor", False)

    # 1. Student Contributor Section
    st.markdown(
        '<div style="font-size:0.75rem;font-weight:600;color:var(--text-muted);'
        'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">Student Contributor</div>',
        unsafe_allow_html=True,
    )
    
    # Disable if Admin is active
    s_disabled = is_admin
    s_roll = st.text_input(
        "Enter Roll Number to contribute", 
        key="student_gate", 
        disabled=s_disabled,
        help="Login as Student Contributor. Mutually exclusive with Admin role."
    )
    
    if s_roll:
        if len(s_roll) >= 4:
            st.session_state.is_contributor = True
            st.session_state.contributor_id = s_roll
            # Forcibly log out Admin
            st.session_state.is_admin = False
            st.success(f"✓ Contributor: {s_roll}")
        else:
            st.error("✕ Invalid Roll Number")
            st.session_state.is_contributor = False
    
    if is_admin:
        st.info("🔒 Student access locked while Admin is active.")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    # 2. Admin Authentication Section
    st.markdown(
        '<div style="font-size:0.75rem;font-weight:600;color:var(--text-muted);'
        'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">Admin Authentication</div>',
        unsafe_allow_html=True,
    )
    
    # Disable if Student Contributor is active
    a_disabled = is_contributor
    admin_secret = os.getenv("ADMIN_SECRET", "admin123")
    user_secret = st.text_input(
        "Enter Secret to unlock Admin features", 
        type="password", 
        key="admin_gate", 
        disabled=a_disabled,
        help="Login as Admin. Mutually exclusive with Student role."
    )
    
    if user_secret == admin_secret:
        st.session_state.is_admin = True
        # Forcibly log out Student
        st.session_state.is_contributor = False
        st.session_state.contributor_id = None
        st.success("✓ Admin Unlocked")
    else:
        st.session_state.is_admin = False
        if user_secret:
            st.error("✕ Invalid Secret")

    if is_contributor:
        st.info("🔒 Admin access locked while Contributor is active.")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    groq_set = bool(os.getenv("GROQ_API_KEY"))
    status_cls = "sidebar-key-ok" if groq_set else "sidebar-key-missing"
    status_icon = "✓" if groq_set else "○"
    status_text = (
        "Groq LLM + AI Agent active"
        if groq_set
        else "Local retrieval · AI Agent disabled"
    )
    st.markdown(
        f'<div class="sidebar-key-status {status_cls}">'
        f"<span>{status_icon}</span><span>{status_text}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="position:fixed;bottom:18px;left:0;right:0;'
        'text-align:center;font-size:0.7rem;color:var(--text-muted);'
        'font-family:var(--font-mono)">v1.0 · ChromaDB · LangChain · Groq</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="hero-header">
        <h1 class="hero-title">University <span>Knowledge Base</span></h1>
        <p class="hero-subtitle">
            Intelligent document retrieval — semantic search across handbooks,
            circulars &amp; policies
        </p>
        <div class="hero-pills">
            <span class="hero-pill">ChromaDB</span>
            <span class="hero-pill">LangChain</span>
            <span class="hero-pill">Streamlit</span>
            <span class="hero-pill">Groq LLM</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Welcome Dashboard ────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="welcome-container">
        <div class="welcome-greeting">👋 Welcome to University KB</div>
        <p class="welcome-subtext">
            Your intelligent repository for campus policies, handbooks, and documents. 
            Follow the paths below to get started.
        </p>
        <div class="dashboard-grid">
            <div class="dashboard-card">
                <div class="card-icon">🎓</div>
                <div class="card-title">Students</div>
                <div class="card-desc">Search campus policies and circulars using natural language.</div>
            </div>
            <div class="dashboard-card">
                <div class="card-icon">🗳️</div>
                <div class="card-title">Contributors</div>
                <div class="card-desc">Enter your roll number in the sidebar, then upload documents for AI review.</div>
            </div>
            <div class="dashboard-card">
                <div class="card-icon">🛠️</div>
                <div class="card-title">Admins</div>
                <div class="card-desc">Use the admin secret to unlock ingestion, approvals, and AI agent tools.</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tab_labels = ["  🎓  Student Search  "]
is_admin = st.session_state.get("is_admin", False)
is_contributor = st.session_state.get("is_contributor", False)

if is_contributor:
    tab_labels += ["  📤  Contributor Portal  "]

if is_admin:
    tab_labels += ["  🛠️  Admin Panel  ", "  🤖  AI Agent  "]

tabs = st.tabs(tab_labels)
tab_student = tabs[0]

# Mapping tabs to labels
tab_contributor = None
tab_admin = None
tab_agent = None

current_idx = 1
if is_contributor:
    tab_contributor = tabs[current_idx]
    current_idx += 1

if is_admin:
    tab_admin = tabs[current_idx]
    tab_agent = tabs[current_idx + 1]

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — STUDENT SEARCH
# ═══════════════════════════════════════════════════════════════════════════

with tab_student:
    st.markdown(
        '<div class="section-heading">Campus Information Chatbot</div>'
        '<div class="section-caption">'
        "Ask anything about college rules, campus facilities, or academic policies. "
        "Use natural language to get instant answers from our verified internal sources."
        "</div>",
        unsafe_allow_html=True,
    )

    retriever = get_retriever()
    
    # ── Advanced Filters (Optional) ─────────────────────────────────────
    with st.expander("🛠️ Advanced Search Filters", expanded=False):
        all_topics = [""] + stats.unique_topics
        all_depts  = [""] + stats.unique_departments
        all_years  = [""] + ([str(y) for y in range(stats.year_range[1], stats.year_range[0] - 1, -1)] if stats.year_range[0] else [])
        
        f1, f2, f3 = st.columns(3)
        with f1: f_topic = st.selectbox("Topic", all_topics, key="s_topic")
        with f2: f_dept = st.selectbox("Department", all_depts, key="s_dept")
        with f3: f_year = st.selectbox("Year", all_years, key="s_year")

    # ── Try Asking (Suggestion Chips) ───────────────────────────────────
    st.markdown('<div class="section-caption" style="margin-top:24px; margin-bottom:8px;">TRY ASKING</div>', unsafe_allow_html=True)
    
    # Use columns for a grid of 2x2 clickable chips
    sq1, sq2 = st.columns(2)
    suggestions = [
        ("📜 What are the exam attendance rules?", "What are the exam attendance rules?"),
        ("📘 What is the fee structure for CSE?", "What is the fee structure for CSE?"),
        ("🗓️ What were the 2023 circular updates?", "What were the 2023 circular updates?"),
        ("🎓 What documents are required for graduation?", "What documents are required for graduation?"),
    ]
    
    suggestion_clicked = None
    with sq1:
        if st.button(suggestions[0][0], key="suggest_1", use_container_width=True):
            suggestion_clicked = suggestions[0][1]
        if st.button(suggestions[2][0], key="suggest_3", use_container_width=True):
            suggestion_clicked = suggestions[2][1]
    with sq2:
        if st.button(suggestions[1][0], key="suggest_2", use_container_width=True):
            suggestion_clicked = suggestions[1][1]
        if st.button(suggestions[3][0], key="suggest_4", use_container_width=True):
            suggestion_clicked = suggestions[3][1]

    # ── Chat Logic ──────────────────────────────────────────────────────
    if "student_messages" not in st.session_state:
        st.session_state.student_messages = []
    
    # Render Chat History
    for msg in st.session_state.student_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "results" in msg and msg["results"]:
                with st.expander("📚 Verified Sources"):
                    for i, chunk in enumerate(msg["results"], 1):
                        st.markdown(f"**Source {i}:** {chunk.metadata.get('source_file')}")
                        st.caption(chunk.content[:400] + "...")

    # Chat Input
    user_query = st.chat_input("Ask about rules, policies, or campus life...")
    
    # If a suggestion was clicked, override user_query
    if suggestion_clicked:
        user_query = suggestion_clicked

    if user_query:
        st.session_state.student_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Campus Assistant is thinking..."):
                s_agent = get_student_agent()
                graph_output = s_agent.ask(
                    query=user_query,
                    profile=st.session_state.user_profile,
                    history=st.session_state.get("student_history", [])
                )
                
                ai_answer = graph_output.get("answer", "I couldn't find a specific answer in the documents.")
                results = graph_output.get("results", [])
                
                # Persist history
                st.session_state.student_history = graph_output.get("history", [])
                
                # Display Answer
                st.markdown(ai_answer)
                
                # Show references if any
                if results:
                    with st.expander("📚 Verified Sources"):
                        for i, chunk in enumerate(results, 1):
                            st.markdown(f"**Source {i}:** {chunk.metadata.get('source_file')}")
                            st.caption(chunk.content[:400] + "...")
                
                st.session_state.student_messages.append({
                    "role": "assistant", 
                    "content": ai_answer,
                    "results": results
                })
        
        st.rerun()

    # Clear Chat Button
    if st.session_state.student_messages:
        if st.sidebar.button("🗑️ Clear Student Chat"):
            st.session_state.student_messages = []
            st.session_state.student_history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TAB: CONTRIBUTOR PORTAL
# ═══════════════════════════════════════════════════════════════════════════

if tab_contributor:
    with tab_contributor:
        st.markdown(
            '<div class="section-heading">Contributor Portal</div>'
            '<div class="section-caption">'
            "Submit new university circulars or policies. Your submissions will be "
            "audited by AI and then reviewed by a human admin before publishing."
            "</div>",
            unsafe_allow_html=True,
        )
        
        c_file = st.file_uploader("Upload Document for Review", type=["pdf", "docx"], key="cont_upload")
        
        # --- AI Auto-Classification logic ---
        if c_file:
            if "last_cont_file" not in st.session_state or st.session_state.last_cont_file != c_file.name:
                with st.spinner("🧠 AI is analyzing document content for metadata..."):
                    ai_meta = classify_uploaded_file(c_file)
                    st.session_state.cont_ai_meta = ai_meta
                    st.session_state.last_cont_file = c_file.name
                    
                    # Directly update input keys to force-refresh the UI
                    st.session_state.cont_topic = ai_meta.get("topic", "")
                    st.session_state.cont_dept = ai_meta.get("department", "GENERAL")
                    if ai_meta.get("year") and ai_meta.get("year") > 0:
                        st.session_state.cont_year = ai_meta.get("year")
                    st.session_state.cont_priority = ai_meta.get("priority", "medium")
                    
                    st.toast(f"AI suggested metadata for {c_file.name}")

        ai_m = st.session_state.get("cont_ai_meta", {})
        
        c_topic = st.text_input("Short Topic Name (e.g. 'exam_rules')", value=ai_m.get("topic", ""), key="cont_topic")
        c_dept = st.text_input("Department", value=ai_m.get("department", "GENERAL"), key="cont_dept")
        c_year = st.number_input("Year", min_value=2000, max_value=2100, value=ai_m.get("year") if ai_m.get("year") and ai_m.get("year") > 0 else datetime.now().year, key="cont_year")
        c_priority = st.selectbox("Priority", options=["high", "medium", "low"], index=["high", "medium", "low"].index(ai_m.get("priority", "medium")), key="cont_priority")
        
        if c_file and c_topic:
            # Show AI Summary if available
            if ai_m.get("overall_summary"):
                st.info(f"📝 **AI Summary:** {ai_m['overall_summary']}")
            
            with st.spinner("AI is auditing your submission..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(c_file.name).suffix) as tmp:
                    tmp.write(c_file.read())
                    tmp_path = tmp.name
                
                # Run AI Audit
                audit_report = get_agent().run(f"verify_contribution_tool for {tmp_path}")
                st.info(f"📋 **AI Audit Report:**\n\n{audit_report}")
                
                if st.button("🚀 Submit for Approval", type="primary"):
                    metadata = {
                        "source_file": c_file.name,
                        "topic": c_topic,
                        "department": c_dept,
                        "year": c_year,
                        "priority": c_priority,
                        "contributor_id": st.session_state.contributor_id,
                        "status": "archived",
                        "verification_status": "pending",
                        "audit_report": audit_report,
                        "quality_score": ai_m.get("topic_confidence", 0.8) # Use AI confidence
                    }
                    res = get_agent().upsert_doc(tmp_path, metadata)
                    os.unlink(tmp_path)
                    
                    if "🚫" in res or "❌" in res:
                        st.error(f"Upload Blocked by Integrity Guardrails:\n\n{res}")
                    else:
                        st.success("✅ Submitted! Your document is now in the admin review queue.")
                        st.cache_resource.clear()
                        # Reset contributor inputs
                        st.session_state.last_cont_file = None
                        st.rerun()

if tab_admin:
    with tab_admin:
        agent = get_agent()

        st.markdown(
            '<div class="section-heading">Admin Panel</div>'
            '<div class="section-caption">'
            "Ingest, manage, inspect, and export knowledge-base documents."
            "</div>",
            unsafe_allow_html=True,
        )

        # ── A: KB Statistics ─────────────────────────────────────────────────
        with st.expander("📊  Knowledge Base Statistics", expanded=True):
            s   = get_store().get_stats()
            c1, c2, c3, c4, c5, c6 = st.columns(6)

            for col, number, label in [
                (c1, s.total_chunks,           "Total Chunks"),
                (c2, s.unique_sources,         "Unique Sources"),
                (c3, len(s.unique_topics),     "Topics"),
                (c4, len(s.unique_departments),"Departments"),
                (c5, getattr(s, "archived_chunks", 0), "Archived"),
                (c6, getattr(s, "obsolete_chunks", 0), "Obsolete"),
            ]:
                col.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-number">{number}</div>'
                    f'<div class="stat-label">{label}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if s.doc_type_counts:
                st.markdown(
                    '<div style="margin-top:16px;font-size:0.8rem;font-weight:600;'
                    'color:var(--text-muted);text-transform:uppercase;letter-spacing:0.07em;'
                    'margin-bottom:6px">Doc Type Breakdown</div>',
                    unsafe_allow_html=True,
                )
                st.json(s.doc_type_counts)

            if st.button("🔄  Refresh Stats", key="refresh_stats"):
                st.cache_resource.clear()
                st.rerun()


        # ── B: Ingest Document ───────────────────────────────────────────────
        with st.expander("📤  Ingest / Upsert Document", expanded=False):
            uploaded_file = st.file_uploader(
                "Upload document",
                type=["pdf", "docx", "html", "htm"],
                help="Supported formats: PDF, DOCX, HTML",
            )

            if uploaded_file:
                if "last_admin_file" not in st.session_state or st.session_state.last_admin_file != uploaded_file.name:
                    with st.spinner("🧠 AI is analyzing document content for metadata..."):
                        ai_meta = classify_uploaded_file(uploaded_file)
                        st.session_state.admin_ai_meta = ai_meta
                        st.session_state.last_admin_file = uploaded_file.name
                        
                        # Force update session state keys for Admin inputs
                        st.session_state.ingest_topic = ai_meta.get("topic", "")
                        st.session_state.ingest_dept = ai_meta.get("department", "GENERAL")
                        if ai_meta.get("year") and ai_meta.get("year") > 2000:
                            st.session_state.ingest_year = ai_meta.get("year")
                        st.session_state.ingest_priority = ai_meta.get("priority", "medium")
                        st.session_state.ingest_doc_type = ai_meta.get("doc_type", "other")
                        
                        st.toast(f"AI suggested metadata for {uploaded_file.name}")
                
                suggested = st.session_state.get("admin_ai_meta", {})
                if not suggested:
                     suggested = suggest_metadata_from_filename(uploaded_file.name)
            else:
                suggested = {}

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            m_doc_type = st.selectbox(
                "Doc Type",
                ["handbook", "circular", "policy", "poster", "attendance", "certificate", "event", "other"],
                index=["handbook", "circular", "policy", "poster", "attendance", "certificate", "event", "other"].index(
                    suggested.get("doc_type", "other")
                ),
                key="ingest_doc_type",
            )
            m_topic = st.text_input(
                "Topic", value=suggested.get("topic", ""), key="ingest_topic"
            )
            suggested_year = int(suggested.get("year", 2024))
            suggested_year = suggested_year if suggested_year >= 2000 else 2024
            m_year = st.number_input(
                "Year",
                min_value=2000, max_value=2100,
                value=suggested_year,
                key="ingest_year",
            )
            m_priority = st.selectbox(
                "Priority",
                options=["high", "medium", "low"],
                index=["high", "medium", "low"].index(suggested.get("priority", "medium")),
                key="ingest_priority"
            )
        with col_m2:
            m_dept = st.text_input(
                "Department",
                value=suggested.get("department", "GENERAL"),
                key="ingest_dept",
            )
            m_access = st.selectbox(
                "Access",
                ["public", "internal"],
                index=0 if suggested.get("access", "public") == "public" else 1,
                key="ingest_access",
            )
            m_version = st.text_input(
                "Version",
                value=suggested.get("version", "1.0"),
                key="ingest_version",
            )
            if suggested.get("overall_summary"):
                st.info(f"📝 **AI Summary:** {suggested['overall_summary']}")
        m_section = st.text_input("Section (optional)", key="ingest_section")
        m_effective = st.date_input("Effective Date", value=datetime.now(), key="ingest_effective")

        # ── Real-time Duplication Advisor ────────────────────────────────
        if uploaded_file and m_topic and m_dept:
            dup_result = agent.detect_duplicates(
                {"topic": m_topic, "department": m_dept, "year": m_year}
            )
            if "NO_DUPLICATES" not in dup_result:
                st.markdown(
                    f'<div class="danger-banner" style="margin-top:10px; border-left:4px solid var(--accent-red);">'
                    f'<strong>⚠️ Duplicate Alert:</strong><br>{dup_result}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="font-size:0.8rem; color:var(--accent-teal); margin-top:10px;">'
                    '✓ No duplicate topic/department detected.</div>',
                    unsafe_allow_html=True
                )

        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
        if st.button(
            "⬆️  Ingest Document",
            type="primary",
            disabled=not uploaded_file,
            key="ingest_btn",
            use_container_width=True
        ):
                if uploaded_file:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_file.name).suffix
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    metadata_dict = {
                        "source_file": uploaded_file.name,
                        "doc_type": m_doc_type,
                        "topic": m_topic,
                        "year": m_year,
                        "department": m_dept,
                        "access": m_access,
                        "priority": m_priority,
                        "version": m_version,
                        "section": m_section,
                        "effective_date": m_effective.isoformat(),
                    }

                    with st.spinner("Ingesting…"):
                        result = agent.upsert_doc(tmp_path, metadata_dict)
                    os.unlink(tmp_path)

                    if "SUCCESS" in result or "✅" in result:
                        st.success(result)
                        st.cache_resource.clear()
                    elif "ERROR" in result or "❌" in result:
                        st.error(result)
                    else:
                        st.warning(result)

        # ── B2: Bulk Ingestion Monitor ──────────────────────────────────────
        with st.expander("📦  Bulk Ingestion Monitor", expanded=False):
            st.markdown(
                '<div class="section-caption">'
                "Upload multiple documents at once. Metadata provided below will be used as a template "
                "for all files in the batch. Files with formats like <code>.pdf</code>, <code>.docx</code>, <code>.html</code> "
                "are supported."
                "</div>",
                unsafe_allow_html=True,
            )

            bulk_files = st.file_uploader(
                "Select multiple files",
                type=["pdf", "docx", "html", "htm"],
                accept_multiple_files=True,
                help="Maximum transparency for batch uploads.",
                key="bulk_uploader"
            )

            if bulk_files:
                st.info(f"📋 {len(bulk_files)} files selected for batch ingestion.")
                
                # Batch Metadata Template
                with st.container():
                    st.markdown("##### 🛠️ Batch Metadata Template")
                    use_ai_bulk = st.checkbox("🧠 Use AI to auto-detect metadata for each file individually", value=True, key="bulk_use_ai")
                    
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        b_doc_type = st.selectbox("Doc Type", ["handbook", "circular", "policy", "poster", "attendance", "certificate", "event", "other"], key="bulk_doc_type", disabled=use_ai_bulk)
                        b_year = st.number_input("Year", min_value=2000, max_value=2100, value=datetime.now().year, key="bulk_year", disabled=use_ai_bulk)
                    with col_b2:
                        b_dept = st.text_input("Department", value="GENERAL", key="bulk_dept", disabled=use_ai_bulk)
                        b_access = st.selectbox("Access", ["public", "internal"], key="bulk_access", disabled=use_ai_bulk)
                    
                    b_version = st.text_input("Version", value="1.0", key="bulk_version")
                    b_effective = st.date_input("Effective Date", value=datetime.now(), key="bulk_effective")

                if st.button("🚀  Start Bulk Ingestion", type="primary", use_container_width=True, key="start_bulk_btn"):
                    job_id = f"job_{int(datetime.now().timestamp())}"
                    files_to_process = []
                    
                    # Prepare temporary files
                    temp_paths = []
                    progress_text = "Analyzing files with AI..." if use_ai_bulk else "Preparing files..."
                    with st.status(f"🛠️ {progress_text}", expanded=True) as status:
                        for f in bulk_files:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                                tmp.write(f.read())
                                tmp_path = tmp.name
                                temp_paths.append(tmp_path)
                                
                                if use_ai_bulk:
                                    st.write(f"🧠 AI Classifying `{f.name}`...")
                                    ai_results = classify_uploaded_file(f)
                                    metadata = {
                                        "source_file": f.name,
                                        "doc_type": ai_results.get("doc_type", "other"),
                                        "year": ai_results.get("year", b_year),
                                        "department": ai_results.get("department", b_dept),
                                        "topic": ai_results.get("topic", "general"),
                                        "access": ai_results.get("access", b_access),
                                        "priority": ai_results.get("priority", "medium"),
                                        "version": b_version,
                                        "effective_date": b_effective.isoformat(),
                                        "quality_score": ai_results.get("topic_confidence", 0.8)
                                    }
                                else:
                                    metadata = {
                                        "source_file": f.name,
                                        "doc_type": b_doc_type,
                                        "year": b_year,
                                        "department": b_dept,
                                        "access": b_access,
                                        "priority": "medium",
                                        "version": b_version,
                                        "effective_date": b_effective.isoformat(),
                                    }
                                
                                files_to_process.append({
                                    "path": tmp_path,
                                    "metadata": metadata
                                })
                        
                        st.write("🚀 Initializing ingestion job...")
                        
                        # Use the NEW generator-based bulk ingestion for real-time progress
                        progress_bar = st.progress(0, text="Starting ingestion...")
                        
                        final_report = None
                        for count, detail, current_report in agent.bulk_ingest_gen(files_to_process, job_id):
                            # Update progress bar
                            percent = count / len(files_to_process)
                            progress_bar.progress(percent, text=f"Processing file {count}/{len(files_to_process)}")
                            
                            # Show detail if it's a file completion
                            if detail:
                                if detail.status == "success":
                                    st.write(f"✅ {detail.filename}: Ingested {detail.chunks} chunks.")
                                elif detail.status == "unsupported":
                                    st.write(f"⚠️ {detail.filename}: Unsupported format.")
                                else:
                                    st.write(f"❌ {detail.filename}: Failed - {detail.error}")
                            
                            final_report = current_report
                        
                        report = final_report
                        status.update(label="✅ Batch Ingestion Complete!", state="complete", expanded=False)

                    # Cleanup temp files
                    for p in temp_paths:
                        if os.path.exists(p): os.unlink(p)

                    st.session_state.last_ingestion_report = report
                    st.cache_resource.clear()
                    st.rerun()

            # Display Last Report if exists
            if "last_ingestion_report" in st.session_state:
                rep = st.session_state.last_ingestion_report
                st.markdown("---")
                st.markdown(f"### 📊 Ingestion Report: `{rep.job_id}`")
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Files", rep.total_files)
                c2.metric("Success", rep.successful_files)
                c3.metric("Failed", rep.failed_files, delta=-rep.failed_files, delta_color="inverse")
                c4.metric("Unsupported", rep.unsupported_files)
                c5.metric("Chunks Created", rep.total_chunks)

                if rep.file_details:
                    df_rep = pd.DataFrame([d.model_dump() for d in rep.file_details])
                    st.table(df_rep)
                
                if st.button("🧹 Clear Report"):
                    del st.session_state.last_ingestion_report
                    st.rerun()

        # ── C: Delete by Metadata ────────────────────────────────────────────
        with st.expander("🗑️  Delete Documents by Metadata", expanded=False):
            st.markdown(
                '<div class="danger-banner">'
                "⚠️  Deletions are permanent and cannot be undone. "
                "Apply filters carefully before confirming."
                "</div>",
                unsafe_allow_html=True,
            )

            d1, d2, d3 = st.columns(3)
            with d1:
                d_topic   = st.text_input("Topic",   key="del_topic")
                d_year    = st.number_input(
                    "Year (0 = any)", min_value=0, max_value=2100, value=0, key="del_year"
                )
            with d2:
                d_dept    = st.text_input("Department", key="del_dept")
                d_version = st.text_input("Version",    key="del_version")
            with d3:
                d_access  = st.selectbox("Access",   ["", "public", "internal"], key="del_access")
                d_doctype = st.selectbox(
                    "Doc Type", ["", "handbook", "circular", "policy", "other"], key="del_doctype"
                )

            confirm_delete = st.checkbox(
                "I understand this is irreversible and confirm deletion"
            )

            if st.button(
                "🗑️  Delete Matching Chunks",
                type="secondary",
                disabled=not confirm_delete,
                key="delete_btn",
            ):
                filters: dict = {}
                if d_topic:   filters["topic"]    = d_topic
                if d_year:    filters["year"]     = d_year
                if d_dept:    filters["department"] = d_dept
                if d_version: filters["version"]  = d_version
                if d_access:  filters["access"]   = d_access
                if d_doctype: filters["doc_type"] = d_doctype

                if not filters:
                    st.error("Specify at least one filter to prevent accidental mass deletion.")
                else:
                    with st.spinner("Deleting…"):
                        result = agent.delete_by_metadata(filters)
                    if "SUCCESS" in result:
                        st.success(result)
                        st.cache_resource.clear()
                    else:
                        st.error(result)

        # ── D: Export Metadata ───────────────────────────────────────────────
        with st.expander("💾  Export Metadata Backup", expanded=False):
            all_meta = get_store().get_all_metadata()

            if not all_meta:
                st.markdown(
                    '<div class="empty-state" style="padding:30px 20px;">'
                    '<div class="empty-state-icon">📭</div>'
                    '<div class="empty-state-title">Nothing to export</div>'
                    '<div class="empty-state-body">Ingest documents first.</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                df = pd.DataFrame(all_meta)

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "⬇️  Download JSON",
                        data=json.dumps(all_meta, indent=2).encode(),
                        file_name="kb_metadata_backup.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with dl2:
                    csv_buf = io.StringIO()
                    df.to_csv(csv_buf, index=False)
                    st.download_button(
                        "⬇️  Download CSV",
                        data=csv_buf.getvalue().encode(),
                        file_name="kb_metadata_backup.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                st.dataframe(df, use_container_width=True, height=320)

        # ── E: Reindex Recommendation ────────────────────────────────────────
        with st.expander("🔄  Reindex Recommendation", expanded=False):
            st.caption(
                "Analyse the knowledge base for fragmentation, stale vectors, "
                "or coverage gaps."
            )
            if st.button("Analyse KB Health", key="health_btn"):
                with st.spinner("Analysing…"):
                    rec = agent.recommend_reindex()
                if "RECOMMENDATION" in rec.upper():
                    st.warning(rec)
                else:
                    st.success(rec)

        # ── G: Version Lineage & History Manager ─────────────────────────────
        with st.expander("📜 Version Lineage & History Manager", expanded=False):
            st.markdown(
                '<div style="font-size:0.85rem; color:var(--text-muted); margin-bottom:15px;">'
                "Inspect the lifecycle of specific circulars and perform rollbacks to previous versions."
                "</div>",
                unsafe_allow_html=True
            )
            
            vh1, vh2 = st.columns([2, 1])
            with vh1:
                h_topic = st.text_input("Topic to Inspect (e.g. 'exam_rules')", key="vh_topic")
            with vh2:
                h_dept = st.text_input("Department", value="GENERAL", key="vh_dept")
                
            if st.button("🔍  Fetch Version History"):
                if not h_topic:
                    st.error("Please enter a topic.")
                else:
                    history = get_store().get_version_history(h_topic, h_dept or "GENERAL")
                    if not history:
                        st.info(f"No history found for topic '{h_topic}'.")
                    else:
                        for doc in history:
                            v_status = doc.get("status") or "active"
                            status_color = "#34d399" if v_status == "active" else "#94a3b8"
                            if v_status == "obsolete": status_color = "#ef4444"
                            
                            with st.container():
                                st.markdown(
                                    f'<div style="padding:15px; border:1px solid var(--border-subtle); '
                                    f'border-radius:8px; margin-bottom:10px; border-left:5px solid {status_color};">'
                                    f'<div style="display:flex; justify-content:space-between;">'
                                    f'<strong>{doc.get("source_file", "Unknown")} (v{doc.get("version", "1.0")})</strong>'
                                    f'<span style="background:{status_color}; color:white; padding:2px 8px; '
                                    f'border-radius:4px; font-size:0.7rem; font-weight:bold;">{v_status.upper()}</span>'
                                    f'</div>'
                                    f'<div style="font-size:0.8rem; color:var(--text-muted); margin-top:5px;">'
                                    f'Effective: {doc.get("effective_date", "N/A")} | Year: {doc.get("year", "N/A")}</div>'
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                                
                                c1, c2, c3 = st.columns([1,1,1])
                                if doc.get("status") != "active":
                                    with c1:
                                        if st.button(f"Activate {doc.get('version')}", key=f"rollback_{doc.get('version')}_{doc.get('year')}"):
                                            with st.spinner("Rolling back..."):
                                                res = agent.rollback_version(h_topic, h_dept, doc.get("source_file"))
                                                st.success(res)
                                                st.cache_resource.clear()
                                                st.rerun()
                                
                                if doc.get("status") == "active":
                                    with c2:
                                        if st.button("Archive Old Docs", key=f"sup_{doc.get('version')}"):
                                            with st.spinner("Archiving..."):
                                                res = agent.supersede_document(h_topic, h_dept, doc.get("source_file"))
                                                st.success(res)
                                                st.cache_resource.clear()
                                                st.rerun()

        # ── I: Pending Approvals Queue ────────────────────────────────────────
        with st.expander("📥  Pending Approvals Queue", expanded=True):
            st.caption("Review student submissions and AI audit reports.")
            
            # Fetch pending chunks EFFICIENTLY
            pending = get_store().get_pending_chunks()
            
            if not pending:
                st.info("No pending submissions to review.")
            else:
                # Group by source file
                pending_sources = {}
                for m in pending:
                    src = m.get("source_file", "unknown")
                    pending_sources.setdefault(src, []).append(m)
                    
                for src, metas in pending_sources.items():
                    m = metas[0]
                    # Get IDs for all chunks of this pending document
                    # Important: Only target chunks that are ACTUALLY pending
                    with st.container():
                        st.markdown(
                            f'<div style="padding:15px; border:1px solid var(--border-subtle); border-radius:8px; margin-bottom:10px; border-left:5px solid #f59e0b;">'
                            f'<strong>{src}</strong> submitted by student <code>{m.get("contributor_id")}</code><br>'
                            f'<span style="font-size:0.8rem; color:var(--text-muted);">Topic: {m.get("topic")} | Dept: {m.get("department")} | Year: {m.get("year")}</span>'
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        
                        if m.get("audit_report"):
                            with st.expander("📄 View AI Audit Report"):
                                st.info(m.get("audit_report"))
                        
                        bq1, bq2, _ = st.columns([1, 1, 2])
                        with bq1:
                            if st.button("✅ Approve & Publish", key=f"appr_{src}"):
                                with st.spinner("Publishing..."):
                                    # Update metadata: change status to active and verified
                                    # Target ONLY the chunks for this source that are pending
                                    try:
                                        store = get_store()
                                        res_q = store._collection.get(
                                            where={
                                                "$and": [
                                                    {"source_file": {"$eq": src}},
                                                    {"verification_status": {"$eq": "pending"}}
                                                ]
                                            },
                                            include=[]
                                        )
                                        target_ids = res_q.get("ids", [])
                                        if target_ids:
                                            store.update_metadata(target_ids, {
                                                "status": "active",
                                                "verification_status": "verified",
                                                "is_archived": False
                                            })
                                            st.success(f"Published {src}!")
                                            st.cache_resource.clear()
                                            st.rerun()
                                        else:
                                            st.error("Could not locate pending chunks for this document.")
                                    except Exception as e:
                                        st.error(f"Approval failed: {e}")
                        with bq2:
                            if st.button("❌ Reject", key=f"rej_{src}"):
                                with st.spinner("Deleting..."):
                                    try:
                                        store = get_store()
                                        res_q = store._collection.get(
                                            where={
                                                "$and": [
                                                    {"source_file": {"$eq": src}},
                                                    {"verification_status": {"$eq": "pending"}}
                                                ]
                                            },
                                            include=[]
                                        )
                                        target_ids = res_q.get("ids", [])
                                        if target_ids:
                                            store.delete_chunks(target_ids)
                                            st.warning(f"Rejected and deleted {src}.")
                                            st.cache_resource.clear()
                                            st.rerun()
                                        else:
                                            st.error("Could not locate pending chunks for this document.")
                                    except Exception as e:
                                        st.error(f"Rejection failed: {e}")
        # ── H: Automated Deduplication Sweep ─────────────────────────────────
        with st.expander("🧹  Automated Deduplication Sweep", expanded=False):
            st.markdown(
                '<div class="section-caption">'
                "Scan the KB for superseded document versions and semantically redundant content. "
                "Identified duplicates will be moved to <code>archive_kb</code>."
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("🚀  Run Duplicate Sweep", type="primary", key="sweep_btn"):
                with st.spinner("Sweeping for duplicates…"):
                    from src.admin_agent import _auto_deduplicate
                    report = _auto_deduplicate(agent._store)
                
                if "SUCCESS" in report or "✅" in report:
                    st.success(report)
                    st.cache_resource.clear()
                else:
                    st.info(report)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — ADMIN AI AGENT
# ═══════════════════════════════════════════════════════════════════════════

if tab_agent:
    with tab_agent:
        st.markdown(
            '<div class="section-heading">Admin AI Agent</div>'
            '<div class="section-caption">'
            "Manage your knowledge base through natural-language conversation."
            "</div>",
            unsafe_allow_html=True,
        )

        if not os.getenv("GROQ_API_KEY"):
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-state-icon">🤖</div>'
                '<div class="empty-state-title">AI Agent Unavailable</div>'
                '<div class="empty-state-body">'
                "<code>GROQ_API_KEY</code> is not set in your environment.<br>"
                "Set the variable and restart the app to enable this feature.<br><br>"
                "All admin tasks remain available in the <strong>Admin Panel</strong> tab."
                "</div></div>",
                unsafe_allow_html=True,
            )
        else:
            agent_inst = get_agent()

            # Chat history in session state
            if "agent_messages" not in st.session_state:
                st.session_state.agent_messages = []

            # Display chat history (Standard Flow)
            for msg in st.session_state.agent_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Input (remains at bottom naturally)
            if user_input := st.chat_input("Ask the Admin Agent… (e.g. 'Show KB stats')"):
                st.session_state.agent_messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Executing admin command…"):
                        response = agent_inst.run(user_input)
                    st.markdown(response)
                    st.session_state.agent_messages.append({"role": "assistant", "content": response})
                
                st.rerun()

            col_clear, col_space = st.columns([1, 3])
            with col_clear:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.agent_messages = []
                    st.rerun()

            # ─────────────────────────────────────────────────────────────────
            # DEBUG LOGS (Traceability)
            # ─────────────────────────────────────────────────────────────────
            with st.expander("🛠️ Debug Information (Agent Logs)", expanded=False):
                log_file = "data/agent_debug.log"
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    st.text_area("Live Agent Logs (Last 50 events)", value="".join(lines[-50:]), height=300)
                    if st.button("🗑️ Clear Debug Logs"):
                        os.remove(log_file)
                        st.rerun()
                else:
                    st.info("No debug logs found yet. Run the agent first!")

        st.markdown("")
        st.markdown(
            '<div style="margin-top:30px; padding-top:20px; border-top:1px solid rgba(201,168,76,0.2);">'
            '<div class="section-caption" style="margin-bottom:15px;">💡 Example Prompts</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        examples = [
            "Show me the current KB statistics.",
            "Check for duplicate exam rules documents.",
            "Are there any reindexing recommendations?",
            "What topics are available in the knowledge base?",
            "Delete all chunks from year 2022 with topic exam_rules.",
        ]

        for ex in examples:
            st.markdown(
                f'<div class="prompt-chip" style="'
                'padding:10px 14px;margin:6px 0;'
                'background:rgba(201,168,76,0.08);border-left:3px solid #c9a84c;'
                'border-radius:4px;font-size:0.9rem;cursor:pointer;">'
                f'{ex}</div>',
                unsafe_allow_html=True,
            )
