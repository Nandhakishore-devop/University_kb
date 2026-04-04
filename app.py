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

    /* ── Danger Zone ──────────────────────────────────────────────────── */
    .danger-banner {
        background: rgba(244,63,94,0.07);
        border: 1px solid rgba(244,63,94,0.2);
        border-radius: var(--radius-sm);
        padding: 12px 16px;
        font-size: 0.83rem;
        color: #f87295;
        font-family: var(--font-body);
        margin-bottom: 16px;
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


@st.cache_resource
def get_agent() -> AdminAgent:
    return AdminAgent(store=get_store())


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

store = get_store()
stats = store.get_stats()

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

    st.markdown(
        '<div style="font-size:0.75rem;font-weight:600;color:var(--text-muted);'
        'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">Topics</div>',
        unsafe_allow_html=True,
    )
    if stats.unique_topics:
        chips = "".join(
            f'<span class="sidebar-topic-chip">{t}</span>'
            for t in stats.unique_topics[:10]
        )
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.markdown(
            '<span style="font-size:0.78rem;color:var(--text-muted);">'
            "No topics yet.</span>",
            unsafe_allow_html=True,
        )

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

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tab_student, tab_admin, tab_agent = st.tabs(
    ["  🎓  Student Search  ", "  🛠️  Admin Panel  ", "  🤖  AI Agent  "]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — STUDENT SEARCH
# ═══════════════════════════════════════════════════════════════════════════

with tab_student:
    st.markdown(
        '<div class="section-heading">Search University Documents</div>'
        '<div class="section-caption">'
        "Full-text and semantic search across public handbooks, circulars, and policies."
        "</div>",
        unsafe_allow_html=True,
    )

    retriever = get_retriever()

    # ── Search bar ──────────────────────────────────────────────────────
    q_col, k_col = st.columns([6, 1])
    with q_col:
        query = st.text_input(
            "query",
            placeholder="e.g. What are the rules for examinations?",
            label_visibility="collapsed",
        )
    with k_col:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

    # ── Filters ─────────────────────────────────────────────────────────
    st.markdown('<div class="filter-label">Filters</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)

    all_topics = [""] + stats.unique_topics
    all_depts  = [""] + stats.unique_departments
    all_years  = [""] + (
        [str(y) for y in range(stats.year_range[1], stats.year_range[0] - 1, -1)]
        if stats.year_range[0]
        else []
    )

    with f1:
        f_topic   = st.selectbox("Topic", all_topics, key="s_topic")
    with f2:
        f_dept    = st.selectbox("Department", all_depts, key="s_dept")
    with f3:
        f_year    = st.selectbox("Year", all_years, key="s_year")
    with f4:
        f_doctype = st.selectbox(
            "Doc Type", ["", "handbook", "circular", "policy", "other"], key="s_dtype"
        )

    search_clicked = st.button(
        "🔍  Search", type="primary", use_container_width=True, key="search_btn"
    )

    # ── Execute search ───────────────────────────────────────────────────
    if search_clicked and query.strip():
        filters = SearchFilter(
            topic=f_topic or None,
            department=f_dept or None,
            year=int(f_year) if f_year else None,
            doc_type=f_doctype or None,
            access="public",
        )

        with st.spinner("Searching knowledge base…"):
            results = retriever.search(
                query=query,
                filters=filters,
                top_k=top_k,
                admin_override=False,
            )

        if not results:
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-state-icon">🔎</div>'
                '<div class="empty-state-title">No results found</div>'
                '<div class="empty-state-body">Try broader keywords or adjust your filters.<br>'
                "The knowledge base may not contain documents matching this query.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="results-summary">✦ Found '
                f'<strong>{len(results)}</strong> relevant chunk'
                f"{'s' if len(results) != 1 else ''} for your query</div>",
                unsafe_allow_html=True,
            )

            for i, chunk in enumerate(results, 1):
                access     = chunk.metadata.get("access", "public")
                card_cls   = "result-card" + (" internal" if access == "internal" else "")
                meta_items = "".join([
                    _meta_item("📁", "Topic",   str(chunk.metadata.get("topic",   ""))),
                    _meta_item("🏛",  "Dept",    str(chunk.metadata.get("department", ""))),
                    _meta_item("📅", "Year",    str(chunk.metadata.get("year",    ""))),
                    _meta_item("📄", "Type",    str(chunk.metadata.get("doc_type",""))),
                    _meta_item("🔖", "Version", str(chunk.metadata.get("version", ""))),
                    _meta_item("📃", "Page",    str(chunk.metadata.get("page_number",""))),
                ])

                st.markdown(
                    f"""
                    <div class="{card_cls}">
                        <div class="result-card-header">
                            <div class="result-card-title">
                                {_rank_badge(i)}&ensp;{chunk.metadata.get("source_file", "—")}
                            </div>
                            <div class="result-card-badges">
                                {_access_badge(access)}
                                {_score_badge(chunk.score)}
                            </div>
                        </div>
                        <p class="result-card-body">{chunk.content[:900]}</p>
                        <div class="result-card-meta">{meta_items}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    elif search_clicked:
        st.warning("Please enter a search query before pressing Search.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — ADMIN PANEL
# ═══════════════════════════════════════════════════════════════════════════

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
        c1, c2, c3, c4 = st.columns(4)

        for col, number, label in [
            (c1, s.total_chunks,           "Total Chunks"),
            (c2, s.unique_sources,         "Unique Sources"),
            (c3, len(s.unique_topics),     "Topics"),
            (c4, len(s.unique_departments),"Departments"),
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

        suggested = {}
        if uploaded_file:
            suggested = suggest_metadata_from_filename(uploaded_file.name)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            m_doc_type = st.selectbox(
                "Doc Type",
                ["handbook", "circular", "policy", "other"],
                index=["handbook", "circular", "policy", "other"].index(
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
        m_section = st.text_input("Section (optional)", key="ingest_section")

        btn_check, btn_ingest = st.columns(2)

        with btn_check:
            if st.button(
                "🔎  Check for Duplicates",
                disabled=not uploaded_file,
                key="check_dup",
            ):
                result = agent.detect_duplicates(
                    {"topic": m_topic, "department": m_dept, "year": m_year}
                )
                if "NO_DUPLICATES" in result:
                    st.success(result)
                else:
                    st.warning(result)

        with btn_ingest:
            if st.button(
                "⬆️  Ingest Document",
                type="primary",
                disabled=not uploaded_file,
                key="ingest_btn",
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
                        "version": m_version,
                        "section": m_section,
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

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — ADMIN AI AGENT
# ═══════════════════════════════════════════════════════════════════════════

with tab_agent:
    st.markdown(
        '<div class="section-heading">Admin AI Agent</div>'
        '<div class="section-caption">'
        "Manage your knowledge base through natural-language conversation. "
        "Requires <code>GROQ_API_KEY</code> in your environment."
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

        # Display chat history
        for msg in st.session_state.agent_messages:
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])

        # Input
        user_input = st.chat_input("Ask the Admin Agent… (e.g. 'Show KB stats')")
        if user_input:
            st.session_state.agent_messages.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Agent thinking…"):
                    response = agent_inst.run(user_input)
                st.markdown(response)

            st.session_state.agent_messages.append(
                {"role": "assistant", "content": response}
            )

        col_clear, col_space = st.columns([1, 3])
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.agent_messages = []
                st.rerun()

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
