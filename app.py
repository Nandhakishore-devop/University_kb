"""
app.py - University Knowledge Base: Streamlit Application Entry Point.

Tabs:
  1. 🎓 Student Search Panel
  2. 🛠️  Admin Panel
  3. 🤖  Admin AI Agent (optional, requires OPENAI_API_KEY)
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
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d3250;
    }

    /* Card-like containers */
    .chunk-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-left: 4px solid #6c63ff;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .chunk-card.internal {
        border-left-color: #ff6b6b;
    }
    .chunk-meta {
        font-size: 0.78rem;
        color: #8892b0;
        margin-top: 8px;
    }
    .score-badge {
        display: inline-block;
        background: #6c63ff22;
        color: #a29bfe;
        border: 1px solid #6c63ff55;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .access-badge-public {
        display: inline-block;
        background: #00b89422;
        color: #00b894;
        border: 1px solid #00b89455;
        border-radius: 20px;
        padding: 2px 8px;
        font-size: 0.72rem;
    }
    .access-badge-internal {
        display: inline-block;
        background: #ff6b6b22;
        color: #ff6b6b;
        border: 1px solid #ff6b6b55;
        border-radius: 20px;
        padding: 2px 8px;
        font-size: 0.72rem;
    }
    .stat-box {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stat-number { font-size: 2rem; font-weight: 700; color: #6c63ff; }
    .stat-label { font-size: 0.85rem; color: #8892b0; margin-top: 4px; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e2130 0%, #2d3250 100%);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border: 1px solid #3d4270;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Singleton resources (cached to avoid re-creating on every rerun)
# ---------------------------------------------------------------------------

CHROMA_DIR = str(Path(__file__).parent / "chroma_db")


@st.cache_resource
def get_store() -> ChromaStore:
    use_openai = bool(os.getenv("OPENAI_API_KEY"))
    return ChromaStore(persist_dir=CHROMA_DIR, use_openai=use_openai)


@st.cache_resource
def get_retriever() -> UniversityRetriever:
    return UniversityRetriever(store=get_store())


@st.cache_resource
def get_agent() -> AdminAgent:
    return AdminAgent(store=get_store())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎓 University KB")
    st.markdown("---")

    store = get_store()
    stats = store.get_stats()

    st.markdown(f"**Total Chunks:** `{stats.total_chunks}`")
    st.markdown(f"**Sources:** `{stats.unique_sources}`")
    if stats.year_range[0]:
        st.markdown(f"**Years:** `{stats.year_range[0]} – {stats.year_range[1]}`")

    st.markdown("---")
    st.markdown("**Topics**")
    if stats.unique_topics:
        for t in stats.unique_topics[:10]:
            st.markdown(f"• `{t}`")
    else:
        st.caption("No data yet. Ingest documents via Admin Panel.")

    st.markdown("---")
    openai_set = bool(os.getenv("OPENAI_API_KEY"))
    if openai_set:
        st.success("🔑 OpenAI key detected — using OpenAI embeddings + AI Agent")
    else:
        st.info("🔑 No OpenAI key — using local embeddings (sentence-transformers)")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="main-header">
        <h1 style="margin:0;color:#e0e0ff">🎓 University Knowledge Base</h1>
        <p style="margin:4px 0 0;color:#8892b0">
            ChromaDB · LangChain · Streamlit — Intelligent Document Retrieval System
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_student, tab_admin, tab_agent = st.tabs(
    ["🎓 Student Search", "🛠️ Admin Panel", "🤖 Admin AI Agent"]
)

# ===========================================================================
# TAB 1: STUDENT SEARCH
# ===========================================================================

with tab_student:
    st.markdown("### Search University Documents")
    st.caption(
        "Search across handbooks, circulars, and policies. "
        "Only public documents are shown here."
    )

    retriever = get_retriever()

    col_q, col_k = st.columns([5, 1])
    with col_q:
        query = st.text_input(
            "Your question or keywords",
            placeholder="e.g. What are the rules for examinations?",
            label_visibility="collapsed",
        )
    with col_k:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

    st.markdown("**Filters** (optional)")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    # Populate filter options from KB
    all_topics = [""] + stats.unique_topics
    all_depts = [""] + stats.unique_departments
    all_years = [""] + (
        [str(y) for y in range(stats.year_range[1], stats.year_range[0] - 1, -1)]
        if stats.year_range[0]
        else []
    )

    with f_col1:
        f_topic = st.selectbox("Topic", all_topics)
    with f_col2:
        f_dept = st.selectbox("Department", all_depts)
    with f_col3:
        f_year = st.selectbox("Year", all_years)
    with f_col4:
        f_doctype = st.selectbox("Doc Type", ["", "handbook", "circular", "policy", "other"])

    search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

    if search_btn and query.strip():
        filters = SearchFilter(
            topic=f_topic or None,
            department=f_dept or None,
            year=int(f_year) if f_year else None,
            doc_type=f_doctype or None,
            access="public",  # Always public for students
        )

        with st.spinner("Searching…"):
            results = retriever.search(
                query=query,
                filters=filters,
                top_k=top_k,
                admin_override=False,
            )

        if not results:
            st.warning("No matching documents found. Try broader search terms or different filters.")
        else:
            st.success(f"Found **{len(results)}** relevant chunks")

            for i, chunk in enumerate(results, 1):
                access = chunk.metadata.get("access", "public")
                card_cls = "chunk-card" + (" internal" if access == "internal" else "")
                access_badge = (
                    f'<span class="access-badge-public">public</span>'
                    if access == "public"
                    else f'<span class="access-badge-internal">internal</span>'
                )
                score_pct = f"{chunk.score * 100:.1f}%"

                st.markdown(
                    f"""
                    <div class="{card_cls}">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                            <strong style="color:#e0e0ff">#{i} · {chunk.metadata.get('source_file','')}</strong>
                            <span>
                                {access_badge}
                                &nbsp;<span class="score-badge">Score: {score_pct}</span>
                            </span>
                        </div>
                        <p style="color:#ccd6f6;line-height:1.6;margin:0 0 8px">{chunk.content[:800]}</p>
                        <div class="chunk-meta">
                            📁 <b>Topic:</b> {chunk.metadata.get('topic','')} &nbsp;|&nbsp;
                            🏛️ <b>Dept:</b> {chunk.metadata.get('department','')} &nbsp;|&nbsp;
                            📅 <b>Year:</b> {chunk.metadata.get('year','')} &nbsp;|&nbsp;
                            📄 <b>Type:</b> {chunk.metadata.get('doc_type','')} &nbsp;|&nbsp;
                            🔖 <b>Version:</b> {chunk.metadata.get('version','')} &nbsp;|&nbsp;
                            📃 <b>Page:</b> {chunk.metadata.get('page_number','')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    elif search_btn:
        st.warning("Please enter a search query.")

# ===========================================================================
# TAB 2: ADMIN PANEL
# ===========================================================================

with tab_admin:
    agent = get_agent()

    st.markdown("### Admin Panel")
    st.caption("Ingest, manage, and export documents in the knowledge base.")

    # ------------------------------------------------------------------
    # Section A: KB Stats
    # ------------------------------------------------------------------
    with st.expander("📊 Knowledge Base Statistics", expanded=True):
        s = get_store().get_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f'<div class="stat-box"><div class="stat-number">{s.total_chunks}</div>'
            f'<div class="stat-label">Total Chunks</div></div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="stat-box"><div class="stat-number">{s.unique_sources}</div>'
            f'<div class="stat-label">Unique Sources</div></div>',
            unsafe_allow_html=True,
        )
        c3.markdown(
            f'<div class="stat-box"><div class="stat-number">{len(s.unique_topics)}</div>'
            f'<div class="stat-label">Topics</div></div>',
            unsafe_allow_html=True,
        )
        c4.markdown(
            f'<div class="stat-box"><div class="stat-number">{len(s.unique_departments)}</div>'
            f'<div class="stat-label">Departments</div></div>',
            unsafe_allow_html=True,
        )

        if s.doc_type_counts:
            st.markdown("**Doc type breakdown:**")
            st.json(s.doc_type_counts)

        if st.button("🔄 Refresh Stats"):
            st.cache_resource.clear()
            st.rerun()

    # ------------------------------------------------------------------
    # Section B: Ingest / Upsert Document
    # ------------------------------------------------------------------
    with st.expander("📤 Ingest / Upsert Document", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload document (PDF, DOCX, HTML)",
            type=["pdf", "docx", "html", "htm"],
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
                "Topic",
                value=suggested.get("topic", ""),
                key="ingest_topic",
            )
            m_year = st.number_input(
                "Year",
                min_value=2000,
                max_value=2100,
                value=int(suggested.get("year", 2024)),
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

        col_ingest1, col_ingest2 = st.columns(2)

        with col_ingest1:
            if st.button("🔎 Check for Duplicates", disabled=not uploaded_file):
                meta_dict = {
                    "topic": m_topic,
                    "department": m_dept,
                    "year": m_year,
                }
                dup_result = agent.detect_duplicates(meta_dict)
                if "NO_DUPLICATES" in dup_result:
                    st.success(dup_result)
                else:
                    st.warning(dup_result)

        with col_ingest2:
            if st.button("⬆️ Ingest Document", type="primary", disabled=not uploaded_file):
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

                    if "SUCCESS" in result:
                        st.success(result)
                        st.cache_resource.clear()
                    else:
                        st.error(result)

    # ------------------------------------------------------------------
    # Section C: Delete by Metadata
    # ------------------------------------------------------------------
    with st.expander("🗑️ Delete Documents by Metadata", expanded=False):
        st.warning(
            "⚠️ This permanently removes chunks from the KB. Use with caution."
        )

        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            d_topic = st.text_input("Topic to delete", key="del_topic")
            d_year = st.number_input(
                "Year (0 = any)", min_value=0, max_value=2100, value=0, key="del_year"
            )
        with d_col2:
            d_dept = st.text_input("Department", key="del_dept")
            d_version = st.text_input("Version", key="del_version")
        with d_col3:
            d_access = st.selectbox(
                "Access", ["", "public", "internal"], key="del_access"
            )
            d_doctype = st.selectbox(
                "Doc Type", ["", "handbook", "circular", "policy", "other"], key="del_doctype"
            )

        confirm_delete = st.checkbox("I confirm I want to delete these chunks")

        if st.button("🗑️ Delete", type="secondary", disabled=not confirm_delete):
            filters: dict = {}
            if d_topic:
                filters["topic"] = d_topic
            if d_year:
                filters["year"] = d_year
            if d_dept:
                filters["department"] = d_dept
            if d_version:
                filters["version"] = d_version
            if d_access:
                filters["access"] = d_access
            if d_doctype:
                filters["doc_type"] = d_doctype

            if not filters:
                st.error("Please specify at least one filter before deleting.")
            else:
                with st.spinner("Deleting…"):
                    result = agent.delete_by_metadata(filters)
                if "SUCCESS" in result:
                    st.success(result)
                    st.cache_resource.clear()
                else:
                    st.error(result)

    # ------------------------------------------------------------------
    # Section D: Export Metadata
    # ------------------------------------------------------------------
    with st.expander("💾 Export Metadata Backup", expanded=False):
        all_meta = get_store().get_all_metadata()

        if not all_meta:
            st.info("No data to export yet.")
        else:
            df = pd.DataFrame(all_meta)

            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                json_bytes = json.dumps(all_meta, indent=2).encode()
                st.download_button(
                    "⬇️ Download JSON",
                    data=json_bytes,
                    file_name="kb_metadata_backup.json",
                    mime="application/json",
                )
            with exp_col2:
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_buf.getvalue().encode(),
                    file_name="kb_metadata_backup.csv",
                    mime="text/csv",
                )

            st.dataframe(df, use_container_width=True, height=300)

    # ------------------------------------------------------------------
    # Section E: Reindex Recommendation
    # ------------------------------------------------------------------
    with st.expander("🔄 Reindex Recommendation", expanded=False):
        if st.button("Analyse KB Health"):
            rec = agent.recommend_reindex()
            if "RECOMMENDATION" in rec.upper():
                st.warning(rec)
            else:
                st.success(rec)

# ===========================================================================
# TAB 3: ADMIN AI AGENT
# ===========================================================================

with tab_agent:
    st.markdown("### 🤖 Admin AI Agent")
    st.caption(
        "Chat with the AI agent to manage your KB using natural language. "
        "Requires `OPENAI_API_KEY` to be set in your environment."
    )

    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "⚠️ `OPENAI_API_KEY` is not set. "
            "The AI Agent is disabled. "
            "You can still use all features in the **Admin Panel** tab manually."
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

        if st.button("🗑️ Clear Chat"):
            st.session_state.agent_messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("**Example prompts:**")
        examples = [
            "Show me the current KB statistics.",
            "Check for duplicate exam rules documents.",
            "Are there any reindexing recommendations?",
            "What topics are available in the knowledge base?",
            "Delete all chunks from year 2022 with topic exam_rules.",
        ]
        for ex in examples:
            st.markdown(f"- *{ex}*")
