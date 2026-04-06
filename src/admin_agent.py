"""
admin_agent.py - LangChain-powered Admin Assistant for the University KB.

Exposes structured Tools that an LLM agent (or a human admin via the Streamlit
UI) can call to manage the knowledge base.

Tools:
  kb_stats_tool            - Return KB statistics
  upsert_doc_tool          - Ingest / upsert a document
  delete_by_metadata_tool  - Delete chunks by filter
  detect_duplicates_tool   - Warn about duplicate/outdated versions
  recommend_reindex_tool   - Suggest re-indexing when needed
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage
from loguru import logger

from src.chroma_store import ChromaStore
from src.ingestion import chunk_document
from src.schemas import DocMetadata, SearchFilter
from src.utils import suggest_metadata_from_filename


def _get_groq_model(api_key: str) -> str:
    """Get the Groq model to use.
    
    Priority:
    1. GROQ_MODEL environment variable (if set)
    2. Default to current recommended model
    
    Available models: Check https://console.groq.com/docs/models
    """
    # Check for user-configured model
    configured_model = os.getenv("GROQ_MODEL")
    if configured_model:
        return configured_model
    
    # Updated model (check console.groq.com for latest)
    # As of 2026, commonly available models include:
    # - llama-3.3-70b-versatile
    # - llama-3.1-405b-reasoning  
    # - gemma-2-9b-it
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------


def _kb_stats(store: ChromaStore) -> str:
    stats = store.get_stats()
    return json.dumps(
        {
            "total_chunks": stats.total_chunks,
            "unique_sources": stats.unique_sources,
            "topics": stats.unique_topics,
            "departments": stats.unique_departments,
            "doc_type_counts": stats.doc_type_counts,
            "year_range": list(stats.year_range),
        },
        indent=2,
    )


def _upsert_doc(
    store: ChromaStore,
    file_path: str,
    metadata_dict: dict,
) -> str:
    """Load, chunk, and upsert a document into ChromaDB."""
    try:
        metadata = DocMetadata(**metadata_dict)
        chunks = chunk_document(file_path, metadata)
        if not chunks:
            file_ext = Path(file_path).suffix.lower()
            if file_ext == ".pdf":
                return (
                    "❌ PDF Extraction Failed\n\n"
                    "This is a scanned/image-based PDF with no selectable text.\n\n"
                    "✅ SOLUTION 1: Enable OCR (recommended)\n"
                    "   - See OCR_SETUP.md for detailed instructions\n"
                    "   - pip install pytesseract Pillow\n"
                    "   - Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "   - Re-upload the PDF (automatic OCR will process it)\n\n"
                    "✅ SOLUTION 2: Pre-convert the PDF\n"
                    "   - Visit: https://www.ilovepdf.com/ocr_pdf\n"
                    "   - Upload scanned PDF → Download OCR'd PDF\n"
                    "   - Upload the converted PDF here\n\n"
                    "✅ SOLUTION 3: Re-export PDF\n"
                    "   - Open source document in original format\n"
                    "   - Export/Print as PDF with text layer"
                )
            else:
                return f"❌ ERROR: No text extracted from {Path(file_path).name}. The file may be empty or corrupted."
        n = store.upsert_chunks(chunks)
        return f"✅ SUCCESS: Upserted {n} chunks from '{Path(file_path).name}' (v{metadata.version})"
    except Exception as e:
        logger.error(f"upsert_doc_tool error: {e}")
        return f"❌ ERROR: {e}"


def _delete_by_metadata(store: ChromaStore, filters: dict) -> str:
    """Delete chunks matching the provided metadata filters."""
    try:
        sf = SearchFilter(**filters)
        n = store.delete_by_filter(sf)
        return f"SUCCESS: Deleted {n} chunks matching filters {filters}"
    except ValueError as e:
        return f"ERROR: {e}"


def _detect_duplicates(store: ChromaStore, new_doc_metadata: dict) -> str:
    """
    Check for existing docs with same topic+department.
    Also performs a semantic check if topic is provided.
    """
    topic = new_doc_metadata.get("topic", "")
    department = new_doc_metadata.get("department", "GENERAL")
    year = int(new_doc_metadata.get("year", 0))

    # 1. Metadata-based check
    existing = store.find_similar_docs(topic, department, year)
    
    warnings = []
    found_metadata_dup = False
    
    if existing:
        found_metadata_dup = True
        for m in existing:
            ex_year = m.get("year", 0)
            ex_ver = m.get("version", "?")
            ex_src = m.get("source_file", "?")
            if ex_year > year:
                warnings.append(
                    f"NEWER VERSION EXISTS (Metadata): '{ex_src}' (year={ex_year}, v={ex_ver}) "
                    f"is newer than the doc you're uploading (year={year})."
                )
            elif ex_year == year:
                warnings.append(
                    f"SAME YEAR CONFLICT (Metadata): '{ex_src}' (v={ex_ver}) has same year."
                )
            else:
                warnings.append(
                    f"OLDER VERSION (Metadata): '{ex_src}' (year={ex_year}, v={ex_ver}) already exists."
                )

    # 2. Semantic-based check (if topic is long enough to be a query)
    if topic and len(topic) > 3:
        semantic_results = store.similarity_search(query=topic, top_k=3)
        for res in semantic_results:
            # If high similarity (> 0.85) but different filename/topic metadata
            if res.score > 0.85:
                src = res.metadata.get("source_file", "?")
                if src != new_doc_metadata.get("source_file"):
                    warnings.append(
                        f"SEMANTIC DUPLICATE: Content in '{src}' is very similar to your topic '{topic}' "
                        f"(Confidence: {res.score*100:.1f}%). Please verify if this is a duplicate."
                    )
                    break

    if not warnings:
        return "NO_DUPLICATES: No existing docs found with same topic and department."

    return "\n".join(warnings)


def _detect_conflicts(store: ChromaStore, topic: str) -> str:
    """
    Retrieve chunks for a topic and ask an LLM to identify any 
    contradictory information (e.g. different dates/rules for the same thing).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "ERROR: Conflict detection requires GROQ_API_KEY to be set in .env file."

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        # 1. Retrieve everything for this topic
        results = store.similarity_search(query=topic, top_k=15)
        if len(results) < 2:
            return "NO_CONFLICTS: Not enough documents in the KB for this topic to compare."

        # 2. Prepare for LLM
        context_text = "\n\n".join(
            [f"--- {r.metadata.get('source_file')} (v{r.metadata.get('version')}, Year {r.metadata.get('year')}) ---\n{r.content}" 
             for r in results]
        )

        llm = ChatOpenAI(
            model="mixtral-8x7b-32768",
            temperature=0.0,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a Conflict Detection Assistant. "
                "Your task is to identify contradictory information in the provided university documents. "
                "Look for inconsistencies in dates, rules, eligibility, or policies for the same topic. "
                "If no conflicts are found, say 'NO_CONFLICTS_FOUND'. "
                "If conflicts are found, list them clearly with the source document names."
            )),
            ("human", "Topic: {topic}\n\nDocuments:\n{context}")
        ])

        chain = prompt | llm
        response = chain.invoke({"context": context_text, "topic": topic})
        return response.content
    except Exception as e:
        return f"ERROR: Conflict detection failed: {e}"


def _recommend_reindex(store: ChromaStore) -> str:
    """Analyse KB health and suggest re-indexing if warranted."""
    stats = store.get_stats()
    recommendations = []

    if stats.total_chunks == 0:
        return "KB is empty. Ingest documents to get started."

    if stats.total_chunks > 10_000:
        recommendations.append(
            f"Large KB ({stats.total_chunks} chunks). "
            "Consider archiving old versions to keep retrieval fast."
        )

    # Check for potential duplicates: multiple sources with same topic
    all_meta = store.get_all_metadata()
    topic_sources: dict[str, set] = {}
    for m in all_meta:
        t = m.get("topic", "")
        s = m.get("source_file", "")
        if t:
            topic_sources.setdefault(t, set()).add(s)

    dup_topics = [t for t, srcs in topic_sources.items() if len(srcs) > 1]
    if dup_topics:
        recommendations.append(
            f"Topics with multiple source files (possible duplicates): "
            f"{', '.join(dup_topics)}. "
            "Run detect_duplicates_tool for each to verify."
        )

    if not recommendations:
        return (
            f"KB looks healthy: {stats.total_chunks} chunks, "
            f"{stats.unique_sources} sources, no obvious duplication issues."
        )
    return "RECOMMENDATIONS:\n" + "\n".join(f"  • {r}" for r in recommendations)


# ---------------------------------------------------------------------------
# AdminAgent: wraps tools and optionally connects an LLM
# ---------------------------------------------------------------------------


class AdminAgent:
    """
    Provides LangChain StructuredTools for KB management.

    If GROK_API_KEY is set, also builds a ReAct agent that can chain
    tool calls based on natural-language admin instructions.
    """

    def __init__(self, store: ChromaStore) -> None:
        self._store = store
        self.tools = self._build_tools()
        self._agent = None  # Lazy-init

    # ------------------------------------------------------------------
    # Tool construction
    # ------------------------------------------------------------------

    def _build_tools(self) -> list:
        store = self._store

        kb_stats = StructuredTool.from_function(
            func=lambda: _kb_stats(store),
            name="kb_stats_tool",
            description="Return JSON statistics about the knowledge base: chunk count, topics, departments, doc types.",
        )

        upsert_doc = StructuredTool.from_function(
            func=lambda file_path, metadata_json: _upsert_doc(
                store, file_path, json.loads(metadata_json)
            ),
            name="upsert_doc_tool",
            description=(
                "Ingest or update a document in the KB. "
                "Args: file_path (str), metadata_json (JSON string with keys: "
                "source_file, doc_type, topic, year, department, access, version, section)."
            ),
        )

        delete_meta = StructuredTool.from_function(
            func=lambda filters_json: _delete_by_metadata(
                store, json.loads(filters_json)
            ),
            name="delete_by_metadata_tool",
            description=(
                "Delete KB chunks matching metadata filters. "
                "Args: filters_json (JSON string; valid keys: topic, year, department, "
                "access, doc_type, version). At least one filter required."
            ),
        )

        detect_dups = StructuredTool.from_function(
            func=lambda metadata_json: _detect_duplicates(
                store, json.loads(metadata_json)
            ),
            name="detect_duplicates_tool",
            description=(
                "Check for existing docs with the same topic and department. "
                "Args: metadata_json (JSON with at least: topic, department, year)."
            ),
        )

        reindex = StructuredTool.from_function(
            func=lambda: _recommend_reindex(store),
            name="recommend_reindex_tool",
            description="Analyse KB health and recommend re-indexing or cleanup actions.",
        )

        detect_conflicts = StructuredTool.from_function(
            func=lambda topic: _detect_conflicts(store, topic),
            name="detect_conflicts_tool",
            description=(
                "Check for contradictory information (conflicts) within a topic. "
                "Args: topic (str)."
            ),
        )

        return [kb_stats, upsert_doc, delete_meta, detect_dups, reindex, detect_conflicts]

    # ------------------------------------------------------------------
    # Direct tool calls (no LLM required)
    # ------------------------------------------------------------------

    def kb_stats(self) -> str:
        return _kb_stats(self._store)

    def upsert_doc(self, file_path: str, metadata_dict: dict) -> str:
        return _upsert_doc(self._store, file_path, metadata_dict)

    def delete_by_metadata(self, filters: dict) -> str:
        return _delete_by_metadata(self._store, filters)

    def detect_duplicates(self, metadata_dict: dict) -> str:
        return _detect_duplicates(self._store, metadata_dict)

    def detect_conflicts(self, topic: str) -> str:
        return _detect_conflicts(self._store, topic)

    def recommend_reindex(self) -> str:
        return _recommend_reindex(self._store)

    def suggest_metadata(self, filename: str) -> dict:
        """Return heuristic metadata suggestion for a given filename."""
        return suggest_metadata_from_filename(filename)

    # ------------------------------------------------------------------
    # LLM agent (optional — requires GROK_API_KEY)
    # ------------------------------------------------------------------

    def _get_llm_agent(self):
        """Lazy-init a ReAct agent backed by Mixtral via Groq."""
        if self._agent:
            return self._agent

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set in .env file. LLM Admin Agent will not be available.")
            return None

        try:
            from langchain_openai import ChatOpenAI
            from langchain.agents import AgentExecutor
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langgraph.prebuilt import create_react_agent

            model = _get_groq_model(api_key)
            
            llm = ChatOpenAI(
                model=model,
                temperature=0.0,
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
            )

            # System prompt
            system_prompt = (
                "You are the University Knowledge Base Admin Assistant. "
                "You help administrators manage university documents stored in ChromaDB. "
                "Available tools:\n"
                "  • kb_stats_tool: Get KB statistics\n"
                "  • upsert_doc_tool: Ingest/update a document\n"
                "  • delete_by_metadata_tool: Delete chunks by filter\n"
                "  • detect_duplicates_tool: Find duplicate documents\n"
                "  • detect_conflicts_tool: Find contentual conflicts\n"
                "  • recommend_reindex_tool: Get system recommendations\n"
                "Always check for duplicates before upserting. "
                "Require explicit confirmation before any deletion. "
                "Suggest metadata based on filenames when not provided."
            )

            # Create agent using LangGraph (more modern approach)
            self._agent = create_react_agent(llm, self.tools, state_modifier=system_prompt)
            logger.info("✅ LLM Admin Agent initialised (Mixtral-8x7b via Groq)")
            return self._agent

        except ImportError:
            # Fallback: use simple LLM-Tools binding
            try:
                from langchain_openai import ChatOpenAI
                
                model = _get_groq_model(api_key)
                
                llm = ChatOpenAI(
                    model=model,
                    temperature=0.0,
                    base_url="https://api.groq.com/openai/v1",
                    api_key=api_key,
                )
                
                # Bind tools to LLM
                self._agent = llm.bind_tools(self.tools)
                logger.info("✅ LLM Admin Agent initialised (tool-binding mode)")
                return self._agent
                
            except Exception as e2:
                logger.error(f"Could not init LLM agent (fallback also failed): {e2}")
                return None
                
        except Exception as e:
            logger.error(f"Could not init LLM agent: {e}")
            return None

    def run(self, instruction: str) -> str:
        """
        Run an admin instruction through the LLM agent if available,
        otherwise return a message asking for API key.
        """
        agent = self._get_llm_agent()
        if agent is None:
            return (
                "⚠️ LLM Admin Agent requires GROQ_API_KEY to be set in .env file.\n\n"
                "✅ To enable:\n"
                "  1. Go to https://console.groq.com\n"
                "  2. Create an API key\n"
                "  3. Add to .env: GROQ_API_KEY=your_key_here\n"
                "  4. Restart the app\n\n"
                "💡 You can still use the manual Admin Panel tools below without the LLM."
            )
        try:
            from langchain_core.messages import HumanMessage
            
            # Handle tool-bound LLM (simple invoke)
            if hasattr(agent, 'bind_tools'):
                response = agent.invoke([HumanMessage(content=instruction)])
                if hasattr(response, 'content'):
                    return response.content
                return str(response)
            
            # Handle ReAct agent (needs dict input)
            elif hasattr(agent, 'invoke'):
                result = agent.invoke({"input": instruction})
                if isinstance(result, dict):
                    return result.get("output", str(result))
                return str(result)
            else:
                return "Error: Unknown agent type"
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Agent error: {e}"
