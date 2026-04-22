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
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage
from loguru import logger

from src.chroma_store import ChromaStore
from src.ingestion import chunk_document
from src.schemas import DocMetadata, SearchFilter, FileIngestionStatus, IngestionJobReport
from src.utils import suggest_metadata_from_filename


# ---------------------------------------------------------------------------
# Integrity Guardrails Configuration
# ---------------------------------------------------------------------------
DUPLICATE_SIMILARITY_THRESHOLD = 0.95
SIMILAR_TEXT_THRESHOLD = 0.85
# ---------------------------------------------------------------------------



def _check_exact_duplicate(store: ChromaStore, new_chunks: list) -> dict:
    """
    Check whether the document is a global duplicate of existing KB content.
    Calculates a document-level similarity percentage.
    """
    if not new_chunks:
        return {"blocked": False, "reason": "", "matches": []}

    match_count = 0
    matches = []
    total_chunks = len(new_chunks)
    
    # We only check up to 20 chunks for performance on large docs
    sample_size = min(total_chunks, 20)
    step = max(1, total_chunks // sample_size)
    sampled_chunks = new_chunks[::step][:sample_size]

    for chunk in sampled_chunks:
        content = chunk.get("text", "") if isinstance(chunk, dict) else (
            getattr(chunk, "page_content", "") or getattr(chunk, "content", "")
        )
        if not content.strip():
            continue

        results = store.similarity_search(query=content, top_k=1, include_all_statuses=True)
        if results and results[0].score >= DUPLICATE_SIMILARITY_THRESHOLD:
            match_count += 1
            top = results[0]
            matches.append({
                "source": top.metadata.get("source_file", "?"),
                "score": round(top.score, 4),
            })

    if not matches:
        return {"blocked": False, "reason": "", "matches": []}

    # Document-level similarity percentage
    similarity_pct = (match_count / len(sampled_chunks)) * 100
    sources = list({m["source"] for m in matches})
    
    is_blocked = similarity_pct >= 75.0  # Block if >75% of content matches
    
    status_msg = "🚫 BLOCKED: Global Duplicate" if is_blocked else "⚠️ WARNING: Heavy Content Overlap"
    reason = (
        f"{status_msg} ({round(similarity_pct, 1)}% identical).\n"
        f"This document is extremely similar to existing sources: {', '.join(sources)}.\n"
    )
    
    if is_blocked:
        reason += "Ingestion blocked to maintain KB integrity and prevent redundancy."
    else:
        reason += "Consider if this should be a new version instead of a new document."

    return {"blocked": is_blocked, "reason": reason, "matches": matches, "similarity_pct": similarity_pct}


def _check_similar_text(store: ChromaStore, new_chunks: list, source_file: str) -> dict:
    """
    Check whether the new document covers the same topic area as existing KB content.
    """
    if not new_chunks:
        return {"blocked": False, "reason": "", "matches": [], "has_similar": False}

    similar_hits: dict[str, float] = {}

    for chunk in new_chunks:
        content = chunk.get("text", "") if isinstance(chunk, dict) else (
            getattr(chunk, "page_content", "") or getattr(chunk, "content", "")
        )
        if not content.strip():
            continue

        results = store.similarity_search(query=content, top_k=3, include_all_statuses=True)
        for res in results:
            src = res.metadata.get("source_file", "?")
            if src == source_file:
                continue
            if SIMILAR_TEXT_THRESHOLD <= res.score < DUPLICATE_SIMILARITY_THRESHOLD:
                if src not in similar_hits or res.score > similar_hits[src]:
                    similar_hits[src] = res.score

    if not similar_hits:
        return {"blocked": False, "reason": "", "matches": [], "has_similar": False}

    matches = [{"source": s, "max_score": round(v, 4)} for s, v in similar_hits.items()]
    sources_str = ", ".join(
        "'" + m["source"] + "' (" + str(round(m["max_score"] * 100, 1)) + "%)"
        for m in matches
    )
    reason = (
        "SIMILAR TOPIC DETECTED: The new document covers the same topic area as "
        "existing KB sources: "
        + sources_str
        + ". Upload blocked to prevent topic overlap. "
        "If this document contains updated information, use supersede_document_tool instead."
    )
    return {"blocked": True, "reason": reason, "matches": matches, "has_similar": True}


def _check_contradictions(store: ChromaStore, new_chunks: list, metadata_dict: dict, api_key: str) -> dict:
    """
    Use an LLM to compare the new document's content against existing truth and flag contradictions.
    """
    if not api_key:
        return {
            "blocked": False,
            "reason": "Contradiction check skipped (GROQ_API_KEY not set).",
            "llm_report": "",
        }

    if not new_chunks:
        return {"blocked": False, "reason": "", "llm_report": ""}

    source_file = metadata_dict.get("source_file", "unknown")
    topic = metadata_dict.get("topic", "")
    dept = metadata_dict.get("department", "")

    # Context collection: BROADEN SEARCH to catch cross-year/cross-dept conflicts
    # 1. Search by topic + department
    # 2. Search by key text snippets from new doc
    new_text_sample = "\n\n".join([
        c.get("text", "") if isinstance(c, dict) else (getattr(c, "page_content", "") or getattr(c, "content", ""))
        for c in new_chunks[:10]
    ])[:8000]

    # Perform multiple global searches
    search_queries = [topic, f"{dept} {topic}", new_text_sample[:150], new_text_sample[500:650]]
    existing_results = []
    seen_ids = set()

    for q in search_queries:
        if not q: continue
        hits = store.similarity_search(query=q, top_k=5, include_all_statuses=True)
        for h in hits:
            # Skip current file if it's already in (unlikely in pre-ingest, but safe)
            if h.metadata.get("source_file") == source_file:
                continue
            h_id = f"{h.metadata.get('source_file')}_{h.metadata.get('version')}_{h.content[:50]}"
            if h_id not in seen_ids:
                existing_results.append(h)
                seen_ids.add(h_id)

    if not existing_results:
        return {"blocked": False, "reason": "No existing content to compare.", "llm_report": ""}

    # Limit context size for LLM budget
    existing_text = "\n\n".join([
        f"--- SOURCE: {r.metadata.get('source_file')} (Dept: {r.metadata.get('department')} Year: {r.metadata.get('year')} v{r.metadata.get('version')}) --- \n{r.content}"
        for r in existing_results[:12]
    ])[:10000]

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(
            model=_get_groq_model(api_key),
            temperature=0.0,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a University Knowledge Base Integrity Auditor.\n"
                "Your goal is to detect DIRECT FACTUAL CONTRADICTIONS or NEAR-DUPLICATE Policies across years/departments.\n\n"
                "RULES:\n"
                "1. If a 2024 policy says 'Fee is $100' and a 2025 policy says 'Fee is $120', this is a CHRONOLOGICAL UPDATE, not a contradiction. Flag it as 'CHRONOLOGICAL_UPDATE'.\n"
                "2. If two policies for the same year/context give different rules, flag as 'SEVERE_CONTRADICTION'.\n"
                "3. If a new policy is 90% identical to an existing policy in another department but with minor changes, flag as 'CROSS_DEPARTMENT_REDUNDANCY'.\n\n"
                "Respond ONLY with a JSON object: { \"has_issue\": bool, \"issue_type\": \"CONTRADICTION\"|\"REDUNDANCY\"|\"UPDATE\", \"severity\": \"HIGH\"|\"MEDIUM\"|\"LOW\", \"details\": [ { \"topic\": str, \"new_says\": str, \"existing_says\": str, \"target_source\": str } ], \"recommendation\": str }"
            )),
            ("human", f"NEW DOCUMENT (Target: {dept} {topic}):\n{new_text_sample}\n\nEXISTING GLOBAL KB CONTENT:\n{existing_text}"),
        ])

        response = llm.invoke(prompt.format_messages(new_text_sample=new_text_sample, existing_text=existing_text))
        raw = response.content.strip()

        # Simple json extraction
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
             raw = match.group(0)
        parsed = json.loads(raw)

        has_issue = parsed.get("has_issue", False)
        severity = parsed.get("severity", "LOW")
        issue_type = parsed.get("issue_type", "NONE")
        
        # Blocking logic: Severe contradictions or High-severity redundancies block upload
        should_block = has_issue and (severity == "HIGH" or (severity == "MEDIUM" and issue_type == "CONTRADICTION"))

        if not has_issue:
            reason = "Integrity scan clean."
        else:
            lines = [f"- [{d.get('target_source', '?')}] {d.get('topic', '?')}: {d.get('new_says', '?')} VS {d.get('existing_says', '?')}"
                     for d in parsed.get("details", [])]
            block_msg = "BLOCKED." if should_block else "WARNING."
            reason = f"INTEGRITY {issue_type} (Severity: {severity}) — {block_msg}\n" + "\n".join(lines)

        return {"blocked": should_block, "reason": reason, "llm_report": raw}

    except Exception as e:
        logger.error(f"Global Integrity Audit Exception: {e}")
        return {"blocked": False, "reason": f"Audit error: {e}", "llm_report": ""}


def _run_upload_guardrails(
    store: ChromaStore,
    new_chunks: list,
    metadata_dict: dict,
    api_key: str,
) -> tuple[bool, str]:
    """Sequence Pre-ingest integrity checks."""
    source_file = metadata_dict.get("source_file", "unknown")
    topic = metadata_dict.get("topic", "")

    # 1. Exact Duplicate
    dup = _check_exact_duplicate(store, new_chunks)
    if dup["blocked"]:
        return True, "🚫 [CHECK 1 – DUPLICATE DETECTED]\n" + dup["reason"]

    # 2. Topic/Contradiction Audit
    sim = _check_similar_text(store, new_chunks, source_file)
    contra = _check_contradictions(store, new_chunks, metadata_dict, api_key)

    if contra["blocked"]:
        report = "🚫 [CHECK 3 – CONFLICT DETECTED]\n" + contra["reason"]
        if sim["blocked"]:
            report += "\n\nℹ️ [CHECK 2 – SIMILAR TOPIC context]\n" + sim["reason"]
        return True, report

    if sim["blocked"]:
        return True, "🚫 [CHECK 2 – SIMILAR TOPIC DETECTED]\n" + sim["reason"]

    # Warnings only
    warnings = []
    if sim.get("matches"):
        warnings.append("⚠️ [CHECK 2 – TOPIC OVERLAP WARNING]\nBelow block threshold but review recommended.")
    if contra.get("blocked") is False and "detected" in contra.get("reason", ""):
         warnings.append("⚠️ [CHECK 3 – CONFLICT WARNING]\n" + contra["reason"])

    if warnings:
         return False, "⚠️ UPLOAD WARNINGS:\n\n" + "\n\n".join(warnings)

    return False, ""


def _bulk_ingest(
    store: ChromaStore,
    files_with_paths: list[dict], # List of {"path": str, "metadata": dict}
    job_id: str
) -> IngestionJobReport:
    """Process a batch of files and return a detailed report."""
    report = IngestionJobReport(
        job_id=job_id,
        start_time=datetime.now().isoformat(),
        total_files=len(files_with_paths)
    )

    valid_extensions = [".pdf", ".docx", ".html", ".htm"]

    for item in files_with_paths:
        file_path = item["path"]
        metadata_dict = item["metadata"]
        filename = metadata_dict.get("source_file", Path(file_path).name)
        
        # 1. Extension check
        ext = Path(file_path).suffix.lower()
        if ext not in valid_extensions:
            report.unsupported_files += 1
            report.file_details.append(FileIngestionStatus(
                filename=filename,
                status="unsupported",
                error=f"Format '{ext}' not supported."
            ))
            report.processed_files += 1
            continue

        # 2. Ingest
        try:
            # Re-use _upsert_doc logic indirectly
            # Note: _upsert_doc returns a string, but we need more structured data.
            # I'll create a slightly cleaner version for internal use or wrap it.
            
            # Setting defaults
            if "status" not in metadata_dict:
                metadata_dict["status"] = "active"
            if "verification_status" not in metadata_dict:
                metadata_dict["verification_status"] = "verified"
            
            metadata = DocMetadata(**metadata_dict)
            chunks = chunk_document(file_path, metadata)
            
            if not chunks:
                report.failed_files += 1
                report.file_details.append(FileIngestionStatus(
                    filename=filename,
                    status="failed",
                    error="No text extracted (may be empty or image-based PDF)"
                ))
            else:
                n = store.upsert_chunks(chunks)
                report.successful_files += 1
                report.total_chunks += n
                report.file_details.append(FileIngestionStatus(
                    filename=filename,
                    status="success",
                    chunks=n
                ))
                logger.info(f"Bulk Audit: {filename} ingested ({n} chunks)")

        except Exception as e:
            report.failed_files += 1
            report.file_details.append(FileIngestionStatus(
                filename=filename,
                status="failed",
                error=str(e)
            ))
            logger.error(f"Bulk Error processing {filename}: {e}")
        
        report.processed_files += 1

    report.end_time = datetime.now().isoformat()
    return report


def _bulk_ingest_gen(
    store: ChromaStore,
    files_with_paths: list[dict], # List of {"path": str, "metadata": dict}
    job_id: str
):
    """Process a batch of files and yield progress after each file."""
    report = IngestionJobReport(
        job_id=job_id,
        start_time=datetime.now().isoformat(),
        total_files=len(files_with_paths)
    )

    valid_extensions = [".pdf", ".docx", ".html", ".htm"]

    for i, item in enumerate(files_with_paths):
        file_path = item["path"]
        metadata_dict = item["metadata"]
        filename = metadata_dict.get("source_file", Path(file_path).name)
        
        detail = None
        
        # 1. Extension check
        ext = Path(file_path).suffix.lower()
        if ext not in valid_extensions:
            report.unsupported_files += 1
            detail = FileIngestionStatus(
                filename=filename,
                status="unsupported",
                error=f"Format '{ext}' not supported."
            )
            report.file_details.append(detail)
            report.processed_files += 1
            yield (i + 1, detail, report)
            continue

        # 2. Ingest
        try:
            if "status" not in metadata_dict:
                metadata_dict["status"] = "active"
            if "verification_status" not in metadata_dict:
                metadata_dict["verification_status"] = "verified"
            
            metadata = DocMetadata(**metadata_dict)
            chunks = chunk_document(file_path, metadata)
            
            if not chunks:
                report.failed_files += 1
                detail = FileIngestionStatus(
                    filename=filename,
                    status="failed",
                    error="No text extracted (may be empty or image-based PDF)"
                )
                report.file_details.append(detail)
            else:
                # RUN GUARDRAILS for Bulk Ingestion
                api_key = os.getenv("GROQ_API_KEY", "")
                is_blocked, guard_report = _run_upload_guardrails(store, chunks, metadata_dict, api_key)
                
                if is_blocked:
                    report.failed_files += 1
                    detail = FileIngestionStatus(
                        filename=filename,
                        status="failed",
                        error=guard_report
                    )
                    report.file_details.append(detail)
                    logger.warning(f"Bulk Guardrails: Blocked {filename}.\n{guard_report}")
                else:
                    n = store.upsert_chunks(chunks)
                    report.successful_files += 1
                    report.total_chunks += n
                    detail = FileIngestionStatus(
                        filename=filename,
                        status="success",
                        chunks=n
                    )
                    # If there was a warning but not blocked, append to detail or just log
                    if guard_report:
                        detail.error = guard_report # Use error field for warnings in UI
                    
                    report.file_details.append(detail)
                    logger.info(f"Bulk Audit: {filename} ingested ({n} chunks)")

        except Exception as e:
            report.failed_files += 1
            detail = FileIngestionStatus(
                filename=filename,
                status="failed",
                error=str(e)
            )
            report.file_details.append(detail)
            logger.error(f"Bulk Error processing {filename}: {e}")
        
        report.processed_files += 1
        yield (i + 1, detail, report)

    report.end_time = datetime.now().isoformat()
    yield (len(files_with_paths), None, report)


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
        # Default overrides for contributors vs admins
        # ONLY apply defaults if the key is missing or specifically tells us to use defaults
        if metadata_dict.get("status") is None:
            metadata_dict["status"] = "active"
        if metadata_dict.get("verification_status") is None:
            metadata_dict["verification_status"] = "verified"
            
        # 1. First Pass: Metadata parsing
        metadata = DocMetadata(**metadata_dict)
        
        # 2. Chunk document to get text for guardrails
        chunks = chunk_document(file_path, metadata)
        if not chunks:
            file_ext = Path(file_path).suffix.lower()
            if file_ext == ".pdf":
                return (
                    "❌ PDF Extraction Failed\n\n"
                    "This is a scanned/image-based PDF with no selectable text.\n\n"
                    "✅ SOLUTION: OCR might be required. Re-upload as a searchable text PDF."
                )
            else:
                return f"❌ ERROR: No text extracted from {Path(file_path).name}. File may be empty."

        # 3. RUN THE INTEGRITY GUARDRAILS
        api_key = os.getenv("GROQ_API_KEY", "")
        is_blocked, report = _run_upload_guardrails(store, chunks, metadata_dict, api_key)
        
        if is_blocked:
            # Audit log the blockage
            logger.warning(f"Guardrails: Upload blocked for {metadata.source_file}.\n{report}")
            return report

        # 4. If not blocked, proceed to upsert
        n = store.upsert_chunks(chunks)
        
        # Audit logging
        logger.info(f"Audit: User {metadata.contributor_id} upserted {metadata.source_file}")
        
        success_msg = f"✅ SUCCESS: Upserted {n} chunks from '{Path(file_path).name}' (v{metadata.version})"
        
        # Surface any persistent warnings if allowed but flagged
        if report:
            return success_msg + f"\n\n{report}"
        
        return success_msg
        
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


def _archive_old_docs(store: ChromaStore, years_threshold: int = 3) -> str:
    """Mark documents older than X years as archived."""
    current_year = datetime.now().year
    target_year = current_year - years_threshold
    
    all_meta = store.get_all_metadata()
    chunks_to_archive = []
    
    # We need to get IDs. Since get_all_metadata doesn't return IDs, we use .get()
    result = store._collection.get(
        where={"year": {"$lt": target_year}},
        include=["metadatas"]
    )
    ids = result.get("ids", [])
    if not ids:
        return f"NO_ACTION: No documents found older than {target_year}."
    
    n = store.update_metadata(ids, {"is_archived": True})
    return f"SUCCESS: Archived {n} chunks from documents published before {target_year}."


def _auto_deduplicate(store: ChromaStore, similarity_threshold: float = 0.98) -> str:
    """Find and archive/delete semantically identical chunks."""
    all_meta = store.get_all_metadata()
    if not all_meta:
        return "KB is empty."

    # 1. Group by Topic + Dept to find superseded versions
    registry: dict[tuple[str, str], list[dict]] = {}
    for m in all_meta:
        key = (m.get("topic", ""), m.get("department", "GENERAL").upper())
        registry.setdefault(key, []).append(m)

    archived_count = 0
    superseded_sources = []

    for (topic, dept), metas in registry.items():
        if not topic: continue
        
        # Sort by year desc, then version desc
        sorted_metas = sorted(
            metas, 
            key=lambda x: (x.get("year", 0), x.get("version", "0")), 
            reverse=True
        )
        
        # Keep the latest, archive everything else
        latest = sorted_metas[0]
        for old in sorted_metas[1:]:
            if not old.get("is_archived"):
                # Query IDs for this old source/version and archive them
                result = store._collection.get(
                    where={
                        "$and": [
                            {"source_file": {"$eq": old.get("source_file")}},
                            {"version": {"$eq": old.get("version")}}
                        ]
                    },
                    include=[]
                )
                ids = result.get("ids", [])
                if ids:
                    store.update_metadata(ids, {"is_archived": True})
                    archived_count += len(ids)
                    superseded_sources.append(f"{old.get('source_file')} (v{old.get('version')})")

    if archived_count == 0:
        return "DEDUPLICATION_REPORT: No redundant versions found. KB is optimized."

    report = (
        f"✅ DEDUPLICATION COMPLETE: Archived {archived_count} chunks.\n"
        f"The following superseded versions were moved to archive:\n"
        + "\n".join([f"  • {s}" for s in set(superseded_sources)])
    )
    return report


def _close_event(agent: AdminAgent, topic: str) -> str:
    """Automate event closure: mark documents as archived and update status."""
    store = agent._store
    from src.schemas import SearchFilter
    
    # Archive all documents related to this topic
    affected = store.update_metadata_by_filter(
        SearchFilter(topic=topic),
        {"status": "archived", "event_status": "closed"}
    )
    
    return f"✅ SUCCESS: Closed event '{topic}'. Archived {affected} chunks."


# ---------------------------------------------------------------------------
# AdminAgent: wraps tools and optionally connects an LLM
def _verify_contribution(store: ChromaStore, file_path: str) -> str:
    """AI-audit a student submission for governance and PII compliance."""
    try:
        from src.ingestion import load_document
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        pages = load_document(file_path)
        if not pages:
            return "❌ Error: Could not extract text from document."
            
        # Sample first 3 pages
        context = "\n\n".join([p[0] for p in pages[:3]])
        
        llm = ChatOpenAI(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are the University Governance Auditor. Audit the following document for knowledge base ingestion.\n"
                "Criteria:\n"
                "1. Relevance: Must be university-related (circulars, rules, events, policies).\n"
                "2. Privacy: No private student/staff phone numbers, home addresses, or private emails.\n"
                "3. Governance: Identify the likely Doc Type (handbook, circular, policy, event).\n"
                "Return a JSON object with: 'is_relevant' (bool), 'contains_pii' (bool), 'pii_details' (string), "
                "'recommended_type' (string), and 'summary' (brief audit note)."
            )),
            ("human", "Document Content Sample:\n\n{context}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"context": context})
        return response.content
    except Exception as e:
        return f"❌ Audit Failed: {e}"


def _get_version_history(store: ChromaStore, topic: str, department: str) -> str:
    history = store.get_version_history(topic, department)
    if not history:
        return f"No history found for topic '{topic}' in department '{department}'."
    return json.dumps(history, indent=2)


def _supersede_document(
    store: ChromaStore, 
    topic: str, 
    department: str, 
    new_source_file: str,
    obsolete_old: bool = False
) -> str:
    """Archive current active documents of this topic and department."""
    history = store.get_version_history(topic, department)
    active_docs = [m for m in history if m.get("status") == "active"]
    
    if not active_docs:
        return f"No active documents found for {topic}/{department} to supersede."
    
    count = 0
    new_status = "obsolete" if obsolete_old else "archived"
    
    for doc in active_docs:
        source = doc.get("source_file")
        if source == new_source_file:
            continue
            
        res = store._collection.get(
            where={"source_file": {"$eq": source}},
            include=[]
        )
        ids = res.get("ids", [])
        if ids:
            store.update_metadata(ids, {
                "status": new_status,
                "is_archived": True,
                "superseded_by": new_source_file
            })
            count += 1
            
    return f"✅ Successfully superseded {count} old versions. Status set to '{new_status}'."


def _rollback_version(
    store: ChromaStore,
    topic: str,
    department: str,
    target_source_file: str
) -> str:
    """Set a specific historical version to 'active' and archive others."""
    history = store.get_version_history(topic, department)
    target_exists = any(m.get("source_file") == target_source_file for m in history)
    
    if not target_exists:
        return f"❌ Target version '{target_source_file}' not found in history."
        
    active_docs = [m for m in history if m.get("status") == "active"]
    for doc in active_docs:
        res = store._collection.get(where={"source_file": {"$eq": doc.get("source_file")}}, include=[])
        if res["ids"]:
            store.update_metadata(res["ids"], {"status": "archived", "is_archived": True})
            
    res = store._collection.get(where={"source_file": {"$eq": target_source_file}}, include=[])
    if res["ids"]:
        store.update_metadata(res["ids"], {
            "status": "active", 
            "is_archived": False,
            "superseded_by": ""
        })
        return f"✅ Successfully rolled back to '{target_source_file}'. It is now the active version."
        
    return "❌ Rollback failed."


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
            description="Use this to get high-level statistics about the KB, including the total number of chunks, unique topics, and document types.",
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

        archive_docs = StructuredTool.from_function(
            func=lambda years_threshold=3: _archive_old_docs(store, years_threshold),
            name="archive_docs_tool",
            description="Archive documents older than a certain number of years. Args: years_threshold (int, default 3).",
        )

        auto_dedup = StructuredTool.from_function(
            func=lambda: _auto_deduplicate(store),
            name="auto_deduplicate_tool",
            description="Automatically identify and handle redundant content in the KB.",
        )

        close_event = StructuredTool.from_function(
            func=lambda topic: _close_event(self, topic),
            name="close_event_tool",
            description="Automate post-event closure: mark documents as closed and archive them. Args: topic (str).",
        )

        get_history = StructuredTool.from_function(
            func=lambda topic, department: _get_version_history(store, topic, department),
            name="get_version_history_tool",
            description="Retrieve the lifecycle history (active, archived, obsolete) of a specific topic/department.",
        )

        supersede = StructuredTool.from_function(
            func=lambda topic, department, new_source_file, obsolete_old=False: _supersede_document(
                store, topic, department, new_source_file, obsolete_old
            ),
            name="supersede_document_tool",
            description="Mark older versions of a document as archived/obsolete when a new one is uploaded.",
        )

        rollback = StructuredTool.from_function(
            func=lambda topic, department, target_source_file: _rollback_version(
                store, topic, department, target_source_file
            ),
            name="rollback_version_tool",
            description="Revert to an older version of a document, making it active and archiving the current one.",
        )

        verify_contribution = StructuredTool.from_function(
            func=lambda file_path: _verify_contribution(store, file_path),
            name="verify_contribution_tool",
            description="Audit a student submission for PII and university relevance using AI.",
        )

        return [
            kb_stats, upsert_doc, delete_meta, detect_dups, 
            reindex, detect_conflicts, archive_docs, auto_dedup, 
            close_event, get_history, supersede, rollback,
            verify_contribution
        ]

    # ------------------------------------------------------------------
    # Direct tool calls (no LLM required)
    # ------------------------------------------------------------------

    def kb_stats(self) -> str:
        return _kb_stats(self._store)

    def upsert_doc(self, file_path: str, metadata_dict: dict) -> str:
        return _upsert_doc(self._store, file_path, metadata_dict)

    def bulk_ingest(self, files_with_paths: list[dict], job_id: str) -> IngestionJobReport:
        """Process a batch of files and return a detailed report."""
        return _bulk_ingest(self._store, files_with_paths, job_id)

    def bulk_ingest_gen(self, files_with_paths: list[dict], job_id: str):
        """Yield progress during batch ingestion."""
        return _bulk_ingest_gen(self._store, files_with_paths, job_id)

    def delete_by_metadata(self, filters: dict) -> str:
        return _delete_by_metadata(self._store, filters)

    def detect_duplicates(self, metadata_dict: dict) -> str:
        return _detect_duplicates(self._store, metadata_dict)

    def detect_conflicts(self, topic: str) -> str:
        return _detect_conflicts(self._store, topic)

    def recommend_reindex(self) -> str:
        return _recommend_reindex(self._store)

    def get_version_history(self, topic: str, department: str) -> str:
        return _get_version_history(self._store, topic, department)

    def supersede_document(self, topic: str, department: str, new_source: str, obsolete: bool = False) -> str:
        return _supersede_document(self._store, topic, department, new_source, obsolete)

    def rollback_version(self, topic: str, department: str, target: str) -> str:
        return _rollback_version(self._store, topic, department, target)

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
                "You are the University Knowledge Base Super-Admin Assistant. "
                "You have FULL AUTHORITY to read, edit, and delete data in ChromaDB using your tools. "
                "Available tools:\n"
                "  • kb_stats_tool: Use to show current KB stats (counts, topics, etc.)\n"
                "  • upsert_doc_tool: Ingest/update a document (requires metadata)\n"
                "  • delete_by_metadata_tool: Delete chunks by filter\n"
                "  • detect_duplicates_tool: Find duplicate documents\n"
                "  • detect_conflicts_tool: Find contentual conflicts\n"
                "  • recommend_reindex_tool: Get system recommendations\n"
                "When providing KB stats, use a professional Markdown summary with bold labels and icons. "
                "Always check for duplicates before upserting. "
                "Require explicit confirmation before any deletion. "
            )

            # Create agent using LangGraph (more modern approach)
            self._agent = create_react_agent(llm, self.tools, state_modifier=system_prompt)
            logger.info("✅ LLM Admin Agent initialised (Mixtral-8x7b via Groq)")
            return self._agent

        except ImportError:
            # Fallback 1: Use create_tool_calling_agent (Legacy LangChain)
            try:
                from langchain.agents import create_tool_calling_agent, AgentExecutor
                from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
                
                model = _get_groq_model(api_key)
                llm = ChatOpenAI(
                    model=model,
                    temperature=0.0,
                    base_url="https://api.groq.com/openai/v1",
                    api_key=api_key,
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ])
                
                # Try tool-calling agent (latest LangChain pattern)
                try:
                    from langchain.agents import create_openai_tools_agent as create_agent_func
                except ImportError:
                    try:
                        from langchain.agents import create_tool_calling_agent as create_agent_func
                    except ImportError:
                        # Fallback for older 0.1.x versions
                        from langchain.agents import create_openai_functions_agent as create_agent_func
                
                agent_obj = create_agent_func(llm, self.tools, prompt)
                self._agent = AgentExecutor(agent=agent_obj, tools=self.tools, verbose=True)
                logger.info("✅ LLM Admin Agent initialised (AgentExecutor mode)")
                return self._agent
            except Exception as e_agent:
                logger.error(f"AgentExecutor fallback failed: {e_agent}")
                
                # Fallback 2: simple LLM-Tools binding (passive mode)
                try:
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model=_get_groq_model(api_key),
                        temperature=0.0,
                        base_url="https://api.groq.com/openai/v1",
                        api_key=api_key,
                    )
                    self._agent = llm.bind_tools(self.tools)
                    logger.info("✅ LLM Admin Agent initialised (passive tool-binding mode)")
                    return self._agent
                except Exception as e2:
                    logger.error(f"Could not init LLM agent (all fallbacks failed): {e2}")
                    return None
                
        except Exception as e:
            logger.error(f"Could not init LLM agent: {e}")
            return None

    def run(self, instruction: str) -> str:
        """
        Run an admin instruction through the LLM agent and ensure tools are executed.
        """
        agent = self._get_llm_agent()
        if agent is None:
            return "❌ Agent unavailable: Missing GROQ_API_KEY."

        try:
            from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

            def _log_debug(msg: str):
                log_file = Path("data/agent_debug.log")
                log_file.parent.mkdir(exist_ok=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] {msg}\n")
            
            _log_debug(f"RUN STARTED: {instruction}")
            
            # Universal Input Builder
            input_msgs = [HumanMessage(content=instruction)]
            
            # Try Case A: LangGraph / Dict-based State
            # We try passing a dict first if it looks like a graph
            if hasattr(agent, 'get_graph'):
                try:
                    _log_debug("Attempting LangGraph (dict) invoke...")
                    result = agent.invoke({"messages": input_msgs})
                    if isinstance(result, dict) and "messages" in result:
                        _log_debug(f"LangGraph returned {len(result['messages'])} messages")
                        for msg in reversed(result["messages"]):
                            content = getattr(msg, 'content', "")
                            if content and str(content).strip():
                                return str(content)
                        return "✅ Command processed successfully (LangGraph mode)."
                except Exception as e:
                    _log_debug(f"LangGraph (dict) invoke failed: {e}. Falling back to list-based.")

            # Case B: Standard Runnable/Executor (List or String based)
            # This handles AgentExecutor and raw ChatOpenAI/Groq models
            _log_debug("Attempting Standard (list/str) invoke...")
            
            # Handle AgentExecutor specifically
            if hasattr(agent, 'agent'):
                try:
                    result = agent.invoke({"input": instruction, "chat_history": []})
                    return result.get("output", "✅ Task completed.")
                except Exception as e:
                    _log_debug(f"AgentExecutor failed: {e}")
            
            # Fallback to direct model invoke (Hand-rolled ReAct)
            response = agent.invoke(input_msgs)
            
            # If the LLM wants to call tools...
            if hasattr(response, 'tool_calls') and response.tool_calls:
                msgs = [HumanMessage(content=instruction), response]
                results_text = []
                
                for tc in response.tool_calls:
                    t_name = tc["name"]
                    t_args = tc.get("args", {})
                    t_id = tc.get("id")
                    
                    # Find and run the tool
                    for tool in self.tools:
                        if tool.name == t_name:
                            try:
                                res = tool.invoke(t_args)
                                msgs.append(ToolMessage(content=str(res), tool_call_id=t_id))
                                results_text.append(f"Output of {t_name}: {res}")
                            except Exception as te:
                                msgs.append(ToolMessage(content=f"Error: {te}", tool_call_id=t_id))
                            break
                
                # Get final answer from LLM with tool outputs
                final_response = agent.invoke(msgs)
                _log_debug(f"Manual ReAct: final response received: {type(final_response)}")
                if hasattr(final_response, 'content') and str(final_response.content).strip():
                    return str(final_response.content)
                
                # Absolute Fail-safe: Return beautifully formatted summary if LLM final summary is blank
                summary_parts = ["### 📊 Knowledge Base Status Report\n"]
                for res_entry in results_text:
                    if "kb_stats_tool" in res_entry:
                        try:
                            # Extract JSON if possible
                            json_str = res_entry.split(":", 1)[1].strip()
                            s = json.loads(json_str)
                            summary_parts.append(f"- **Total Chunks**: `{s.get('total_chunks', 0)}`")
                            summary_parts.append(f"- **Unique Sources**: `{s.get('unique_sources', 0)}`")
                            summary_parts.append(f"- **Verified Chunks**: `{s.get('verified_chunks', 0)}` ✅")
                            summary_parts.append(f"- **Topics**: {', '.join([f'`{t}`' for t in s.get('topics', [])])}")
                            summary_parts.append(f"- **Departments**: {', '.join([f'`{d}`' for d in s.get('departments', [])])}")
                        except:
                            summary_parts.append(res_entry)
                    else:
                        summary_parts.append(res_entry)
                
                _log_debug("Fallback to formatted results summary.")
                return "\n".join(summary_parts)

            if hasattr(response, 'content') and response.content:
                _log_debug("Returned direct response content.")
                return response.content
            
            _log_debug(f"Empty response fallthrough. Response: {response}")
            return f"⚠️ Agent processed the request but returned no text content. Details: {response}"

        except Exception as e:
            logger.error(f"CRITICAL: Agent run failed: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Agent Error: {e}"
