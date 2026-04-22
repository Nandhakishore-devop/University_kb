"""
ai_classifier.py - AI-Assisted Metadata Classification for University KB.

Given an uploaded document (path or UploadedFile bytes), reads the first
2-3 pages and asks the Groq LLM to predict:
  - department
  - topic
  - year
  - access level
  - priority
  - doc_type

Returns an AIClassificationResult with per-field confidence scores and
human-readable reasoning. Falls back to suggest_metadata_from_filename()
when no API key is available.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Helpers: text extraction from document bytes / path
# ---------------------------------------------------------------------------

def _extract_text_sample(file_path: str, max_pages: int = 3) -> str:
    """Extract plain text from the first `max_pages` pages of a document."""
    ext = Path(file_path).suffix.lower()
    text_parts: list[str] = []

    try:
        if ext == ".pdf":
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                text_parts.append(page.get_text())
            doc.close()

        elif ext == ".docx":
            from docx import Document
            doc = Document(file_path)
            for i, para in enumerate(doc.paragraphs):
                if i >= 80:  # ~3 pages worth of paragraphs
                    break
                text_parts.append(para.text)

        elif ext in (".html", ".htm"):
            from bs4 import BeautifulSoup
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "lxml")
            raw = soup.get_text(separator="\n")
            text_parts = raw.splitlines()[:200]

    except Exception as e:
        logger.warning(f"Text extraction failed for {file_path}: {e}")

    combined = "\n".join(text_parts).strip()
    # Truncate to ~3000 chars so we stay within token budget
    return combined[:3000]


# ---------------------------------------------------------------------------
# Classification schema
# ---------------------------------------------------------------------------

DEPARTMENT_OPTIONS = ["CSE", "ECE", "MECH", "CIVIL", "ADMIN", "FINANCE", "HR", "IT", "GENERAL"]
TOPIC_HINT = "snake_case tag like exam_rules, fee_structure, lab_policy, placement_cell"
DOC_TYPE_OPTIONS = ["handbook", "circular", "policy", "poster", "attendance", "certificate", "event", "other"]
ACCESS_OPTIONS = ["public", "internal"]
PRIORITY_OPTIONS = ["high", "medium", "low"]

_CLASSIFICATION_SYSTEM_PROMPT = """\\
You are a University Document Metadata Classifier.
Analyze the provided document text and classify it for a university knowledge base.

Return a single valid JSON object (no markdown, no explanation outside the JSON) with exactly this structure:
{{
  "department": "<one of: CSE, ECE, MECH, CIVIL, ADMIN, FINANCE, HR, IT, GENERAL>",
  "department_confidence": <float 0.0-1.0>,
  "department_reasoning": "<one sentence>",

  "topic": "<snake_case topic tag, e.g. exam_rules or fee_structure>",
  "topic_confidence": <float 0.0-1.0>,
  "topic_reasoning": "<one sentence>",

  "year": <integer year like 2024, or 0 if unknown>,
  "year_confidence": <float 0.0-1.0>,
  "year_reasoning": "<one sentence>",

  "access": "<public or internal>",
  "access_confidence": <float 0.0-1.0>,
  "access_reasoning": "<one sentence>",

  "priority": "<high, medium, or low>",
  "priority_confidence": <float 0.0-1.0>,
  "priority_reasoning": "<one sentence>",

  "doc_type": "<one of: handbook, circular, policy, poster, attendance, certificate, event, other>",
  "doc_type_confidence": <float 0.0-1.0>,
  "doc_type_reasoning": "<one sentence>",

  "overall_summary": "<2-3 sentences describing what this document is about>"
}}

Guidelines:
- 'priority' = high if it is time-sensitive (urgent circular, exam notice) or affects all students; medium for general policies; low for supplementary / archival content.
- 'access' = internal if the document contains staff-only info, financial details, HR records, or unpublished internal communications.
- 'year' should be extracted from the document text (look for a publication date); if absent, use the document's academic year or 0.
- 'topic' MUST be a concise, lowercase, snake_case tag (e.g. 'regulations', 'curriculum', 'vision_mission'). DO NOT return 'document' or leave it empty.
"""


# ---------------------------------------------------------------------------
# Core classification function
# ---------------------------------------------------------------------------

def classify_document(
    file_path: str,
    filename: str = "",
) -> dict:
    """
    Classify a document using Groq LLM.

    Args:
        file_path: Absolute path to the temporary file on disk.
        filename:  Original filename (used as fallback hint).

    Returns:
        A dict matching AIClassificationResult fields. Always returns a dict
        even when the API call fails (falls back to heuristic suggestions).
    """
    text_sample = _extract_text_sample(file_path)

    # ---- Try LLM classification ------------------------------------------
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and text_sample:
        result = _classify_with_llm(text_sample, filename, api_key)
        if result:
            result["source"] = "llm"
            return result

    # ---- Fallback: heuristic from filename --------------------------------
    logger.info("AI classifier: falling back to heuristic (no API key or empty text)")
    return _heuristic_fallback(filename or Path(file_path).name)


def _classify_with_llm(text_sample: str, filename: str, api_key: str) -> Optional[dict]:
    """Call Groq LLM and parse the JSON classification result."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )

        human_content = (
            f"Filename: {filename}\\n\\n"
            f"Document Text (first ~3 pages):\\n{text_sample}"
        )

        messages = [
            SystemMessage(content=_CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\\n?```$", "", raw, flags=re.MULTILINE)

        data = json.loads(raw)
        return _validate_and_clean(data)

    except json.JSONDecodeError as e:
        logger.error(f"AI classifier: JSON parse error — {e}")
        return None
    except Exception as e:
        logger.error(f"AI classifier: LLM call failed — {e}")
        return None


def _validate_and_clean(data: dict) -> dict:
    """Ensure all required keys exist with sane values."""
    now_year = __import__("datetime").datetime.utcnow().year

    # Mapping to handle common LLM variations
    DEPT_MAP = {
        "computer": "CSE", "cse": "CSE", "it": "IT", "information": "IT",
        "electronics": "ECE", "ece": "ECE", "communication": "ECE",
        "mechanical": "MECH", "mech": "MECH",
        "civil": "CIVIL",
        "admin": "ADMIN", "office": "ADMIN",
        "finance": "FINANCE", "accounts": "FINANCE", "account": "FINANCE",
        "hr": "HR", "human": "HR", "personnel": "HR",
        "general": "GENERAL", "univ": "GENERAL"
    }

    def clamp(v: float) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return 0.5

    def pick_dept(v: str) -> str:
        v = str(v).strip().lower()
        # Direct match check
        for opt in DEPARTMENT_OPTIONS:
            if v == opt.lower():
                return opt
        # Keyword mapping check
        for key, target in DEPT_MAP.items():
            if key in v:
                return target
        return "GENERAL"

    def pick(v: str, options: list[str], default: str) -> str:
        v = str(v).strip().lower()
        for opt in options:
            if v == opt.lower():
                return opt
        # Fuzzy keyword matching for doc_type etc
        DT_MAP = {
            "handbook": "handbook", "regulation": "handbook", "curriculum": "handbook",
            "notice": "circular", "announcement": "circular", "circular": "circular",
            "policy": "policy", "guideline": "policy",
            "poster": "poster", "event": "event",
            "attendance": "attendance", "mark": "attendance",
            "template": "certificate", "certificate": "certificate"
        }
        for key, target in DT_MAP.items():
            if key in v and target in options:
                return target
        return default

    department = pick_dept(data.get("department", "GENERAL"))
    topic_raw  = str(data.get("topic") or data.get("topic_hint") or "general").lower().strip().replace(" ", "_")
    topic      = re.sub(r"[^a-z0-9_]", "", topic_raw)[:40] or "general"
    year_raw   = data.get("year", 0)
    try:
        year = int(year_raw)
        if year < 2000 or year > now_year + 1:
            year = 0
    except Exception:
        year = 0

    access   = pick(data.get("access", "public"),   [o for o in ACCESS_OPTIONS],    "public")
    priority = pick(data.get("priority", "medium"),  [o for o in PRIORITY_OPTIONS],  "medium")
    doc_type = pick(data.get("doc_type", "other"),   [o for o in DOC_TYPE_OPTIONS],  "other")

    return {
        "department":            department,
        "department_confidence": clamp(data.get("department_confidence", 0.5)),
        "department_reasoning":  str(data.get("department_reasoning", "")),

        "topic":            topic,
        "topic_confidence": clamp(data.get("topic_confidence", 0.5)),
        "topic_reasoning":  str(data.get("topic_reasoning", "")),

        "year":            year,
        "year_confidence": clamp(data.get("year_confidence", 0.5)),
        "year_reasoning":  str(data.get("year_reasoning", "")),

        "access":            access,
        "access_confidence": clamp(data.get("access_confidence", 0.5)),
        "access_reasoning":  str(data.get("access_reasoning", "")),

        "priority":            priority,
        "priority_confidence": clamp(data.get("priority_confidence", 0.5)),
        "priority_reasoning":  str(data.get("priority_reasoning", "")),

        "doc_type":            doc_type,
        "doc_type_confidence": clamp(data.get("doc_type_confidence", 0.5)),
        "doc_type_reasoning":  str(data.get("doc_type_reasoning", "")),

        "overall_summary": str(data.get("overall_summary", "")),
        "source": "llm",
    }


def _heuristic_fallback(filename: str) -> dict:
    """Use keyword-based heuristics when LLM is unavailable."""
    from src.utils import suggest_metadata_from_filename

    suggested = suggest_metadata_from_filename(filename)

    return {
        "department":            suggested.get("department", "GENERAL"),
        "department_confidence": 0.4,
        "department_reasoning":  "Estimated from filename keywords.",

        "topic":            suggested.get("topic", "general"),
        "topic_confidence": 0.4,
        "topic_reasoning":  "Derived from filename tokens.",

        "year":            int(suggested.get("year", 0)),
        "year_confidence": 0.5 if suggested.get("year") else 0.1,
        "year_reasoning":  "Extracted from 4-digit year pattern in filename." if suggested.get("year") else "Year not found in filename.",

        "access":            suggested.get("access", "public"),
        "access_confidence": 0.5,
        "access_reasoning":  "Defaulted to public; no LLM available to inspect content.",

        "priority":            "medium",
        "priority_confidence": 0.3,
        "priority_reasoning":  "Default priority; no content analysis performed.",

        "doc_type":            suggested.get("doc_type", "other"),
        "doc_type_confidence": 0.45,
        "doc_type_reasoning":  "Inferred from filename keywords.",

        "overall_summary": "Document analyzed by filename heuristics only (no Groq API key).",
        "source": "heuristic",
    }


# ---------------------------------------------------------------------------
# Convenience: classify from Streamlit UploadedFile bytes
# ---------------------------------------------------------------------------

def classify_uploaded_file(uploaded_file) -> dict:
    """
    Classify a Streamlit UploadedFile (or any file-like with .name and .read()).

    Creates a temporary file, classifies it, then cleans up.
    """
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Reset stream position so caller can still read the file
    uploaded_file.seek(0)

    try:
        result = classify_document(tmp_path, filename=uploaded_file.name)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result
