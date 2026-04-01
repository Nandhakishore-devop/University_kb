"""
utils.py - Shared utilities: logging setup, ID generation, file helpers.
"""

from __future__ import annotations
import hashlib
import os
import sys
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure loguru for the application."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )
    logger.add(
        "university_kb.log",
        rotation="10 MB",
        retention="14 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    )


# Call once at import time with default INFO level.
setup_logging()


# ---------------------------------------------------------------------------
# Deterministic chunk ID
# ---------------------------------------------------------------------------

def make_chunk_id(source_file: str, chunk_index: int, version: str = "1.0") -> str:
    """Generate a deterministic, collision-resistant chunk ID.

    Using a hash means re-ingesting the same document version produces the
    same IDs, enabling upsert semantics in ChromaDB.
    """
    raw = f"{source_file}::{version}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm"}


def get_file_extension(file_path: str | Path) -> str:
    return Path(file_path).suffix.lower()


def validate_file(file_path: str | Path) -> Path:
    """Raise if file doesn't exist or isn't a supported type."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{p.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return p


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Metadata suggestion from filename
# ---------------------------------------------------------------------------

import re
from typing import Optional


_YEAR_PATTERN = re.compile(r"(20\d{2})")
_TYPE_KEYWORDS = {
    "handbook": "handbook",
    "circular": "circular",
    "policy": "policy",
}
_DEPT_KEYWORDS = ["cse", "ece", "mech", "civil", "admin", "finance", "hr", "it"]


def suggest_metadata_from_filename(filename: str) -> dict:
    """Heuristically extract metadata fields from a filename.

    Returns a dict with suggested values (may be empty strings/0).
    """
    stem = Path(filename).stem.lower()
    tokens = re.split(r"[_\-\s]+", stem)

    # Year
    year_match = _YEAR_PATTERN.search(stem)
    year = int(year_match.group(1)) if year_match else 0

    # Doc type
    doc_type = "other"
    for kw, dtype in _TYPE_KEYWORDS.items():
        if kw in stem:
            doc_type = dtype
            break

    # Department
    department = "GENERAL"
    for dept in _DEPT_KEYWORDS:
        if dept in tokens:
            department = dept.upper()
            break

    # Topic: join non-year, non-type tokens
    skip = {str(year), doc_type} | set(_DEPT_KEYWORDS)
    topic_tokens = [t for t in tokens if t not in skip and len(t) > 2]
    topic = "_".join(topic_tokens[:3]) if topic_tokens else stem[:20]

    return {
        "doc_type": doc_type,
        "topic": topic,
        "year": year,
        "department": department,
        "access": "public",
        "version": "1.0",
        "section": "",
    }
