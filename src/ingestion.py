"""
ingestion.py - Document loading and text chunking pipeline.

Supports:
  - PDF  via PyMuPDF (fitz)
  - DOCX via python-docx
  - HTML via BeautifulSoup4

All loaders return a list of (page_text, page_number) tuples, which are
then fed into RecursiveCharacterTextSplitter to produce final chunks.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from src.schemas import DocMetadata
from src.utils import get_file_extension, make_chunk_id, validate_file


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------

def _load_pdf(file_path: Path) -> List[Tuple[str, int]]:
    """Load PDF using PyMuPDF. Returns list of (page_text, page_num)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    pages = []
    doc = fitz.open(str(file_path))
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append((text, i + 1))
    doc.close()
    logger.info(f"PDF loaded: {file_path.name} — {len(pages)} pages with text")
    return pages


def _load_docx(file_path: Path) -> List[Tuple[str, int]]:
    """Load DOCX using python-docx. Treats each paragraph as a unit."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")

    doc = Document(str(file_path))
    # Group paragraphs into pseudo-pages of ~20 paragraphs each
    all_paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    page_size = 20
    pages = []
    for i in range(0, len(all_paras), page_size):
        chunk = "\n".join(all_paras[i : i + page_size])
        pages.append((chunk, i // page_size + 1))

    logger.info(f"DOCX loaded: {file_path.name} — {len(all_paras)} paragraphs")
    return pages


def _load_html(file_path: Path) -> List[Tuple[str, int]]:
    """Load HTML using BeautifulSoup4. Extracts visible text."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4 lxml")

    html = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")

    # Remove script/style tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    logger.info(f"HTML loaded: {file_path.name} — {len(text)} chars")
    return [(text, 1)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(file_path: str | Path) -> List[Tuple[str, int]]:
    """Dispatch to the correct loader based on file extension."""
    p = validate_file(file_path)
    ext = get_file_extension(p)

    loaders = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".html": _load_html,
        ".htm": _load_html,
    }
    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f"No loader for extension: {ext}")

    return loader(p)


def chunk_document(
    file_path: str | Path,
    metadata: DocMetadata,
) -> List[dict]:
    """
    Full pipeline: load → split → attach metadata → return chunk dicts.

    Each returned dict has:
      - id      : deterministic chunk ID
      - text    : chunk text
      - metadata: flat dict for ChromaDB
    """
    p = Path(file_path)
    pages = load_document(p)
    if not pages:
        logger.warning(f"No text extracted from {p.name}")
        return []

    splitter = _get_splitter()
    chunks_out = []
    chunk_global_index = 0

    for page_text, page_num in pages:
        sub_chunks = splitter.split_text(page_text)
        for sub_text in sub_chunks:
            if not sub_text.strip():
                continue
            chunk_id = make_chunk_id(
                metadata.source_file, chunk_global_index, metadata.version
            )
            meta = metadata.to_chroma_dict()
            meta["page_number"] = page_num
            meta["chunk_index"] = chunk_global_index

            chunks_out.append(
                {
                    "id": chunk_id,
                    "text": sub_text.strip(),
                    "metadata": meta,
                }
            )
            chunk_global_index += 1

    logger.info(
        f"Chunked {p.name} (v{metadata.version}) → {len(chunks_out)} chunks"
    )
    return chunks_out
