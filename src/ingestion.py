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
import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.schemas import DocMetadata
from src.utils import get_file_extension, make_chunk_id, validate_file

# Load .env file to get TESSERACT_PATH and other config
load_dotenv()

# ═════════════════════════════════════════════════════════════════════════════
# INITIALIZE TESSERACT PATH (for OCR support)
# ═════════════════════════════════════════════════════════════════════════════
_tesseract_path = os.getenv("TESSERACT_PATH")
if _tesseract_path:
    try:
        import pytesseract
        # Configure the path BEFORE any PDF processing happens
        pytesseract.pytesseract.pytesseract_cmd = _tesseract_path
        # Verify it works immediately
        version = pytesseract.get_tesseract_version()
        logger.info(f"✅ Tesseract OCR ready (v{version}) at {_tesseract_path}")
    except FileNotFoundError:
        logger.error(f"❌ Tesseract not found at: {_tesseract_path}")
        logger.error("   Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    except Exception as e:
        logger.warning(f"⚠️  OCR initialization warning: {e}")
else:
    logger.warning("⚠️  TESSERACT_PATH not set in .env - OCR unavailable for scanned PDFs")


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
    """Load PDF using PyMuPDF. Returns list of (page_text, page_num).
    
    Supports:
      1. Text-based PDFs (normal documents)
      2. Image-based PDFs (scanned documents) — requires OCR
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    pages = []
    try:
        doc = fitz.open(str(file_path))
        total_pages = len(doc)
        text_pages = 0
        
        # First pass: try to extract selectable text
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append((text, i + 1))
                text_pages += 1
        
        # If no text found, try OCR on images
        if text_pages == 0:
            logger.warning(
                f"PDF {file_path.name}: No selectable text found. "
                f"Attempting to extract images and apply OCR..."
            )
            pages = _load_pdf_with_ocr(doc, file_path)
        
        doc.close()
        
        if pages:
            logger.info(
                f"✓ PDF loaded: {file_path.name} — {len(pages)}/{total_pages} pages "
                f"(via {'text extraction' if text_pages > 0 else 'OCR'})"
            )
        else:
            logger.warning(
                f"✗ PDF {file_path.name}: Failed to extract text or images. "
                f"The file may be corrupted or encrypted."
            )
    except Exception as e:
        logger.error(f"Error loading PDF {file_path.name}: {e}")
        raise
    
    return pages


def _load_pdf_with_ocr(doc, file_path: Path) -> List[Tuple[str, int]]:
    """Extract images from PDF and apply OCR. Requires pytesseract + Pillow + Tesseract-OCR system package."""
    pages = []
    
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io
        
        # Ensure Tesseract path is set as failsafe
        _tesseract_path = os.getenv("TESSERACT_PATH")
        if _tesseract_path:
            pytesseract.pytesseract.pytesseract_cmd = _tesseract_path
    except ImportError as e:
        error_msg = "❌ Required OCR libraries not installed!"
        if "pytesseract" in str(e) or "PIL" in str(e):
            error_msg += "\n   → Run: pip install pytesseract Pillow"
        if "fitz" in str(e):
            error_msg += "\n   → Run: pip install pymupdf"
        error_msg += (
            "\n   → Also install Tesseract-OCR system package:"
            "\n   → https://github.com/UB-Mannheim/tesseract/wiki"
        )
        logger.error(error_msg)
        return []
    
    try:
        total_pages = len(doc)
        ocr_pages = 0
        logger.info(f"Starting OCR processing on {total_pages} pages...")
        
        for i, page in enumerate(doc):
            logger.info(f"Processing page {i + 1}/{total_pages} with OCR...")
            text_found = False
            
            # Try to extract images from page
            image_list = page.get_images()
            
            if image_list:
                # Convert PDF image to PIL Image and OCR
                for img_index in image_list:
                    try:
                        xref = img_index[0]
                        pix = fitz.Pixmap(doc, xref)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Apply OCR
                        text = pytesseract.image_to_string(img).strip()
                        if text:
                            pages.append((text, i + 1))
                            ocr_pages += 1
                            text_found = True
                            logger.debug(f"✓ OCR extracted {len(text)} chars from page {i + 1} image")
                            break  # Use first image per page
                    except Exception as img_err:
                        logger.debug(f"Image OCR failed on page {i+1}: {img_err}")
            
            # If no text found yet, render page as image and OCR
            if not text_found:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    text = pytesseract.image_to_string(img).strip()
                    if text:
                        pages.append((text, i + 1))
                        ocr_pages += 1
                        logger.debug(f"✓ OCR extracted {len(text)} chars from rendered page {i + 1}")
                    else:
                        logger.warning(f"⚠️  Page {i+1}: No text detected by OCR (page may be blank)")
                except Exception as render_err:
                    logger.warning(f"❌ Could not OCR page {i+1}: {render_err}")
        
        if pages:
            logger.info(f"✅ OCR processing complete: {ocr_pages}/{total_pages} pages successfully processed")
        else:
            logger.error(f"❌ OCR processing failed: No text extracted from any page")
    except Exception as e:
        logger.error(f"❌ OCR processing error: {e}")
        return []
    
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
        file_ext = p.suffix.lower()
        error_msg = f"No text extracted from {p.name}"
        
        if file_ext == ".pdf":
            error_msg += (
                " — This is likely an image-based or scanned PDF. "
                "Please verify the PDF contains selectable text or use OCR."
            )
        elif file_ext in [".docx", ".doc"]:
            error_msg += " — The Word document may be corrupted or empty."
        elif file_ext in [".html", ".htm"]:
            error_msg += " — The HTML file may be empty or improperly formatted."
        
        logger.warning(error_msg)
        return []

    splitter = _get_splitter()
    chunks_out = []
    chunk_global_index = 0

    logger.info(f"Chunking {p.name}: {len(pages)} pages → processing into chunks...")
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
