# 🎓 University Knowledge Base

A production-grade document retrieval system for university circulars, handbooks,
and policy documents — built with **ChromaDB**, **LangChain**, and **Streamlit**.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Document Ingestion** | PDF (text + scanned/OCR), DOCX, HTML |
| **OCR Support** | Automatic OCR for scanned PDFs using Tesseract |
| **Vector Store** | ChromaDB persistent on disk with upsert/delete |
| **Embeddings** | Groq API with `mixtral-8x7b-32768` or local `all-MiniLM-L6-v2` |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` |
| **Metadata Filtering** | Topic, year, department, access, doc type, version |
| **Access Control** | Students see only `public` docs; admins see all |
| **Admin Agent** | LangChain tool-calling agent (Groq-powered) for KB management |
| **Conflict Detection** | LLM analysis identifies contradictory information |
| **RAG Answers** | Retrieval-Augmented Generation for student questions |
| **Premium UI** | Dark academic theme with gold accents, badges, result cards |
| **Admin Features** | Ingest, delete, export, stats, duplicate detection, reindexing |
| **Export** | JSON and CSV metadata backup download |

---

## 📁 Folder Structure

```
university_kb/
├── app.py                     # Streamlit application (entry point)
├── generate_sample_data.py    # Generates sample university documents
├── test_pipeline.py           # End-to-end test script
├── requirements.txt
├── .env.example
├── README.md
├── chroma_db/                 # ChromaDB persistent storage (auto-created)
├── data/                      # Sample documents (auto-generated)
│   ├── handbook_2024.pdf
│   ├── circular_exam_rules_2023.pdf
│   ├── circular_exam_rules_2022.pdf   ← outdated duplicate
│   ├── cse_department_policy.docx
│   └── internal_staff_policy.html
└── src/
    ├── __init__.py
    ├── schemas.py             # Pydantic models: DocMetadata, ChunkRecord, KBStats
    ├── utils.py               # Logging, ID generation, filename heuristics
    ├── ingestion.py           # Document loaders + RecursiveCharacterTextSplitter
    ├── chroma_store.py        # ChromaDB wrapper: upsert, delete, search, stats
    ├── retriever.py           # UniversityRetriever with access control
    └── admin_agent.py         # LangChain StructuredTools + ReAct agent
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- pip

### 2. Create virtual environment

```bash
cd university_kb
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` downloads a ~90MB model on first run (local embeddings).
> If you have a Grok API key, embeddings will be faster via xAI.

### 4. (Optional) Set Groq API key

The app works perfectly without a Groq API key using local embeddings. However, to enable:
- **AI Admin Agent** (manage KB with natural language)
- **Conflict Detection** (identify contradictory information)
- **RAG Answers** (AI-powered student answers with citations)
- **Faster LLM responses** (Groq is 5x faster than alternatives)

Get a Groq API key from [Groq Console](https://console.groq.com/):

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (starts with gsk_...)
```

Or set it directly:

```bash
export GROQ_API_KEY=gsk_...      # Linux/macOS
set GROQ_API_KEY=gsk_...         # Windows
```

### 4b. (Optional) Enable OCR for Scanned PDFs

To automatically extract text from scanned/image-based PDFs:

**Windows:**
1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location: `C:\Program Files\Tesseract-OCR`
3. Add to `.env`:
   ```
   TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```
4. Install Python packages:
   ```bash
   pip install pytesseract Pillow
   ```

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
pip install pytesseract Pillow
```

See [OCR_SETUP.md](OCR_SETUP.md) for detailed instructions.

### 5. Generate sample documents

```bash
python generate_sample_data.py
```

This creates 5 sample university documents in `data/`.

### 6. Run the test pipeline (optional)

```bash
python test_pipeline.py
```

This ingests all sample docs, runs search queries, and tests admin operations.

### 7. Launch the Streamlit app

```bash
streamlit run app.py
```

Open your browser to **http://localhost:8501**.

---

## 🖥️ Using the Application

### Student Search Panel

1. Enter a search query (e.g. `"exam hall ticket rules"`)
2. Apply optional filters: topic, department, year, doc type
3. Adjust Top-K slider (1–20)
4. Click **Search** — results show with similarity scores and metadata

> Students only see documents with `access = "public"`.

### Admin Panel

**Ingest / Upsert:**
1. Upload a PDF, DOCX, or HTML file
2. The system auto-suggests metadata from the filename
3. Review/edit the metadata form
4. Click **Check for Duplicates** first (recommended)
5. Click **Ingest Document**

**Delete:**
1. Set one or more filter fields (topic, year, dept, etc.)
2. Check the confirmation checkbox
3. Click **Delete**

**Export:**
- Download full metadata as JSON or CSV for backup

**KB Stats:**
- View chunk count, source count, topic breakdown in real time

### Admin AI Agent Tab

Requires `GROQ_API_KEY`. Chat with an AI agent to manage the knowledge base:

```
> Show me KB statistics
> Check for duplicate exam rules documents
> Delete all chunks from 2022 with topic exam_rules
> What reindexing actions do you recommend?
> How many internal documents do we have?
```

**Features:**
- Natural language commands executed as ChromaDB operations
- Conflict detection when ingesting new documents
- Reindexing recommendations based on KB health
- Chat history preserved during session

---

## 🗄️ Document Metadata Schema

```python
class DocMetadata(BaseModel):
    source_file: str        # Original filename
    doc_type: str           # handbook | circular | policy | other
    section: str            # Section/chapter within the doc
    topic: str              # Primary topic tag (auto-normalised to snake_case)
    year: int               # Publication year
    department: str         # Owning department (auto-uppercased)
    access: str             # public | internal
    version: str            # Version string (e.g. "1.0", "2.0")
    uploaded_time: str      # ISO-8601 UTC timestamp (auto-set)
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | None | Enables Groq LLM, AI Agent, conflict detection, RAG answers |
| `TESSERACT_PATH` | None | Path to Tesseract OCR executable (Windows required, optional macOS/Linux) |
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `LOG_LEVEL` | INFO | Logging verbosity (INFO, DEBUG, WARNING, ERROR) |

---

## 🔌 LLM & Embedding Models

| Mode | Model | Notes |
|---|---|---|
| **Groq** | `mixtral-8x7b-32768` | Requires API key; 5x faster than alternatives; free tier available |
| **Local Embeddings** | `all-MiniLM-L6-v2` | No key needed; ~90MB download; runs on CPU |

**Why Groq?**
- ⚡ **5x faster** LLM responses (1-3s vs 3-8s)
- 💰 **Better pricing** than xAI/OpenAI
- 🎯 **Optimized for RAG** with MoE architecture
- 🆓 **Free tier available** for development

---

## 🧪 Sample Dataset

| File | Type | Access | Year | Notes |
|---|---|---|---|---|
| `handbook_2024.pdf` | handbook | public | 2024 | Full student handbook |
| `circular_exam_rules_2023.pdf` | circular | public | 2023 | Current exam rules |
| `circular_exam_rules_2022.pdf` | circular | public | 2022 | **Outdated duplicate** |
| `cse_department_policy.docx` | policy | public | 2024 | CSE department rules |
| `internal_staff_policy.html` | policy | **internal** | 2024 | Staff only — hidden from students |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Streamlit UI (app.py)
    │
    ├── Student Panel → UniversityRetriever → ChromaStore.similarity_search()
    │                                              │
    │                                         ChromaDB (cosine similarity)
    │                                              │
    │                                         ChunkRecord[] (with scores)
    │
    └── Admin Panel  → AdminAgent
                            ├── kb_stats_tool()
                            ├── upsert_doc_tool()   → ingestion.chunk_document()
                            │                              │
                            │                         ChromaStore.upsert_chunks()
                            ├── delete_by_metadata_tool()
                            ├── detect_duplicates_tool()
                            └── recommend_reindex_tool()
```

---

## � UI Preview

The application features a **premium dark academic theme** with:

- **Hero Header** — Gradient background with status badge
- **Result Cards** — Similarity scores, rank badges, access level indicators
- **Stat Cards** — KB statistics with real-time metrics
- **Badges** — Score badges, rank indicators, access level tags
- **Color Scheme** — Gold (#c9a84c), Teal (#14b8a6), Red accents on dark background

---

## 🐛 Troubleshooting

**Q: Scanned PDF shows "PDF Extraction Failed" error**

A: Install and configure Tesseract OCR:
1. Follow the [OCR_SETUP.md](OCR_SETUP.md) guide
2. Set `TESSERACT_PATH` in `.env`
3. Restart the app and re-upload

**Q: "GROQ_API_KEY is not set" message**

A: Set your Groq API key in `.env` or environment. The app still works with local embeddings, but AI features are disabled.

**Q: Slow document ingestion**

A: Large PDFs (100+ pages) may take several minutes. OCR processing is page-by-page and slower than text extraction.

**Q: Embeddings are too slow**

A: Use Groq API instead of local embeddings. Set `GROQ_API_KEY` for 5x faster inference.

---

## �📝 License

MIT — free for academic and personal use.
