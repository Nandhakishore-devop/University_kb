# 🎓 University Knowledge Base

A production-grade document retrieval system for university circulars, handbooks,
and policy documents — built with **ChromaDB**, **LangChain**, and **Streamlit**.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Document Ingestion** | PDF (PyMuPDF), DOCX (python-docx), HTML (BeautifulSoup4) |
| **Vector Store** | ChromaDB persistent on disk with upsert/delete |
| **Embeddings** | OpenAI `text-embedding-3-small` (if key set) or local `all-MiniLM-L6-v2` |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` |
| **Metadata Filtering** | Topic, year, department, access, doc type, version |
| **Access Control** | Students see only `public` docs; admins see all |
| **Admin Agent** | LangChain tool-calling agent (GPT-4o-mini) for KB management |
| **Student UI** | Similarity search with metadata cards and score badges |
| **Admin UI** | Ingest, delete, export, stats, duplicate detection |
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
> If you have an OpenAI key, the download is skipped.

### 4. (Optional) Set OpenAI API key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Or set it directly:

```bash
export OPENAI_API_KEY=sk-...   # Linux/macOS
set OPENAI_API_KEY=sk-...      # Windows
```

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

Requires `OPENAI_API_KEY`. Chat with the agent:

```
> Show me KB statistics
> Check for duplicate exam rules documents
> Delete all chunks from 2022 with topic exam_rules
> What reindexing actions do you recommend?
```

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
| `OPENAI_API_KEY` | None | Enables OpenAI embeddings + AI Agent |
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |

---

## 🔌 Embedding Models

| Mode | Model | Notes |
|---|---|---|
| **OpenAI** | `text-embedding-3-small` | Requires API key; fast; costs ~$0.02/1M tokens |
| **Local** | `all-MiniLM-L6-v2` | No key needed; ~90MB download; runs on CPU |

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

## 📝 License

MIT — free for academic and personal use.
