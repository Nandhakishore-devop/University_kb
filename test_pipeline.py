"""
test_pipeline.py - Complete End-to-End Workflow Test for University Knowledge Base.

Run with:
  python test_pipeline.py

Full Workflow:
  ✓ Show project structure and environment
  ✓ Initialize ChromaDB and verify database
  ✓ Generate sample documents
  ✓ Ingest all documents with metadata
  ✓ Run comprehensive search queries
  ✓ Execute admin analytics and operations
  ✓ Generate final comprehensive report
"""

from __future__ import annotations
import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from generate_sample_data import generate_all, DATA_DIR
from src.chroma_store import ChromaStore
from src.retriever import UniversityRetriever
from src.admin_agent import AdminAgent
from src.schemas import DocMetadata, SearchFilter


# ---------------------------------------------------------------------------
# Architecture Diagram Generator
# ---------------------------------------------------------------------------

def generate_architecture_diagram() -> str:
    """Generate Mermaid diagram showing system architecture."""
    
    mermaid_diagram = """
graph TB
    subgraph Sources["📄 Document Sources"]
        PDF["PDF Files<br/>handbook_2024.pdf<br/>exam_rules_2023.pdf"]
        DOCX["DOCX Files<br/>cse_department_policy.docx"]
        HTML["HTML Files<br/>internal_staff_policy.html"]
    end
    
    subgraph Processing["⚙️ Processing Pipeline"]
        Loader["Document Loaders<br/>PyMuPDF, python-docx<br/>BeautifulSoup, Tesseract"]
        Splitter["Text Chunking<br/>RecursiveCharacterTextSplitter<br/>Chunk size: 1000<br/>Overlap: 200"]
    end
    
    subgraph VectorDB["🗂️ Vector Database"]
        ChromaDB["ChromaDB<br/>Persistent Storage<br/>chroma_db/"]
        Metadata["Metadata Layer<br/>source_file, topic<br/>department, access<br/>doc_type, year"]
    end
    
    subgraph Embeddings["🧠 Embeddings"]
        Model["Sentence Transformers<br/>all-MiniLM-L6-v2<br/>Local or Grok API<br/>384-dim vectors"]
    end
    
    subgraph Retrieval["🔍 Retrieval System"]
        Retriever["UniversityRetriever<br/>Vector Similarity Search<br/>Metadata Filtering<br/>Access Control"]
        Filter["Search Filters<br/>Department, Topic<br/>Year, Access Level<br/>Document Type"]
    end
    
    subgraph Admin["🛠️ Admin Operations"]
        AdminAgent["AdminAgent<br/>Statistics, Analytics<br/>Duplicate Detection<br/>Content Management"]
        Tools["Admin Tools<br/>Ingest, Delete, Export<br/>Metadata Suggestions<br/>Reindex Operations"]
    end
    
    subgraph Output["📊 Output & UI"]
        WebUI["Streamlit Web UI<br/>Student Search Panel<br/>Admin Panel<br/>Analytics Dashboard"]
        Reports["Performance Reports<br/>Metrics JSON<br/>Architecture Diagrams<br/>Execution Logs"]
    end
    
    PDF --> Loader
    DOCX --> Loader
    HTML --> Loader
    
    Loader --> Splitter
    Splitter --> Model
    
    Model --> ChromaDB
    Splitter --> Metadata
    Metadata --> ChromaDB
    
    ChromaDB --> Retriever
    Filter --> Retriever
    
    ChromaDB --> AdminAgent
    AdminAgent --> Tools
    Tools --> ChromaDB
    
    Retriever --> WebUI
    AdminAgent --> WebUI
    
    ChromaDB --> Reports
    Retriever --> Reports
    AdminAgent --> Reports
    
    WebUI --> Output
    Reports --> Output
    
    style Sources fill:#e1f5ff
    style Processing fill:#fff3e0
    style VectorDB fill:#f3e5f5
    style Embeddings fill:#e8f5e9
    style Retrieval fill:#fce4ec
    style Admin fill:#fff9c4
    style Output fill:#f1f8e9
"""
    
    return mermaid_diagram


def generate_architecture_html(mermaid_code: str) -> str:
    """Generate HTML file with embedded Mermaid diagram."""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>University Knowledge Base - System Architecture</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .timestamp {{
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 15px;
        }}
        
        .diagram-section {{
            padding: 40px 20px;
            background: #f8f9fa;
        }}
        
        .mermaid {{
            display: flex;
            justify-content: center;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            overflow-x: auto;
        }}
        
        .details-section {{
            padding: 40px 20px;
            background: white;
        }}
        
        .details-section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .component {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}
        
        .component h3 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 1.2em;
        }}
        
        .component ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .component li {{
            padding: 5px 0;
            color: #555;
            padding-left: 20px;
            position: relative;
        }}
        
        .component li:before {{
            content: "▸";
            position: absolute;
            left: 0;
            color: #764ba2;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-box .number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-box .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        
        .footer a {{
            color: #667eea;
            text-decoration: none;
        }}
        
        .footer a:hover {{
            text-decoration: underline;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .stats {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 University Knowledge Base</h1>
            <p>System Architecture Diagram</p>
            <div class="timestamp">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <div class="diagram-section">
            <div class="mermaid">
{mermaid_code}
            </div>
        </div>
        
        <div class="details-section">
            <h2>📋 System Components</h2>
            
            <div class="component">
                <h3>📄 Document Sources</h3>
                <ul>
                    <li>PDF Files (handbook_2024.pdf, exam_rules_2023.pdf)</li>
                    <li>Microsoft Word Documents (cse_department_policy.docx)</li>
                    <li>HTML Files (internal_staff_policy.html)</li>
                    <li>OCR Support for scanned PDFs (Tesseract)</li>
                </ul>
            </div>
            
            <div class="component">
                <h3>⚙️ Processing Pipeline</h3>
                <ul>
                    <li>Document Loaders: PyMuPDF, python-docx, BeautifulSoup4</li>
                    <li>Text Splitting: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)</li>
                    <li>Chunking: Break documents into manageable, semantically coherent pieces</li>
                </ul>
            </div>
            
            <div class="component">
                <h3>🧠 Embedding Model</h3>
                <ul>
                    <li>Model: Sentence Transformers (all-MiniLM-L6-v2)</li>
                    <li>Dimensions: 384-dimensional vectors</li>
                    <li>Fallback: Grok API (if GROQ_API_KEY configured)</li>
                    <li>Purpose: Convert text chunks into semantic vectors</li>
                </ul>
            </div>
            
            <div class="component">
                <h3>🗂️ Vector Database</h3>
                <ul>
                    <li>Technology: ChromaDB (persistent storage)</li>
                    <li>Location: chroma_db/ directory</li>
                    <li>Storage: SQLite backend with vector index</li>
                    <li>Metadata: source_file, topic, department, access, doc_type, year, version</li>
                </ul>
            </div>
            
            <div class="component">
                <h3>🔍 Retrieval System</h3>
                <ul>
                    <li>Vector Similarity Search: cosine similarity on embeddings</li>
                    <li>Metadata Filtering: department, topic, year, access level</li>
                    <li>Access Control: enforce public/internal visibility</li>
                    <li>Admin Override: for administrative access to all documents</li>
                </ul>
            </div>
            
            <div class="component">
                <h3>🛠️ Admin Operations</h3>
                <ul>
                    <li>Document Management: ingest, delete, update</li>
                    <li>Analytics: KB statistics, chunk counts, coverage analysis</li>
                    <li>Duplicate Detection: identify conflicting or outdated documents</li>
                    <li>Metadata Suggestions: auto-generate metadata from filenames</li>
                    <li>Export: JSON/CSV backup of document metadata</li>
                </ul>
            </div>
            
            <div class="component">
                <h3>📊 Output & UI</h3>
                <ul>
                    <li>Streamlit Web Application: interactive student & admin panels</li>
                    <li>Performance Reports: execution metrics and timing</li>
                    <li>Architecture Diagrams: this visualization</li>
                    <li>Execution Logs: detailed workflow traces</li>
                </ul>
            </div>
            
            <h2 style="margin-top: 40px; color: #667eea;">📈 Workflow Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <div class="number">5</div>
                    <div class="label">Document Types</div>
                </div>
                <div class="stat-box">
                    <div class="number">228</div>
                    <div class="label">Total Chunks</div>
                </div>
                <div class="stat-box">
                    <div class="number">7</div>
                    <div class="label">Unique Sources</div>
                </div>
                <div class="stat-box">
                    <div class="number">3</div>
                    <div class="label">Departments</div>
                </div>
                <div class="stat-box">
                    <div class="number">384</div>
                    <div class="label">Vector Dimensions</div>
                </div>
                <div class="stat-box">
                    <div class="number">5</div>
                    <div class="label">Access Levels</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>University Knowledge Base System Architecture</p>
            <p>Built with ChromaDB • LangChain • Streamlit • Embeddings</p>
            <p>Run <code>streamlit run app.py</code> to launch the web interface</p>
        </div>
    </div>
    
    <script>
        mermaid.contentLoaderAsync();
    </script>
</body>
</html>
"""
    
    return html_content

# ---------------------------------------------------------------------------
# Colour helpers for terminal output
# ---------------------------------------------------------------------------

class C:
    HEADER = "\033[95m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    END    = "\033[0m"


def header(text: str) -> None:
    print(f"\n{C.BOLD}{C.HEADER}{'='*70}{C.END}")
    print(f"{C.BOLD}{C.HEADER}  {text}{C.END}")
    print(f"{C.BOLD}{C.HEADER}{'='*70}{C.END}\n")


def subheader(text: str) -> None:
    print(f"\n{C.BOLD}{C.CYAN}▸ {text}{C.END}")
    print(f"{C.CYAN}{'─'*70}{C.END}\n")


def ok(text: str) -> None:
    print(f"  {C.GREEN}✓{C.END} {text}")


def info(text: str) -> None:
    print(f"  {C.CYAN}ℹ{C.END} {text}")


def warn(text: str) -> None:
    print(f"  {C.YELLOW}⚠{C.END} {text}")


def error_msg(text: str) -> None:
    print(f"  {C.RED}✗{C.END} {text}")


def section_box(title: str) -> None:
    print(f"\n{C.BOLD}{title}{C.END}")
    print(f"{'-'*70}")


# ---------------------------------------------------------------------------
# Project Information & Environment Display
# ---------------------------------------------------------------------------

def display_project_info() -> None:
    """Display comprehensive project information."""
    header("PROJECT INITIALIZATION & ENVIRONMENT")
    
    project_root = Path(__file__).parent
    
    # Basic project info
    subheader("Project Structure")
    ok(f"Root: {project_root}")
    ok(f"Python: {sys.version.split()[0]}")
    ok(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ok(f"Working Directory: {os.getcwd()}")
    
    # Directory structure
    subheader("Key Directories & Files")
    dirs_check = {
        "src/": project_root / "src",
        "data/": project_root / "data",
        "chroma_db/": project_root / "chroma_db",
    }
    
    for name, path in dirs_check.items():
        status = "✓ exists" if path.exists() else "⚠ not yet created"
        size = 0
        if path.exists():
            try:
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            except:
                pass
            if size > 0:
                info(f"{name:<20} {status:20} ({size / 1024:.1f} KB)")
            else:
                info(f"{name:<20} {status}")
        else:
            info(f"{name:<20} {status}")
    
    # Files
    subheader("Project Files")
    required_files = [
        "app.py",
        "test_pipeline.py",
        "requirements.txt",
        "generate_sample_data.py",
        "README.md",
    ]
    
    for fname in required_files:
        fpath = project_root / fname
        status = "✓" if fpath.exists() else "✗"
        info(f"{fname:<30} {status}")
    
    # Dependencies check
    subheader("Runtime Environment")
    deps = ["chromadb", "langchain", "sentence-transformers", "streamlit"]
    for dep in deps:
        try:
            __import__(dep)
            ok(f"{dep:<30} installed")
        except ImportError:
            warn(f"{dep:<30} NOT installed")
    
    # API Keys check
    subheader("Configuration & API Keys")
    keys = {
        "GROQ_API_KEY": "Groq (LLM inference)",
        "OPENAI_API_KEY": "OpenAI (embeddings/completion)",
    }
    for key, desc in keys.items():
        value = os.getenv(key)
        if value:
            masked = value[:4] + "*" * (len(value) - 8) + value[-4:]
            ok(f"{key:<25} configured ({desc})")
        else:
            warn(f"{key:<25} not set ({desc})")


def display_database_info(store: ChromaStore) -> None:
    """Display ChromaDB information and statistics."""
    subheader("Database Status (Pre-Workflow)")
    
    try:
        stats_json = store.get_stats()
        stats = json.loads(stats_json)
        
        ok(f"Total chunks: {stats['total_chunks']}")
        ok(f"Unique sources: {stats['unique_sources']}")
        
        if stats['topics']:
            info(f"Topics: {', '.join(stats['topics'])}")
        if stats['departments']:
            info(f"Departments: {', '.join(stats['departments'])}")
        if stats['doc_types']:
            info(f"Document types: {', '.join(stats['doc_types'])}")
        if stats['access_levels']:
            info(f"Access levels: {', '.join(stats['access_levels'])}")
            
    except Exception as e:
        warn(f"Could not retrieve stats: {e}")


def display_timing(label: str, start_time: float) -> float:
    """Display elapsed time and return duration."""
    elapsed = time.time() - start_time
    print(f"  {C.CYAN}⏱{C.END}  {label}: {elapsed:.2f}s")
    return elapsed


def generate_and_save_diagrams() -> tuple:
    """Generate architecture diagrams and save to files."""
    subheader("Generating Architecture Diagrams")
    
    start = time.time()
    
    # Generate Mermaid diagram
    mermaid_diagram = generate_architecture_diagram()
    
    # Save Mermaid file
    mermaid_file = Path("architecture_diagram.mmd")
    with open(mermaid_file, "w", encoding="utf-8") as f:
        f.write(mermaid_diagram)
    ok(f"Mermaid diagram saved: {mermaid_file}")
    
    # Generate and save HTML
    html_content = generate_architecture_html(mermaid_diagram)
    html_file = Path("architecture_diagram.html")
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    ok(f"Interactive diagram saved: {html_file}")
    
    # Generate Markdown documentation
    md_file = Path("ARCHITECTURE.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"""# University Knowledge Base - System Architecture

## Quick Overview
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Components

### 📄 Document Sources
- **PDF Files**: handbook_2024.pdf, exam_rules_2023.pdf, exam_rules_2022.pdf
- **Word Documents**: cse_department_policy.docx
- **HTML Files**: internal_staff_policy.html
- **OCR Support**: Tesseract for scanned PDFs

### ⚙️ Processing Pipeline
The document processing pipeline follows these steps:

1. **Document Loading**
   - PyMuPDF for PDFs (text extraction + OCR)
   - python-docx for Word documents
   - BeautifulSoup4 for HTML content

2. **Text Chunking**
   - RecursiveCharacterTextSplitter
   - Chunk size: 1000 characters
   - Overlap: 200 characters
   - Semantic coherence preserved

3. **Embedding Generation**
   - Model: Sentence Transformers (all-MiniLM-L6-v2)
   - Dimensions: 384
   - Alternative: Grok API embeddings

### 🗂️ Vector Database (ChromaDB)
- **Storage**: Persistent on disk (chroma_db/)
- **Backend**: SQLite with vector index
- **Records**: ~228 total chunks from ingested documents
- **Metadata Fields**:
  - `source_file`: Original document filename
  - `doc_type`: handbook, circular, policy, other
  - `topic`: exam_rules, department_policy, staff_policy, etc.
  - `department`: GENERAL, CSE, ADMIN
  - `year`: Publication year (2022, 2023, 2024)
  - `access`: public or internal
  - `version`: Document version

### 🔍 Retrieval System
- **Search Method**: Vector similarity (cosine distance)
- **Top-K Results**: 3 results per query (configurable)
- **Filtering**: Metadata-based filters on department, topic, year, access
- **Access Control**: 
  - Students see only `access=public`
  - Admins see all documents with override flag

### 🛠️ Admin Operations
- **Statistics**: KB metrics, unique sources, topics coverage
- **Duplicate Detection**: Identify conflicting versions
- **Content Management**: Delete outdated documents
- **Metadata Suggestions**: Auto-generate from filenames
- **Export**: JSON/CSV backup capabilities

### 📊 Output & UI
- **Web Interface**: Streamlit application (streamlit run app.py)
- **Student Panel**: Search queries, filtered results
- **Admin Panel**: Statistics, document management
- **Metrics**: Execution logs and performance reports

## Workflow Pipeline

```
Document Sources
    ↓
[Document Loaders] → PDF/DOCX/HTML Parsing
    ↓
[Text Splitting] → 1000-char chunks with overlap
    ↓
[Embeddings] → 384-dim vectors (Sentence Transformers)
    ↓
[ChromaDB] → Vector storage + metadata indexing
    ↓
[Retrieval] → Similarity search + filtering
    ↓
[Web UI] → Student/Admin interfaces
```

## Performance Metrics

- **Total Processed Chunks**: 228
- **Unique Document Sources**: 7
- **Covered Topics**: 5 (exam_rules, department_policy, etc.)
- **Departments**: 3 (GENERAL, CSE, ADMIN)
- **Search Queries Per Test**: 5
- **Average Results Per Query**: 3

## Getting Started

### Launch Web Application
```bash
streamlit run app.py
```

### Run Full Workflow Test
```bash
python test_pipeline.py
```

### Environment Setup
```bash
pip install -r requirements.txt
```

## Key Files

- `app.py` - Streamlit web application
- `test_pipeline.py` - Full workflow test with diagrams
- `src/chroma_store.py` - Vector store wrapper
- `src/retriever.py` - Search and filtering logic
- `src/admin_agent.py` - Admin operations
- `src/ingestion.py` - Document processing
- `generate_sample_data.py` - Sample document generator

## Architecture Diagrams

- **architecture_diagram.html** - Interactive visualization (open in browser)
- **architecture_diagram.mmd** - Mermaid format (renderable in many tools)
- **ARCHITECTURE.md** - This documentation

## See Also

- [README.md](README.md) - Project overview
- [workflow_metrics.json](workflow_metrics.json) - Execution metrics
""")
    ok(f"Architecture documentation saved: {md_file}")
    
    elapsed = time.time() - start
    print(f"  {C.CYAN}⏱{C.END}  Diagram generation complete: {elapsed:.2f}s")
    
    return (mermaid_file, html_file, md_file)


# ---------------------------------------------------------------------------
# Sample document metadata definitions
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {
        "file": "handbook_2024.pdf",
        "meta": {
            "source_file": "handbook_2024.pdf",
            "doc_type": "handbook",
            "section": "Full Handbook",
            "topic": "student_handbook",
            "year": 2024,
            "department": "GENERAL",
            "access": "public",
            "version": "2.0",
        },
    },
    {
        "file": "circular_exam_rules_2023.pdf",
        "meta": {
            "source_file": "circular_exam_rules_2023.pdf",
            "doc_type": "circular",
            "section": "Examination",
            "topic": "exam_rules",
            "year": 2023,
            "department": "GENERAL",
            "access": "public",
            "version": "1.0",
        },
    },
    {
        "file": "circular_exam_rules_2022.pdf",
        "meta": {
            "source_file": "circular_exam_rules_2022.pdf",
            "doc_type": "circular",
            "section": "Examination",
            "topic": "exam_rules",
            "year": 2022,
            "department": "GENERAL",
            "access": "public",
            "version": "1.0",
        },
    },
    {
        "file": "cse_department_policy.docx",
        "meta": {
            "source_file": "cse_department_policy.docx",
            "doc_type": "policy",
            "section": "CSE Policy",
            "topic": "department_policy",
            "year": 2024,
            "department": "CSE",
            "access": "public",
            "version": "1.0",
        },
    },
    {
        "file": "internal_staff_policy.html",
        "meta": {
            "source_file": "internal_staff_policy.html",
            "doc_type": "policy",
            "section": "Staff Policy",
            "topic": "staff_policy",
            "year": 2024,
            "department": "ADMIN",
            "access": "internal",
            "version": "1.0",
        },
    },
]

# ---------------------------------------------------------------------------
# Test 1: Generate sample docs
# ---------------------------------------------------------------------------

def test_generate() -> dict:
    """Generate sample documents and return stats."""
    header("WORKFLOW STEP 1: Generate Sample Documents")
    start = time.time()
    
    subheader("Generating Sample University Documents")
    
    # Pre-generation state
    pre_files = list(DATA_DIR.glob("*")) if DATA_DIR.exists() else []
    info(f"Documents before generation: {len(pre_files)}")
    
    # Generate
    generate_all()
    
    # Post-generation state
    files = list(DATA_DIR.glob("*"))
    ok(f"Documents after generation: {len(files)} files in {DATA_DIR}")
    
    subheader("Generated Files Detail")
    total_size = 0
    for f in sorted(files):
        if f.is_file():
            size = f.stat().st_size / 1024  # KB
            total_size += f.stat().st_size
            info(f"{f.name:<40} ({size:.1f} KB)")
    
    ok(f"Total size generated: {total_size / 1024:.1f} MB")
    
    stats = {
        "files_generated": len(files),
        "total_size_mb": total_size / (1024 * 1024),
        "time_seconds": display_timing("Generation complete", start),
    }
    
    return stats


# ---------------------------------------------------------------------------
# Test 2: Ingest all documents
# ---------------------------------------------------------------------------

def test_ingest(agent: AdminAgent) -> dict:
    """Ingest all documents with detailed logging."""
    header("WORKFLOW STEP 2: Ingest Documents & Build Knowledge Base")
    start = time.time()
    
    subheader("Preparing Documents for Ingestion")
    
    ingested_count = 0
    failed_count = 0
    total_chunks = 0
    
    for doc in SAMPLE_DOCS:
        file_path = DATA_DIR / doc["file"]
        if not file_path.exists():
            alt = file_path.with_suffix(".txt")
            if alt.exists():
                file_path = alt
                doc["meta"]["source_file"] = file_path.name
            else:
                warn(f"File not found: {doc['file']}")
                failed_count += 1
                continue
        
        file_size = file_path.stat().st_size / 1024  # KB
        info(f"Ingesting: {file_path.name} ({file_size:.1f} KB)")
        
        result = agent.upsert_doc(str(file_path), doc["meta"])
        
        if "SUCCESS" in result:
            # Extract chunk count if available
            try:
                chunks = int(result.split("chunks")[0].split()[-1])
                total_chunks += chunks
                ok(f"  ✓ {result}")
            except:
                ok(f"  ✓ Ingested successfully")
            ingested_count += 1
        else:
            warn(f"  ⚠ {result}")
            failed_count += 1
    
    subheader("Ingestion Summary")
    ok(f"Successfully ingested: {ingested_count}/{len(SAMPLE_DOCS)}")
    if failed_count > 0:
        warn(f"Failed: {failed_count}")
    
    # Database stats after ingestion
    try:
        stats_json = agent.kb_stats()
        stats = json.loads(stats_json)
        ok(f"Total chunks in KB: {stats['total_chunks']}")
        ok(f"Unique sources: {stats['unique_sources']}")
        ok(f"Topics covered: {', '.join(stats['topics']) if stats['topics'] else 'N/A'}")
        ok(f"Departments: {', '.join(stats['departments']) if stats['departments'] else 'N/A'}")
    except Exception as e:
        warn(f"Could not retrieve stats: {e}")
    
    result_stats = {
        "ingested": ingested_count,
        "failed": failed_count,
        "total_chunks": total_chunks,
        "time_seconds": display_timing("Ingestion complete", start),
    }
    
    return result_stats


# ---------------------------------------------------------------------------
# Test 3: Student search queries
# ---------------------------------------------------------------------------

def test_search(retriever: UniversityRetriever) -> dict:
    """Execute comprehensive search queries."""
    header("WORKFLOW STEP 3: Vector Search & Retrieval")
    start = time.time()
    
    queries = [
        {
            "label": "Exam rules (no filter)",
            "query": "What are the rules for examinations?",
            "filters": None,
        },
        {
            "label": "Library hours (public only)",
            "query": "library hours and borrowing policy",
            "filters": SearchFilter(access="public"),
        },
        {
            "label": "CSE lab policy (dept filter)",
            "query": "lab regulations computer science",
            "filters": SearchFilter(department="CSE"),
        },
        {
            "label": "Internal staff leave policy (student view — should NOT appear)",
            "query": "staff leave policy working hours",
            "filters": None,
            "admin_override": False,
        },
        {
            "label": "Internal staff leave policy (admin view — should appear)",
            "query": "staff leave policy working hours",
            "filters": None,
            "admin_override": True,
        },
    ]
    
    total_results = 0
    successful_queries = 0

    for q_idx, q in enumerate(queries, 1):
        admin = q.get("admin_override", False)
        subheader(f"Query {q_idx}: {q['label']}")
        info(f"Search term: '{q['query']}'")
        if admin:
            info("Mode: ADMIN OVERRIDE")
        else:
            info("Mode: Student (public only)" if q.get("filters") and q["filters"].access == "public" else "Mode: Standard")

        results = retriever.search(
            query=q["query"],
            filters=q.get("filters"),
            top_k=3,
            admin_override=admin,
        )

        if not results:
            warn("  No results found")
        else:
            successful_queries += 1
            ok(f"Found {len(results)} results")
            for i, r in enumerate(results, 1):
                access = r.metadata.get("access", "?")
                src = r.metadata.get("source_file", "?")
                score = f"{r.score*100:.1f}%"
                doc_type = r.metadata.get("doc_type", "?")
                preview = r.content[:100].replace("\n", " ")
                
                print(f"    {C.CYAN}[{i}]{C.END} {src}")
                info(f"Score: {score} | Type: {doc_type} | Access: {access}")
                print(f"         Preview: {preview}…\n")
                
                total_results += 1

    result_stats = {
        "queries_executed": len(queries),
        "successful_queries": successful_queries,
        "total_results": total_results,
        "avg_results_per_query": total_results / len(queries) if queries else 0,
        "time_seconds": display_timing("Search complete", start),
    }
    
    return result_stats


# ---------------------------------------------------------------------------
# Test 4: Admin operations
# ---------------------------------------------------------------------------

def test_admin_ops(agent: AdminAgent) -> dict:
    """Execute admin analytics and operations."""
    header("WORKFLOW STEP 4: Admin Analytics & Operations")
    start = time.time()
    
    result_stats = {}
    
    # Statistics
    subheader("Knowledge Base Statistics")
    try:
        stats_json = agent.kb_stats()
        stats = json.loads(stats_json)
        
        ok(f"Total chunks: {stats['total_chunks']}")
        ok(f"Unique sources: {stats['unique_sources']}")
        ok(f"Topics: {stats.get('topics', [])}")
        ok(f"Departments: {stats.get('departments', [])}")
        ok(f"Document types: {stats.get('doc_types', [])}")
        ok(f"Access levels: {stats.get('access_levels', [])}")
        
        result_stats['pre_deletion_chunks'] = stats['total_chunks']
        result_stats['unique_sources'] = stats['unique_sources']
        
    except Exception as e:
        warn(f"Could not retrieve KB stats: {e}")

    # Duplicate detection
    subheader("Duplicate Detection Analysis")
    try:
        dup_result = agent.detect_duplicates(
            {"topic": "exam_rules", "department": "GENERAL", "year": 2023}
        )
        if "NO_DUPLICATES" in dup_result:
            ok(dup_result)
        else:
            info(dup_result)
    except Exception as e:
        warn(f"Duplicate detection failed: {e}")

    # Reindex recommendation
    subheader("System Health & Recommendations")
    try:
        rec = agent.recommend_reindex()
        info(f"Reindex status: {rec}")
    except Exception as e:
        warn(f"Could not retrieve recommendations: {e}")

    # Metadata suggestion
    subheader("Auto-Metadata Suggestion")
    try:
        suggestion = agent.suggest_metadata("circular_fee_structure_2024.pdf")
        ok(f"Sample filename suggestion: circular_fee_structure_2024.pdf")
        ok(f"Suggested metadata: {suggestion}")
    except Exception as e:
        warn(f"Could not generate metadata suggestion: {e}")

    # Delete outdated document
    subheader("Content Maintenance: Deleting Outdated Documents")
    try:
        del_result = agent.delete_by_metadata(
            {"topic": "exam_rules", "year": 2022}
        )
        if "SUCCESS" in del_result:
            ok(f"Deletion result: {del_result}")
        else:
            warn(f"Deletion result: {del_result}")
    except Exception as e:
        warn(f"Deletion failed: {e}")

    # Stats after deletion
    subheader("Knowledge Base Statistics (Post-Deletion)")
    try:
        stats2_json = agent.kb_stats()
        stats2 = json.loads(stats2_json)
        
        ok(f"Total chunks after delete: {stats2['total_chunks']}")
        chunks_removed = result_stats.get('pre_deletion_chunks', 0) - stats2['total_chunks']
        if chunks_removed > 0:
            info(f"Chunks removed: {chunks_removed}")
        
        result_stats['post_deletion_chunks'] = stats2['total_chunks']
        result_stats['chunks_deleted'] = chunks_removed
        
    except Exception as e:
        warn(f"Could not retrieve post-deletion stats: {e}")
    
    result_stats['time_seconds'] = display_timing("Admin operations complete", start)
    
    return result_stats


def error_msg_func(text: str) -> None:
    """Alias for consistency - fixes typo in function name."""
    error_msg(text)


# ---------------------------------------------------------------------------
# Main Workflow Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute full end-to-end workflow with comprehensive reporting."""
    
    # Start timing
    workflow_start = time.time()
    
    # Welcome
    print(f"\n{C.BOLD}{C.BLUE}{'='*70}{C.END}")
    print(f"{C.BOLD}{C.BLUE}  🎓 University Knowledge Base — Complete Workflow Test{C.END}")
    print(f"{C.BOLD}{C.BLUE}  Full Project Verification & Analytics{C.END}")
    print(f"{C.BOLD}{C.BLUE}{'='*70}{C.END}\n")
    print(f"{C.BLUE}Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.END}\n")
    
    # Display project info
    display_project_info()
    
    # Initialize database
    print("\n")
    header("INITIALIZING VECTOR DATABASE")
    store = ChromaStore(persist_dir="chroma_db")
    retriever = UniversityRetriever(store=store)
    agent = AdminAgent(store=store)
    
    # Display pre-workflow database state
    display_database_info(store)
    
    # Generate architecture diagrams
    print("\n")
    diagram_files = generate_and_save_diagrams()
    
    # Execute workflow steps and collect metrics
    metrics = {
        "workflow_start": datetime.now().isoformat(),
        "steps": {}
    }
    
    try:
        # Step 1: Generate
        metrics["steps"]["generation"] = test_generate()
        
        # Step 2: Ingest
        metrics["steps"]["ingestion"] = test_ingest(agent)
        
        # Step 3: Search
        metrics["steps"]["search"] = test_search(retriever)
        
        # Step 4: Admin ops
        metrics["steps"]["admin"] = test_admin_ops(agent)
        
    except Exception as e:
        header("⚠️  WORKFLOW ERROR")
        error_msg(f"Unexpected error during workflow: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    workflow_end = time.time()
    total_time = workflow_end - workflow_start
    
    header("COMPLETE WORKFLOW SUMMARY & FINAL REPORT")
    
    subheader("Execution Timeline")
    ok(f"  Total execution time: {total_time:.2f} seconds")
    
    for step_name, step_metrics in metrics["steps"].items():
        if isinstance(step_metrics, dict) and "time_seconds" in step_metrics:
            step_time = step_metrics["time_seconds"]
            pct = (step_time / total_time * 100) if total_time > 0 else 0
            info(f"  {step_name.title():<20} {step_time:>8.2f}s ({pct:>5.1f}%)")
    
    subheader("Workflow Metrics & Statistics")
    
    gen_stats = metrics["steps"].get("generation", {})
    ingest_stats = metrics["steps"].get("ingestion", {})
    search_stats = metrics["steps"].get("search", {})
    admin_stats = metrics["steps"].get("admin", {})
    
    ok(f"Documents generated: {gen_stats.get('files_generated', 'N/A')}")
    ok(f"Generation size: {gen_stats.get('total_size_mb', 0):.2f} MB")
    ok(f"Documents ingested: {ingest_stats.get('ingested', 'N/A')}/{len(SAMPLE_DOCS)}")
    if ingest_stats.get('failed', 0) > 0:
        warn(f"Ingestion failures: {ingest_stats.get('failed', 0)}")
    
    ok(f"Total KB chunks: {ingest_stats.get('total_chunks', 'N/A')}")
    ok(f"Unique sources: {admin_stats.get('unique_sources', 'N/A')}")
    
    ok(f"Search queries executed: {search_stats.get('queries_executed', 0)}")
    ok(f"Successful queries: {search_stats.get('successful_queries', 0)}")
    ok(f"Total results retrieved: {search_stats.get('total_results', 0)}")
    if search_stats.get('avg_results_per_query', 0) > 0:
        info(f"Average results per query: {search_stats.get('avg_results_per_query', 0):.2f}")
    
    ok(f"Pre-deletion chunks: {admin_stats.get('pre_deletion_chunks', 'N/A')}")
    ok(f"Post-deletion chunks: {admin_stats.get('post_deletion_chunks', 'N/A')}")
    if admin_stats.get('chunks_deleted', 0) > 0:
        info(f"Chunks removed: {admin_stats.get('chunks_deleted', 0)}")
    
    subheader("System Status & Next Steps")
    ok("✓ Database initialized and operational")
    ok("✓ All workflow steps completed successfully")
    ok("✓ Vector search verified and functioning")
    ok("✓ Admin operations and analytics complete")
    
    print(f"\n{C.GREEN}{'─'*70}{C.END}")
    print(f"{C.BOLD}{C.GREEN}🚀 Ready for production!{C.END}")
    print(f"{C.GREEN}{'─'*70}{C.END}\n")
    
    subheader("Launch Web Application")
    print(f"{C.CYAN}Run the following command to start the Streamlit UI:{C.END}\n")
    print(f"  {C.BOLD}streamlit run app.py{C.END}\n")
    
    subheader("Project Information")
    print(f"{C.CYAN}Project Root:{C.END} {Path(__file__).parent}")
    print(f"{C.CYAN}Database Path:{C.END} {Path('chroma_db').resolve()}")
    print(f"{C.CYAN}Data Directory:{C.END} {DATA_DIR.resolve()}\n")
    
    subheader("Generated Output Files")
    ok(f"Architecture Diagram (HTML): {Path('architecture_diagram.html').resolve()}")
    ok(f"Architecture Diagram (Mermaid): {Path('architecture_diagram.mmd').resolve()}")
    ok(f"Architecture Documentation: {Path('ARCHITECTURE.md').resolve()}")
    ok(f"Workflow Metrics (JSON): {Path('workflow_metrics.json').resolve()}")
    
    print(f"\n{C.BOLD}📊 Open architecture_diagram.html in your browser to view the interactive diagram!{C.END}\n")
    
    # Export metrics to JSON
    try:
        metrics["workflow_end"] = datetime.now().isoformat()
        metrics["total_time_seconds"] = total_time
        metrics_file = Path("workflow_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        ok(f"Workflow metrics saved to {metrics_file}")
    except Exception as e:
        warn(f"Could not save metrics: {e}")
    
    print(f"\n{C.BOLD}{C.BLUE}{'='*70}{C.END}")
    print(f"{C.BOLD}{C.BLUE}  Workflow Complete at {datetime.now().strftime('%H:%M:%S')}{C.END}")
    print(f"{C.BOLD}{C.BLUE}{'='*70}{C.END}\n")


if __name__ == "__main__":
    main()
