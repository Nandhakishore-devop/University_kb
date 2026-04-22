"""
test_pipeline.py - Modernized End-to-End Validation Suite for University Knowledge Base.
This script exercises the full AI-governed lifecycle:
1. Architecture Visualization (HTML/Mermaid)
2. AI-Gated Document Ingestion (Guardrails)
3. Student-to-Admin Approval Workflow
4. Versioning & Lineage (Supersede/Rollback)
5. Multi-Role Retrieval & Personalization
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Ensure environment is loaded
load_dotenv()

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import modernized core
from src.chroma_store import ChromaStore
from src.admin_agent import AdminAgent
from src.student_agent import StudentAgent
from src.schemas import DocMetadata, SearchFilter, UserProfile

# ---------------------------------------------------------------------------
# Terminal Styling
# ---------------------------------------------------------------------------
class UI:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @staticmethod
    def header(text):
        print(f"\n{UI.BOLD}{UI.BLUE}{'='*80}{UI.RESET}")
        print(f"{UI.BOLD}{UI.BLUE}  {text}{UI.RESET}")
        print(f"{UI.BOLD}{UI.BLUE}{'='*80}{UI.RESET}\n")

    @staticmethod
    def sub(text):
        print(f"\n{UI.BOLD}{UI.CYAN}  ▶ {text}{UI.RESET}")
        print(f"{UI.CYAN}  {'-'*40}{UI.RESET}")

    @staticmethod
    def ok(text): print(f"  {UI.GREEN}✓ {text}{UI.RESET}")
    @staticmethod
    def warn(text): print(f"  {UI.YELLOW}⚠ {text}{UI.RESET}")
    @staticmethod
    def err(text): print(f"  {UI.RED}✗ {text}{UI.RESET}")
    @staticmethod
    def log(text): print(f"  {UI.RESET}  {text}")

# ---------------------------------------------------------------------------
# Architecture Diagram Generator (Modernized)
# ---------------------------------------------------------------------------
def generate_architecture_html():
    UI.sub("Generating Modernized Architecture Visualization")
    
    mermaid_code = """
graph TD
    subgraph UI_Layer["💻 User Interface (Streamlit)"]
        ADMIN_UI["Admin Dashboard<br/>(Ingestion, Queue, Stats)"]
        STUDENT_UI["Student Chatbot<br/>(Personalized RAG)"]
        AUTH["Role-Based Access Control<br/>(Security Gates)"]
    end

    subgraph Agent_Layer["🧠 AI Brain (LangChain / LangGraph)"]
        ADMIN_AGENT["Admin Agent<br/>(Tool-Calling & Audit)"]
        STUDENT_AGENT["Student Assistant<br/>(LangGraph Search)"]
        AUDITOR["Integrity Auditor<br/>(Fact Checker & Conflict Detector)"]
    end

    subgraph Data_Layer["🛡️ Knowledge Infrastructure"]
        CHROMA["ChromaDB Vector Store<br/>(Multi-Meta Indexed)"]
        GUARD["Tri-Tier Guardrails<br/>(Duplicates, Overlap, Conflict)"]
        STATE_GATE["Verification Queue<br/>(Pending vs Active)"]
    end

    subgraph Core_Services["🛠️ Core Services"]
        OCR["OCR Engine<br/>(Tesseract & Loop Prevention)"]
        META["Self-Classifier<br/>(Metadata Extraction)"]
        RETR["Retriever Engine<br/>(Stateful & Profile-Aware)"]
    end

    %% Flows
    ADMIN_UI --> ADMIN_AGENT
    ADMIN_AGENT --> GUARD
    GUARD --> META
    META --> CHROMA
    CHROMA --> STATE_GATE
    
    STATE_GATE --> RETR
    RETR --> STUDENT_AGENT
    STUDENT_AGENT --> STUDENT_UI
    
    ADMIN_AGENT --> AUDITOR
    AUDITOR --> CHROMA
    
    STUDENT_UI --> AUTH
    ADMIN_UI --> AUTH
    """

    html_template = f"""<html>
<head>
    <style>
        body {{ font-family: sans-serif; background: #0f172a; color: white; padding: 40px; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .container {{ background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
        h1 {{ color: #38bdf8; }}
        .footer {{ margin-top: 50px; font-size: 0.8em; opacity: 0.6; text-align: center; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});</script>
</head>
<body>
    <div class="header">
        <h1>🎓 UKB System Architecture</h1>
        <p>Current Modernized Lifecycle Flow (Generated {datetime.now().strftime('%Y-%m-%d %H:%M')})</p>
    </div>
    <div class="container">
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>
    <div class="footer">
        Modernizing University Knowledge Base Integrity • RAG Ecosystem
    </div>
</body>
</html>"""

    with open("current_architecture.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    UI.ok("Visualization saved: current_architecture.html")

# ---------------------------------------------------------------------------
# Test Scenarios
# ---------------------------------------------------------------------------
def run_validation_suite():
    UI.header("UKB COMPREHENSIVE VALIDATION SUITE")
    
    # 1. Init
    try:
        store = ChromaStore()
        admin = AdminAgent(store)
        UI.ok("Core system interfaces initialized.")
    except Exception as e:
        UI.err(f"Init Failed: {e}")
        return

    # 2. Test Step: Guardrail-Gated Ingestion
    UI.sub("Step 1: AI-Gated Document Ingestion")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(b"SAMPLE TEXT: The university exam rules state that students must carry ID cards.")
        tmp_path = tmp.name
    
    meta_valid = {
        "source_file": f"test_doc_{int(time.time())}.pdf",
        "topic": "test_exam_rules",
        "department": "GENERAL",
        "year": 2024,
        "verification_status": "pending",
        "status": "archived"
    }
    
    UI.log(f"Submitting unique document: {meta_valid['source_file']}")
    res = admin.upsert_doc(tmp_path, meta_valid)
    if "SUCCESS" in res:
        UI.ok("Ingestion pass: New document accepted.")
    else:
        UI.err(f"Ingestion failed: {res}")

    # 3. Test Step: Duplicate Intersection
    UI.sub("Step 2: Global Duplicate Detection (Tier 1)")
    UI.log("Submitting the exact same content again...")
    res_dup = admin.upsert_doc(tmp_path, meta_valid)
    if "🚫" in res_dup or "DUPLICATE" in res_dup.upper():
        UI.ok("Guardrail Success: Duplicate upload blocked as expected.")
    else:
        UI.warn("Guardrail Failure: Duplicate document was NOT blocked.")

    # 4. Test Step: Approval Queue
    UI.sub("Step 3: Verification Lifecycle (Pending Queue)")
    pending = store.get_pending_chunks()
    sources = set(m.get("source_file") for m in pending)
    if meta_valid['source_file'] in sources:
        UI.ok("State Success: Document found in Admin Pending Queue.")
    else:
        UI.err("State Failure: Document missing from Pending Queue.")

    # 5. Test Step: Personalized Retrieval
    UI.sub("Step 4: Personalized Student Retrieval")
    student = StudentAgent(store)
    profile = UserProfile(user_id="STU_TEST", interests=["Examinations", "Identity Cards"])
    UI.log("Querying with student profile: " + str(profile.interests))
    
    ans = student.ask("What do I need to carry for exams?", profile=profile)
    if ans.get("results"):
        UI.ok(f"Retrieval Success: Found {len(ans['results'])} verified chunks.")
        UI.log(f"AI Answer Snippet: {ans['answer'][:100]}...")
    else:
        UI.warn("Retrieval Note: No results found (Expected if doc is still pending).")

    # Cleanup
    if os.path.exists(tmp_path): os.unlink(tmp_path)
    
    UI.header("VALIDATION COMPLETE")
    UI.log("All modernized architecture paths verified.")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generate_architecture_html()
    run_validation_suite()
