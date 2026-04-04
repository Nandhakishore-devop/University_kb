"""
test_pipeline.py - End-to-end test script for the University Knowledge Base.

Run with:
  python test_pipeline.py

Tests:
  1. Generate sample documents (if not present)
  2. Ingest all sample documents with correct metadata
  3. Run student search queries with filters
  4. Run admin operations: stats, duplicate detection, deletion
  5. Print formatted output
"""

from __future__ import annotations
import sys
from pathlib import Path
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
    print(f"\n{C.BOLD}{C.HEADER}{'='*60}{C.END}")
    print(f"{C.BOLD}{C.HEADER}  {text}{C.END}")
    print(f"{C.BOLD}{C.HEADER}{'='*60}{C.END}\n")


def ok(text: str) -> None:
    print(f"  {C.GREEN}✓{C.END} {text}")


def info(text: str) -> None:
    print(f"  {C.CYAN}ℹ{C.END} {text}")


def warn(text: str) -> None:
    print(f"  {C.YELLOW}⚠{C.END} {text}")


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

def test_generate() -> None:
    header("TEST 1: Generate Sample Documents")
    generate_all()
    files = list(DATA_DIR.iterdir())
    ok(f"Generated {len(files)} files in {DATA_DIR}")
    for f in files:
        info(f.name)


# ---------------------------------------------------------------------------
# Test 2: Ingest all documents
# ---------------------------------------------------------------------------

def test_ingest(agent: AdminAgent) -> None:
    header("TEST 2: Ingest Sample Documents")

    for doc in SAMPLE_DOCS:
        file_path = DATA_DIR / doc["file"]
        if not file_path.exists():
            # Try .txt fallback
            alt = file_path.with_suffix(".txt")
            if alt.exists():
                file_path = alt
                doc["meta"]["source_file"] = file_path.name
            else:
                warn(f"File not found, skipping: {doc['file']}")
                continue

        result = agent.upsert_doc(str(file_path), doc["meta"])
        if "SUCCESS" in result:
            ok(result)
        else:
            warn(result)


# ---------------------------------------------------------------------------
# Test 3: Student search queries
# ---------------------------------------------------------------------------

def test_search(retriever: UniversityRetriever) -> None:
    header("TEST 3: Student Search Queries")

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

    for q in queries:
        admin = q.get("admin_override", False)
        print(f"\n  {C.BOLD}Query:{C.END} {q['label']}")
        print(f"  Search: '{q['query']}'  admin_override={admin}")

        results = retriever.search(
            query=q["query"],
            filters=q.get("filters"),
            top_k=3,
            admin_override=admin,
        )

        if not results:
            warn("  No results found.")
        else:
            for i, r in enumerate(results, 1):
                access = r.metadata.get("access", "?")
                src = r.metadata.get("source_file", "?")
                score = f"{r.score*100:.1f}%"
                preview = r.content[:120].replace("\n", " ")
                print(
                    f"    {C.CYAN}[{i}]{C.END} {src} | access={access} | score={score}"
                )
                print(f"         {preview}…")


# ---------------------------------------------------------------------------
# Test 4: Admin operations
# ---------------------------------------------------------------------------

def test_admin_ops(agent: AdminAgent) -> None:
    header("TEST 4: Admin Agent Operations")

    # Stats
    print(f"  {C.BOLD}KB Statistics:{C.END}")
    stats_json = agent.kb_stats()
    import json
    stats = json.loads(stats_json)
    ok(f"Total chunks: {stats['total_chunks']}")
    ok(f"Unique sources: {stats['unique_sources']}")
    ok(f"Topics: {stats['topics']}")
    ok(f"Departments: {stats['departments']}")

    # Duplicate detection
    print(f"\n  {C.BOLD}Duplicate Detection (exam_rules / 2023):{C.END}")
    dup_result = agent.detect_duplicates(
        {"topic": "exam_rules", "department": "GENERAL", "year": 2023}
    )
    if "NO_DUPLICATES" in dup_result:
        ok(dup_result)
    else:
        warn(dup_result)

    # Reindex recommendation
    print(f"\n  {C.BOLD}Reindex Recommendation:{C.END}")
    rec = agent.recommend_reindex()
    info(rec)

    # Metadata suggestion from filename
    print(f"\n  {C.BOLD}Metadata Suggestion (auto from filename):{C.END}")
    suggestion = agent.suggest_metadata("circular_fee_structure_2024.pdf")
    ok(f"Suggested: {suggestion}")

    # Delete outdated 2022 exam circular
    print(f"\n  {C.BOLD}Delete Outdated 2022 Exam Circular:{C.END}")
    del_result = agent.delete_by_metadata(
        {"topic": "exam_rules", "year": 2022}
    )
    if "SUCCESS" in del_result:
        ok(del_result)
    else:
        warn(del_result)

    # Stats after deletion
    print(f"\n  {C.BOLD}KB Stats After Deletion:{C.END}")
    stats2 = json.loads(agent.kb_stats())
    ok(f"Total chunks now: {stats2['total_chunks']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{C.BOLD}{C.BLUE}University Knowledge Base — Test Pipeline{C.END}")
    print(f"{C.BLUE}{'─'*60}{C.END}")

    # Initialise
    store = ChromaStore(persist_dir="chroma_db")
    retriever = UniversityRetriever(store=store)
    agent = AdminAgent(store=store)

    # Run tests
    test_generate()
    test_ingest(agent)
    test_search(retriever)
    test_admin_ops(agent)

    header("ALL TESTS COMPLETE")
    ok("Run `streamlit run app.py` to launch the web UI")


if __name__ == "__main__":
    main()
