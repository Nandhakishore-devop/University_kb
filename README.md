# 🎓 University Knowledge Base (UKB)
## *Modern, AI-Governed, and Secure Document Intelligence*

The **University Knowledge Base (UKB)** is a state-of-the-art RAG platform designed to serve as the unified source of truth for campus rules, policies, and circulars. It features a tri-tier integrity system that prevents contradictions, redundant document processing with real-time feedback, and a secure multi-role ecosystem for Students, Contributors, and Administrators.

---

## 🚀 Key Features

### 🛡️ AI Integrity Guardrails
The UKB features a sophisticated three-tier safety system to ensure every document ingested is unique and consistent:
*   **Tier 1: Duplicate Prevention**: Prevents redundant uploads by calculating document-level similarity scores (Blocking matches >75%).
*   **Tier 2: Semantic Overlap Analysis**: Warns administrators when a new document covers existing topic areas, suggesting consolidation.
*   **Tier 3: AI Contradiction Auditor**: Uses **Groq LLM** to deep-scan for factual conflicts (e.g., fee mismatches or rule changes) across departments and years.

### 👥 Multi-Role Ecosystem
*   **Student Hub**: A high-speed, personalized chatbot using **LangGraph** to provide cited answers to campus queries.
*   **Contributor Portal**: Allows student contributors to submit circulars for review. Includes instant AI auto-metadata extraction and audit reports.
*   **Admin Command Center**: Feature-rich dashboard for bulk ingestion (with streaming progress bars), version management (Supersede/Rollback), and a pending approval queue.

### 🔐 Exclusive Security Architecture
*   **Role-Locking**: Sidebars and input fields are dynamically disabled based on active sessions to prevent permission leakage.
*   **Isolated Sessions**: Switching between Admin and Contributor roles automatically clears sensitive states, ensuring a clean audit trail.
*   **Visibility Gates**: "Pending" submissions are indexed but remain invisible to the public chatbot until verified by a human administrator.

---

## 🛠️ Technology Stack
*   **Language**: Python 3.10+
*   **Framework**: [Streamlit](https://streamlit.io/) (Premium Dark Mode Aesthetics)
*   **Orchestration**: [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
*   **Vector Engine**: [ChromaDB](https://www.trychroma.com/)
*   **Inference**: [Groq Cloud](https://groq.com/) (Llama 3.3 70B / Mixtral 8x7B)
*   **Vision**: Tesseract OCR for scanned PDF processing

---

## 📂 Project Architecture

```text
university_kb/
├── app.py                      # Main UI Hub & Dashboard
├── src/
│   ├── admin_agent.py          # AI Audit & Ingestion Logic
│   ├── ai_classifier.py        # Automated Metadata Extraction
│   ├── chroma_store.py         # Persistent Vector Database API
│   ├── student_agent.py        # LangGraph Stateful Chat Logic
│   └── schemas.py              # Unified Data Models
├── data/                       # Local Logs & Contributor Metadata
└── secure_storage/             # PII-Stripped Document Records
```

---

## 🏁 Quick Start

### 1. Requirements
Ensure you have Python installed and [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for scanned document support.

### 2. Setup Environment
```bash
pip install -r requirements.txt
```

### 3. Configure `.env`
```env
GROQ_API_KEY=gsk_your_key_here
ADMIN_SECRET=your_admin_secret
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 4. Launch
```bash
streamlit run app.py
```

---

## 📝 Governance Flow
1. **Submit**: Contributors upload files; AI generates an **Audit Report**.
2. **Pending**: Files are ingested into the "Shadow Queue" with `pending` status.
3. **Review**: Admins inspect the AI audit findings in the **Approval Queue**.
4. **Publish**: Approval flips the document to `active`, making it instantly searchable by the student chatbot.

---
*Developed for UKB - Modernizing University Knowledge Engagement.*
