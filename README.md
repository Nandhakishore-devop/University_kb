# 🎓 University Knowledge Base (UKB)
## Modern, Conversational, and Secure Document Intelligence

A state-of-the-art Knowledge Base platform for universities, featuring a **LangGraph-powered Student Chatbot** and a **Robust AI Administrative Assistant**. The system leverages **ChromaDB** for vector storage, **Groq** for high-speed inference, and **Streamlit** for a premium user experience.

---

## 🌟 Key Pillars

### 🤖 1. Student Campus Chatbot (LangGraph)
*   **Conversational Logic**: Not just a search bar, but a stateful AI assistant that remembers conversation context.
*   **Verified Sources**: Every answer is backed by document citations available in expandable "Verified Sources" blocks.
*   **Persistent Profiles**: Student interests (e.g., "Scholarships", "Campus Life") persist across sessions via a secure JSON profile layer.

### 🛡️ 2. Robust Admin AI Agent
*   **Triple-Layer Execution**: Guaranteed tool execution using a fallback chain: LangGraph ReAct → Legacy AgentExecutor → Manual ReAct Orchestration.
*   **Super-Admin Identity**: Confidently manages the KB with natural language: *"Show stats"*, *"Delete all 2022 exam rules"*, *"Scan for duplicates"*.
*   **Diagnostic Visibility**: Built-in "Agent Logs" viewer for real-time traceability of AI thought processes.

### 📊 3. Modern Infrastructure
*   **Hybrid Search**: High-performance vector search (ChromaDB) with metadata filtering (Topic, Dept, Year).
*   **Audit & Analytics**: Transparent administrative action logging and engagement metrics (Resolution Rate, AVG Chunks).
*   **Secure Access**: Partitioned access with Student ID authentication and Admin Secret protection.

---

## 📂 Project Structure

```text
university_kb/
├── app.py                     # Main UI (Entry point)
├── data/                      # Source documents, user profiles, and logs
├── src/
│   ├── student_agent.py       # LangGraph stateful student logic
│   ├── admin_agent.py         # Robust tool-calling admin agent
│   ├── chroma_store.py        # Vector database management
│   ├── retriever.py           # Personalized search engine
│   └── services/              # Profile, Report, Audit services
└── tests/                     # Verification scripts
```

---

## 🚀 Quick Start

### 1. Prerequisites
*   Python 3.10+
*   Tesseract OCR (Optional, for scanned PDFs)

### 2. Installation
```bash
# Clone and enter directory
cd university_kb

# Setup virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root based on `.env.example`:
```env
GROQ_API_KEY=gsk_...        # Required for Chatbot/Agent
ADMIN_SECRET=university     # Key to access Admin Panel
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe  # For OCR
```

### 4. Preparation & Launch
```bash
# Generate sample data (Optional)
python generate_sample_data.py

# Launch the platform
streamlit run app.py
```

---

## 🖥️ User Roles

### **Students**
- **Login**: Enter any Student ID (e.g., `STU123`).
- **Personalize**: Set interests in the sidebar to re-rank search results.
- **Chat**: Use the "Campus Chatbot" for natural language queries about policies.

### **Administrators**
- **Login**: Use the `ADMIN_SECRET` defined in your `.env`.
- **Manage**: Upload docs, edit metadata, or sweep for duplicates via the dashboard.
- **Automate**: Use the **Admin AI Agent** to perform complex clean-up tasks via chat.

---

## 🛠️ Tech Stack
*   **Framework**: [Streamlit](https://streamlit.io/)
*   **LLM Orchestration**: [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
*   **Vector Store**: [ChromaDB](https://www.trychroma.com/)
*   **Model Provider**: [Groq](https://groq.com/) (Llama 3.3 70B & Mixtral 8x7B)

---

## 📝 License
MIT License - Developed for **Modernizing University Knowledge Base**.
