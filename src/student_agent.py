"""
student_agent.py - LangGraph-powered Student Assistant for the University KB.
Manages stateful search flows, personalization, and conversational retrieval.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.chroma_store import ChromaStore
from src.retriever import UniversityRetriever
from src.schemas import StudentState, UserProfile, ChunkRecord


class StudentAgent:
    """
    A stateful AI agent for students that manages search results and conversation
    using LangGraph.
    """

    def __init__(self, store: ChromaStore) -> None:
        self._store = store
        self._retriever = UniversityRetriever(store)
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Define the LangGraph workflow for student search."""
        workflow = StateGraph(StudentState)

        # 1. Define nodes
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("personalize", self._node_personalize)
        workflow.add_node("respond", self._node_respond)

        # 2. Define edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "personalize")
        workflow.add_edge("personalize", "respond")
        workflow.add_edge("respond", END)

        return workflow.compile()

    # --- Nodes ---

    def _node_retrieve(self, state: StudentState) -> Dict[str, Any]:
        """Fetch chunks from the vector store based on query."""
        logger.debug(f"Graph[retrieve]: query='{state['query']}'")
        # We fetch more chunks to allow for personalization re-ranking
        results = self._retriever.search(
            query=state["query"],
            top_k=5,
            user_profile=state["user_profile"]
        )
        return {"results": results}

    def _node_personalize(self, state: StudentState) -> Dict[str, Any]:
        """Apply additional state-based logic or filtering (managed results)."""
        # The retriever already does some re-ranking, but we can add
        # custom 'management' logic here if needed (e.g. filtering based on history)
        logger.debug(f"Graph[personalize]: processing {len(state['results'])} results")
        return {"results": state["results"]}

    def _node_respond(self, state: StudentState) -> Dict[str, Any]:
        """Generate a natural language answer if LLM is available."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or not state["results"]:
            return {"answer": None}

        try:
            llm = ChatOpenAI(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                temperature=0.2,
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
            )

            context = "\n\n".join([f"Source: {r.metadata.get('source_file')}\nContent: {r.content}" for r in state["results"]])
            
            system_msg = (
                "You are the University Campus Guide. Use the following context to answer the student's question faithfully. "
                "If the answer isn't in the context, say you don't know based on available documents. "
                "Be helpful, concise, and professional.\n\n"
                f"Context:\n{context}"
            )
            
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=state["query"])
            ]
            
            response = llm.invoke(messages)
            
            # Update history
            new_history = state.get("history", []) + [
                {"role": "user", "content": state["query"]},
                {"role": "assistant", "content": response.content}
            ]
            
            return {"answer": response.content, "history": new_history}
        except Exception as e:
            logger.error(f"Graph[respond] error: {e}")
            return {"answer": "I'm sorry, I encountered an error while processing your request."}

    # --- Public API ---

    def ask(self, query: str, profile: UserProfile, history: List[dict] = None) -> Dict[str, Any]:
        """Entry point for the Streamlit UI to trigger the graph."""
        initial_state: StudentState = {
            "query": query,
            "user_profile": profile,
            "results": [],
            "answer": None,
            "history": history or []
        }
        return self._graph.invoke(initial_state)
