"""
retriever.py - LangChain VectorStoreRetriever wrapper around ChromaStore.

Provides a thin layer so the student panel can use LangChain-style retrieval
while benefiting from our custom ChromaStore for stats and admin ops.
"""

from __future__ import annotations
import os
from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger

from src.chroma_store import ChromaStore
from src.schemas import ChunkRecord, SearchFilter, UserProfile


class UniversityRetriever:
    """
    Retrieves document chunks relevant to a student query.

    Access control:
      - Students always see only 'public' documents (access="public").
      - Admin override can see internal docs too.
    """

    def __init__(self, store: ChromaStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Public retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: Optional[SearchFilter] = None,
        top_k: int = 5,
        admin_override: bool = False,
        user_profile: Optional[UserProfile] = None,
    ) -> List[ChunkRecord]:
        """
        Run similarity search with optional metadata filters.

        For student-facing queries, enforces access='public' unless
        admin_override is True.
        """
        if not query.strip():
            return []

        # Enforce public-only and 'active' status for students
        if not admin_override:
            if filters is None:
                filters = SearchFilter(
                    access="public", 
                    status="active", 
                    verification_status="verified"
                )
            else:
                # Clone with safety filters forced
                filters = filters.model_copy(update={
                    "access": "public",
                    "status": "active",
                    "verification_status": "verified"
                })

        results = self._store.similarity_search(
            query=query,
            search_filter=filters,
            top_k=top_k,
        )

        logger.info(
            f"Retrieved {len(results)} chunks for query='{query[:50]}…' "
            f"(admin={admin_override})"
        )
        return results

    def search_events(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[ChunkRecord]:
        """Semantic search specifically for university events and posters."""
        filters = SearchFilter(doc_type="poster", access="public")
        return self.search(query, filters=filters, top_k=top_k)

    def get_recommendations(
        self,
        reference_chunk_id: Optional[str] = None,
        user_profile: Optional[UserProfile] = None,
        top_k: int = 5,
    ) -> List[ChunkRecord]:
        """
        Recommend documents based on a reference chunk or user profile.
        - If reference_chunk_id is provided, finds similar content (item-to-item).
        - If user_profile is provided, finds content aligned with user interests (user-to-item).
        """
        if reference_chunk_id:
            # Find the reference chunk's content
            res = self._store._collection.get(ids=[reference_chunk_id], include=["documents", "metadatas"])
            if not res or not res["documents"]:
                return []
            
            content = res["documents"][0]
            # Search for similar content, excluding the reference chunk itself
            results = self.search(query=content, top_k=top_k + 1)
            recs = [r for r in results if r.chunk_id != reference_chunk_id][:top_k]
            for r in recs:
                r.explanation = f"Similar to content you are currently viewing: '{r.metadata.get('topic')}'"
                r.evidence = ["semantic_similarity", r.metadata.get("topic", "general")]
            return recs

        if user_profile and user_profile.interests:
            # Multi-topic semantic search based on interests
            combined_query = " ".join(user_profile.interests)
            results = self.search(query=combined_query, top_k=top_k)
            for r in results:
                r.explanation = f"Recommended based on your interest in: {', '.join(user_profile.interests[:3])}"
                r.evidence = user_profile.interests
            return results

        return []

    def get_event_summary(self, topic: str) -> dict:
        """
        Produce a trustworthy summary for an event topic, sourced only
        from 'verified' documents (posters, circulars).
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {"summary": "API Key missing.", "verified": False}

        # Filter for verified event docs
        filters = SearchFilter(topic=topic, access="public")
        # Note: SearchFilter needs verification_status handling? 
        # I'll manually filter for now as SearchFilter lacks it.
        
        results = self.search(query=topic, filters=filters, top_k=10)
        verified_records = [r for r in results if r.metadata.get("verification_status") == "verified"]
        
        if not verified_records:
            return {
                "summary": "No verified records found for this event. Summary unavailable for safety.",
                "verified": False,
                "sources": []
            }

        # Summarize via LLM
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        context = "\n\n".join([r.content for r in verified_records])
        llm = ChatOpenAI(model="mixtral-8x7b-32768", temperature=0, api_key=api_key, base_url="https://api.groq.com/openai/v1")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following event based ONLY on the verified records provided. Provide keys: Date, Venue, Goal, and Eligibility."),
            ("human", "Verified Documents:\n{context}")
        ])
        
        try:
            chain = prompt | llm
            response = chain.invoke({"context": context})
            return {
                "summary": response.content,
                "verified": True,
                "sources": [r.metadata.get("source_file") for r in verified_records]
            }
        except Exception as e:
            return {"summary": f"Summary generation failed: {e}", "verified": False}

    # ------------------------------------------------------------------
    # LangChain Document format (for compatibility with chains/agents)
    # ------------------------------------------------------------------

    def search_as_lc_docs(
        self,
        query: str,
        filters: Optional[SearchFilter] = None,
        top_k: int = 5,
        admin_override: bool = False,
    ) -> List[Document]:
        """Same as search() but returns LangChain Document objects."""
        records = self.search(query, filters, top_k, admin_override)
        return [
            Document(
                page_content=r.content,
                metadata={**r.metadata, "score": r.score, "chunk_id": r.chunk_id},
            )
            for r in records
        ]

    # ------------------------------------------------------------------
    # AI Answer (RAG)
    # ------------------------------------------------------------------

    def get_rag_answer(
        self,
        query: str,
        filters: Optional[SearchFilter] = None,
        top_k: int = 5,
    ) -> dict:
        """
        Retrieve relevant chunks and generate an AI answer using RAG.
        Returns a dict: {"answer": str, "sources": List[dict]}
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {
                "answer": "⚠️ RAG answering requires GROQ_API_KEY. Showing retrieved chunks only.",
                "sources": [],
            }

        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        # 1. Retrieve
        records = self.search(query, filters, top_k, admin_override=False)
        if not records:
            return {"answer": "I couldn't find any relevant information in the knowledge base.", "sources": []}

        # 2. Prepare Context
        context_text = "\n\n".join(
            [f"--- SOURCE: {r.metadata.get('source_file')} (Page {r.metadata.get('page_number','?')}) ---\n{r.content}" 
             for r in records]
        )

        # 3. Generate Answer
        llm = ChatOpenAI(
            model="mixtral-8x7b-32768",
            temperature=0,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are the University Knowledge Base Assistant. "
                "Answer the user's question based strictly on the provided context. "
                "If the context doesn't contain the answer, say you don't know. "
                "Cite your sources using the source filename (e.g. [handbook_2024.pdf]). "
                "Keep the answer professional and concise."
            )),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

        try:
            chain = prompt | llm
            response = chain.invoke({"context": context_text, "question": query})
            return {
                "answer": response.content,
                "sources": [r.model_dump() for r in records]
            }
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            return {"answer": f"Error generating answer: {e}", "sources": [r.model_dump() for r in records]}
