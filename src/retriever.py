"""
retriever.py - LangChain VectorStoreRetriever wrapper around ChromaStore.

Provides a thin layer so the student panel can use LangChain-style retrieval
while benefiting from our custom ChromaStore for stats and admin ops.
"""

from __future__ import annotations
from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger

from src.chroma_store import ChromaStore
from src.schemas import ChunkRecord, SearchFilter


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
    ) -> List[ChunkRecord]:
        """
        Run similarity search with optional metadata filters.

        For student-facing queries, enforces access='public' unless
        admin_override is True.
        """
        if not query.strip():
            return []

        # Enforce public-only for students
        if not admin_override:
            if filters is None:
                filters = SearchFilter(access="public")
            else:
                # Clone with access forced to public
                filters = filters.model_copy(update={"access": "public"})

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
