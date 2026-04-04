"""
chroma_store.py - ChromaDB persistent vector store with upsert/delete/query.

Wraps LangChain's Chroma integration and exposes clean methods for the
ingestion pipeline, student retriever, and admin agent tools.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from loguru import logger

from src.schemas import ChunkRecord, DocMetadata, KBStats, SearchFilter
from src.utils import ensure_dir


# ---------------------------------------------------------------------------
# Embedding model factory
# ---------------------------------------------------------------------------

def _get_embedding_function(use_openai: bool = False):
    """Return an embedding function compatible with ChromaDB's raw client.

    Priority:
      1. Grok (if use_openai=True and GROK_API_KEY set)
      2. sentence-transformers local model (always available)
    """
    if use_openai and os.getenv("GROK_API_KEY"):
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            logger.info("Using Grok embeddings (xAI)")
            return OpenAIEmbeddingFunction(
                api_key=os.getenv("GROK_API_KEY"),
                model_name="text-embedding-3-small",
                api_base="https://api.x.ai/v1",
            )
        except Exception as e:
            logger.warning(f"Grok embedding init failed ({e}); falling back to local")

    # Local sentence-transformers fallback
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        logger.info("Using local sentence-transformers embeddings (all-MiniLM-L6-v2)")
        return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    except ImportError:
        raise ImportError(
            "Install sentence-transformers: pip install sentence-transformers"
        )


# ---------------------------------------------------------------------------
# ChromaStore class
# ---------------------------------------------------------------------------

COLLECTION_NAME = "university_kb"


class ChromaStore:
    """Persistent ChromaDB collection with upsert, delete, and search."""

    def __init__(
        self,
        persist_dir: str = "chroma_db",
        use_openai: bool = False,
    ) -> None:
        self._persist_dir = str(ensure_dir(persist_dir))
        self._embed_fn = _get_embedding_function(use_openai)

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Handle embedding function conflicts by deleting and recreating collection
        try:
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except ValueError as e:
            if "embedding function" in str(e).lower():
                logger.warning(
                    f"Embedding function conflict detected: {e}. "
                    f"Deleting and recreating collection with new embedding function."
                )
                # Delete the old collection
                try:
                    self._client.delete_collection(name=COLLECTION_NAME)
                    logger.info(f"Deleted old collection '{COLLECTION_NAME}'")
                except Exception as del_err:
                    logger.warning(f"Could not delete collection: {del_err}")
                
                # Create new collection with new embedding function
                self._collection = self._client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=self._embed_fn,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(f"Created new collection '{COLLECTION_NAME}' with {self._embed_fn.name() if hasattr(self._embed_fn, 'name') else 'custom'} embedding")
            else:
                raise
        
        logger.info(
            f"ChromaStore ready — collection '{COLLECTION_NAME}' "
            f"({self._collection.count()} chunks) at {self._persist_dir}"
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: List[dict]) -> int:
        """Insert or update chunks. Returns count added/updated."""
        if not chunks:
            return 0

        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(chunks)} chunks")
        return len(chunks)

    def delete_by_filter(self, search_filter: SearchFilter) -> int:
        """Delete all chunks matching the given filter. Returns deleted count.

        ChromaDB's delete requires explicit IDs, so we first query matching
        IDs then delete them.
        """
        where = search_filter.to_where_clause()
        if where is None:
            raise ValueError("delete_by_filter requires at least one filter condition")

        # Fetch IDs matching the filter (get up to 10 000 chunks)
        result = self._collection.get(
            where=where,
            limit=10_000,
            include=[],  # IDs only
        )
        ids = result.get("ids", [])
        if not ids:
            logger.info("delete_by_filter: no matching chunks found")
            return 0

        self._collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} chunks matching filter {where}")
        return len(ids)

    # ------------------------------------------------------------------
    # Read / search operations
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        search_filter: Optional[SearchFilter] = None,
        top_k: int = 5,
    ) -> List[ChunkRecord]:
        """Return top-k chunks most similar to query, with metadata & score."""
        where = search_filter.to_where_clause() if search_filter else None

        kwargs: dict = {
            "query_texts": [query],
            "n_results": min(top_k, max(1, self._collection.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            result = self._collection.query(**kwargs)
        except Exception as e:
            logger.error(f"similarity_search failed: {e}")
            return []

        records: List[ChunkRecord] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # Cosine distance → similarity score (0–1, higher = more similar)
            score = max(0.0, 1.0 - dist)
            records.append(
                ChunkRecord(chunk_id=cid, content=doc, metadata=meta, score=score)
            )

        return records

    def get_all_metadata(self, limit: int = 50_000) -> List[dict]:
        """Fetch all stored metadata dicts (no embeddings)."""
        result = self._collection.get(limit=limit, include=["metadatas"])
        return result.get("metadatas", [])

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> KBStats:
        """Compute aggregate KB statistics from stored metadata."""
        all_meta = self.get_all_metadata()
        if not all_meta:
            return KBStats()

        sources = {m.get("source_file", "") for m in all_meta}
        topics = sorted({m.get("topic", "") for m in all_meta if m.get("topic")})
        departments = sorted(
            {m.get("department", "") for m in all_meta if m.get("department")}
        )

        doc_type_counts: dict[str, int] = {}
        years = []
        for m in all_meta:
            dt = m.get("doc_type", "other")
            doc_type_counts[dt] = doc_type_counts.get(dt, 0) + 1
            y = m.get("year", 0)
            if y:
                years.append(y)

        year_range = (min(years), max(years)) if years else (0, 0)

        return KBStats(
            total_chunks=len(all_meta),
            unique_sources=len(sources),
            unique_topics=topics,
            unique_departments=departments,
            doc_type_counts=doc_type_counts,
            year_range=year_range,
        )

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Duplicate detection helpers
    # ------------------------------------------------------------------

    def find_chunks_by_source(self, source_file: str) -> List[dict]:
        """Return metadata for all chunks from a given source file."""
        try:
            result = self._collection.get(
                where={"source_file": {"$eq": source_file}},
                limit=10_000,
                include=["metadatas"],
            )
            return result.get("metadatas", [])
        except Exception:
            return []

    def find_similar_docs(
        self, topic: str, department: str, year: int
    ) -> List[dict]:
        """Find existing docs with same topic+department (any year)."""
        try:
            result = self._collection.get(
                where={
                    "$and": [
                        {"topic": {"$eq": topic}},
                        {"department": {"$eq": department.upper()}},
                    ]
                },
                limit=5_000,
                include=["metadatas"],
            )
            metas = result.get("metadatas", [])
            # De-duplicate by source_file + version
            seen = set()
            unique = []
            for m in metas:
                key = (m.get("source_file"), m.get("version"))
                if key not in seen:
                    seen.add(key)
                    unique.append(m)
            return unique
        except Exception:
            return []
