"""
schemas.py - Pydantic data models for the University Knowledge Base.
Defines DocMetadata, ChunkRecord, and KB statistics schemas.
"""

from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Document-level metadata (supplied by admin at ingest time)
# ---------------------------------------------------------------------------

class DocMetadata(BaseModel):
    """Metadata attached to every chunk of an ingested document."""

    source_file: str = Field(..., description="Original filename (e.g. handbook_2024.pdf)")
    doc_type: Literal["handbook", "circular", "policy", "other"] = Field(
        "other", description="Document category"
    )
    section: str = Field("", description="Section or chapter within the document")
    topic: str = Field("", description="Primary topic tag (e.g. 'exam_rules')")
    year: int = Field(..., description="Publication year (e.g. 2024)")
    department: str = Field("general", description="Owning department (e.g. 'CSE')")
    access: Literal["public", "internal"] = Field(
        "public", description="Visibility scope"
    )
    version: str = Field("1.0", description="Document version string")
    uploaded_time: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO-8601 UTC upload timestamp",
    )

    @model_validator(mode="after")
    def _normalise(self) -> "DocMetadata":
        self.topic = self.topic.lower().strip().replace(" ", "_")
        self.department = self.department.upper().strip()
        return self

    def to_chroma_dict(self) -> dict:
        """Return a flat dict suitable for ChromaDB metadata storage.

        ChromaDB metadata values must be str | int | float | bool.
        """
        return {
            "source_file": self.source_file,
            "doc_type": self.doc_type,
            "section": self.section,
            "topic": self.topic,
            "year": self.year,
            "department": self.department,
            "access": self.access,
            "version": self.version,
            "uploaded_time": self.uploaded_time,
        }


# ---------------------------------------------------------------------------
# A single retrieved chunk with its score
# ---------------------------------------------------------------------------

class ChunkRecord(BaseModel):
    """One chunk returned from a similarity search."""

    chunk_id: str
    content: str
    metadata: dict
    score: float = 0.0


# ---------------------------------------------------------------------------
# KB-level statistics
# ---------------------------------------------------------------------------

class KBStats(BaseModel):
    """Aggregate statistics about the knowledge base."""

    total_chunks: int = 0
    unique_sources: int = 0
    unique_topics: list[str] = Field(default_factory=list)
    unique_departments: list[str] = Field(default_factory=list)
    doc_type_counts: dict[str, int] = Field(default_factory=dict)
    year_range: tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
# Filter object for retrieval & deletion
# ---------------------------------------------------------------------------

class SearchFilter(BaseModel):
    """Structured filter that maps to ChromaDB 'where' clauses."""

    topic: Optional[str] = None
    year: Optional[int] = None
    department: Optional[str] = None
    access: Optional[Literal["public", "internal"]] = None
    doc_type: Optional[str] = None
    version: Optional[str] = None

    def to_where_clause(self) -> Optional[dict]:
        """Convert to a ChromaDB $and where clause; returns None if no filters."""
        conditions = []
        if self.topic:
            conditions.append({"topic": {"$eq": self.topic}})
        if self.year:
            conditions.append({"year": {"$eq": self.year}})
        if self.department:
            conditions.append({"department": {"$eq": self.department.upper()}})
        if self.access:
            conditions.append({"access": {"$eq": self.access}})
        if self.doc_type:
            conditions.append({"doc_type": {"$eq": self.doc_type}})
        if self.version:
            conditions.append({"version": {"$eq": self.version}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
