"""
schemas.py - Pydantic data models for the University Knowledge Base.
Defines DocMetadata, ChunkRecord, and KB statistics schemas.
"""

from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Document-level metadata (supplied by admin at ingest time)
# ---------------------------------------------------------------------------

class DocMetadata(BaseModel):
    """Metadata attached to every chunk of an ingested document."""

    source_file: str = Field(..., description="Original filename (e.g. handbook_2024.pdf)")
    doc_type: Literal["handbook", "circular", "policy", "poster", "attendance", "certificate", "event", "other"] = Field(
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
    status: Literal["active", "archived", "obsolete"] = Field(
        "active", description="Lifecycle status"
    )
    effective_date: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat().split("T")[0],
        description="Date document guidance takes effect"
    )
    superseded_by: Optional[str] = Field(None, description="ID of the document that replaced this one")
    uploaded_time: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO-8601 UTC upload timestamp",
    )
    contributor_id: str = Field("system", description="User/Volunteer ID who ingested doc")
    verification_status: Literal["pending", "verified"] = Field(
        "pending", description="Status of human verification"
    )
    verification_time: Optional[str] = Field(None, description="ISO-8601 timestamp of verification")
    is_archived: bool = Field(False, description="Deprecated in favor of 'status'")
    quality_score: float = Field(1.0, description="Data quality score (0.0-1.0)")
    priority: Literal["high", "medium", "low"] = Field("medium", description="Document priority")
    audit_report: Optional[str] = Field(None, description="AI-generated audit summary for contributors")
    parent_doc_id: Optional[str] = Field(None, description="Reference to parent doc (for versioning/dedup)")
    
    # Event Operations
    waitlist_capacity: int = Field(0, description="Max waitlist size")
    event_status: Literal["draft", "open", "closed", "cancelled"] = Field("draft")
    eligibility_rules: Optional[str] = Field(None, description="Natural language eligibility rules")

    @model_validator(mode="after")
    def _normalise(self) -> "DocMetadata":
        self.topic = self.topic.lower().strip().replace(" ", "_")
        self.department = self.department.upper().strip()
        return self

    def to_chroma_dict(self) -> dict:
        """Flatten for ChromaDB storage (converting bools to int, etc)."""
        data = self.model_dump()
        # Convert types Chroma might struggle with (depending on version)
        # though modern Chroma is better, flat and simple is safer.
        flat = {}
        for k, v in data.items():
            if isinstance(v, bool):
                flat[k] = int(v)
            elif v is None:
                flat[k] = ""
            elif isinstance(v, (list, dict)):
                import json
                flat[k] = json.dumps(v)
            else:
                flat[k] = v
        return flat


# ---------------------------------------------------------------------------
# Search / Chunk results
# ---------------------------------------------------------------------------

class ChunkRecord(BaseModel):
    """A single retrieved document chunk with associated metadata and relevance score."""

    chunk_id: str
    content: str
    metadata: dict
    score: float = 0.0
    explanation: Optional[str] = Field(None, description="Reason for recommendation")
    evidence: list[str] = Field(default_factory=list, description="Matching terms or activity tags")


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
    verified_chunks: int = 0
    unique_contributors: int = 0
    archived_chunks: int = 0
    obsolete_chunks: int = 0


class ParticipationStats(BaseModel):
    """Detailed engagement and volunteer metrics."""
    query_resolution_rate: float = 0.0  # % of queries resolved by AI
    monthly_active_users: int = 0
    volunteer_retention_rate: float = 0.0
    contributors_per_dept: dict[str, int] = Field(default_factory=dict)
    avg_verified_per_volunteer: float = 0.0
    departmental_parity_gap: list[str] = Field(default_factory=list) # Depts with < 3 contributors


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
    status: Optional[str] = None
    is_archived: Optional[bool] = None
    verification_status: Optional[str] = None
    priority: Optional[str] = None

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
        if self.status:
            conditions.append({"status": {"$eq": self.status}})
        if self.is_archived is not None:
            conditions.append({"is_archived": {"$eq": self.is_archived}})
        if self.verification_status:
            conditions.append({"verification_status": {"$eq": self.verification_status}})
        if self.priority:
            conditions.append({"priority": {"$eq": self.priority}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


# ---------------------------------------------------------------------------
# Personalization & Activity Tracking
# ---------------------------------------------------------------------------

class UserActivity(BaseModel):
    """Tracks a single user interaction with a document/chunk."""
    user_id: str
    chunk_id: str
    interaction_type: Literal["view", "search", "click", "feedback"]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    query: Optional[str] = None


class UserProfile(BaseModel):
    """Aggregated user interests and activity summary."""
    user_id: str
    interests: list[str] = Field(default_factory=list)  # Top topics
    interest_vector: Optional[list[float]] = None      # Aggregated embedding
    last_active: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    activity_count: int = 0
    privacy_opt_out: bool = Field(False, description="Whether to disable activity tracking")


class StudentState(TypedDict):
    """State managed by the LangGraph Student Agent."""
    query: str
    user_profile: UserProfile
    results: list[ChunkRecord]
    answer: Optional[str]
    history: list[dict]


class AuditLog(BaseModel):
    """Tracks administrative and sensitive system actions."""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    actor_id: str
    action: str
    target_id: str
    details: Optional[str] = None


# ---------------------------------------------------------------------------
# Bulk Ingestion Monitoring
# ---------------------------------------------------------------------------

class FileIngestionStatus(BaseModel):
    """Tracks the outcome of a single file in a batch ingestion job."""
    filename: str
    status: Literal["success", "failed", "unsupported"]
    chunks: int = 0
    error: Optional[str] = None


class IngestionJobReport(BaseModel):
    """Aggregate statistics for a bulk ingestion job."""
    job_id: str
    start_time: str
    end_time: Optional[str] = None
    total_files: int
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    unsupported_files: int = 0
    total_chunks: int = 0
    file_details: list[FileIngestionStatus] = Field(default_factory=list)
