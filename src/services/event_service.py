"""
event_service.py - Service for managing event lifecycles.
"""
from typing import Optional
from src.chroma_store import ChromaStore
from src.schemas import DocMetadata

class EventService:
    def __init__(self, store: ChromaStore):
        self.store = store

    def close_event(self, topic: str) -> bool:
        """Close an event and archive its documents."""
        result = self.store._collection.get(
            where={"topic": {"$eq": topic}},
            include=["metadatas"]
        )
        ids = result.get("ids", [])
        if not ids:
            return False
            
        self.store.update_metadata(ids, {"event_status": "closed", "is_archived": True})
        return True
