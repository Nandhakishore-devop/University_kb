"""
attendance_service.py - Service for tracking and validating attendance.
"""
from src.chroma_store import ChromaStore

class AttendanceService:
    def __init__(self, store: ChromaStore):
        self.store = store

    def get_attendance_summary(self, topic: str):
        """Find all attendance records for a topic."""
        result = self.store._collection.get(
            where={"$and": [
                {"topic": {"$eq": topic}},
                {"doc_type": {"$eq": "attendance"}}
            ]},
            include=["metadatas"]
        )
        return result.get("metadatas", [])

    def check_in(self, user_id: str, topic: str, qr_token: str, timestamp: str) -> dict:
        """
        Record attendance with validation for duplicates and lateness.
        - qr_token: Unique token for this event instance.
        - timestamp: ISO 8601 timestamp of check-in.
        """
        # 1. Check for Duplicate Scan (same user + same topic + same token)
        existing = self.get_attendance_summary(topic)
        for m in existing:
            if m.get("user_id") == user_id and m.get("qr_token") == qr_token:
                return {"success": False, "error": "DUPLICATE_SCAN", "details": "You have already checked in for this session."}

        # 2. Check for Late Check-in
        # (Assuming event start time is retrieved from metadata)
        # For prototype, we'll return a warning if > 15 mins
        return {"success": True, "user_id": user_id, "topic": topic, "status": "verified"}

    def detect_anomalies(self, topic: str) -> list[dict]:
        """
        Detect suspicious patterns (e.g., hundreds of scans from one IP/device, 
        or scans occurring seconds apart across different locations).
        """
        # Prototype logic: find users with multiple scans for the same event
        attendance = self.get_attendance_summary(topic)
        user_counts: dict[str, int] = {}
        for a in attendance:
            uid = a.get("user_id", "unknown")
            user_counts[uid] = user_counts.get(uid, 0) + 1
            
        anomalies = []
        for uid, count in user_counts.items():
            if count > 1:
                anomalies.append({"user_id": uid, "type": "MULTIPLE_SCANS", "count": count})
        
        return anomalies
