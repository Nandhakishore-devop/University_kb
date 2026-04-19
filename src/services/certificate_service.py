"""
certificate_service.py - Service for managing certificate eligibility.
"""
from src.chroma_store import ChromaStore
from src.services.attendance_service import AttendanceService

class CertificateService:
    def __init__(self, store: ChromaStore):
        self.store = store
        self.attendance_service = AttendanceService(store)

    def check_eligibility(self, user_id: str, topic: str) -> dict:
        """
        Check if a student is eligible for a certificate and return a clear message.
        """
        # 1. Fetch attendance records
        attendance = self.attendance_service.get_attendance_summary(topic)
        total_sessions = len(attendance)  # Simplified: count chunks
        
        # Mock logic for threshold
        at_threshold = 0.85 
        # Calculate actual attendance for user_id...
        
        # Return clear feedback
        return {
            "eligible": False,
            "message": f"Ineligible for '{topic}' certificate. Requirements: 85% attendance. You achieved: 60%.",
            "required": "85%",
            "achieved": "60%",
            "missing": "25%"
        }
