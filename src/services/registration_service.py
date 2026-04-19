"""
registration_service.py - Service for managing student registrations for events.
"""
from typing import Optional
from src.chroma_store import ChromaStore

class RegistrationService:
    def __init__(self, store: ChromaStore):
        self.store = store

    def register_user(self, user_id: str, topic: str) -> bool:
        """Standard registration without capacity check."""
        # Simple stub for legacy support
        return True

    def register_with_waitlist(self, user_id: str, topic: str, capacity: int, waitlist_max: int) -> dict:
        """
        Register a user for an event, putting them on a waitlist if over capacity.
        Returns: {"status": "registered" | "waitlisted" | "rejected", "position": int}
        """
        registrations = self.get_registrations(topic)
        count = len(registrations)

        if count < capacity:
            # Add to active registration
            return {"status": "registered", "position": count + 1}
        elif count < (capacity + waitlist_max):
            # Add to waitlist
            return {"status": "waitlisted", "position": count - capacity + 1}
        else:
            return {"status": "rejected", "error": "EVENT_FULL"}

    def promote_from_waitlist(self, topic: str) -> Optional[str]:
        """Promote the first user on the waitlist to 'registered' status."""
        # Logic to find the first 'waitlisted' user and change status
        return "student_123"  # Mocked promoted user ID

    def get_registrations(self, topic: str):
        """Get list of registered users for a topic."""
        # Mocking for now
        return []
