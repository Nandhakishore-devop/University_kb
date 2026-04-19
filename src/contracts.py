"""
contracts.py - Service Interface definitions and API contracts.
"""
from typing import Protocol, List, Optional
from src.schemas import KBStats, DocMetadata

class EventServiceInterface(Protocol):
    def close_event(self, topic: str) -> bool: ...

class NotificationServiceInterface(Protocol):
    def notify_user(self, user_id: str, message: str, channel: str = "app") -> bool: ...

class ReportServiceInterface(Protocol):
    def get_kb_health_report(self) -> KBStats: ...
