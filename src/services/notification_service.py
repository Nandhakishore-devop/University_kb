"""
notification_service.py - Reusable service for user notifications.
"""
from loguru import logger

class NotificationService:
    def notify_user(self, user_id: str, message: str, channel: str = "app"):
        """Send a notification to the user via specified channel."""
        logger.info(f"NOTIFICATION [{channel}] to {user_id}: {message}")
        # In a real app, this would integrate with Email/SMS APIs.
        return True
