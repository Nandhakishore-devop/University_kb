"""
profile_service.py - Manages persistent student profiles in the University KB.
Stores and retrieves UserProfile objects from a local JSON database.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from loguru import logger
from src.schemas import UserProfile


class ProfileService:
    """Service to persist and retrieve student profiles."""

    def __init__(self, storage_path: str = "data/profiles.json") -> None:
        self.storage_path = Path(storage_path)
        # Ensure the directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file if not exists
        if not self.storage_path.exists():
            with open(self.storage_path, "w") as f:
                json.dump({}, f)

    def _load_all(self) -> Dict[str, Any]:
        """Read the entire profile store."""
        try:
            with open(self.storage_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
            return {}

    def _save_all(self, data: Dict[str, Any]) -> None:
        """Write the entire profile store."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")

    def get_profile(self, user_id: str) -> UserProfile:
        """Fetch a profile for the given user_id, or create a default one."""
        profiles = self._load_all()
        
        if user_id in profiles:
            p_data = profiles[user_id]
            try:
                # Reconstruct UserProfile object
                return UserProfile(**p_data)
            except Exception as e:
                logger.warning(f"Failed to parse profile for {user_id}: {e}")
        
        # Return a fresh profile if not found
        return UserProfile(user_id=user_id)

    def save_profile(self, profile: UserProfile) -> None:
        """Update or insert a profile in the store."""
        profiles = self._load_all()
        # Convert Pydantic model to dict
        profiles[profile.user_id] = profile.model_dump()
        self._save_all(profiles)
        logger.info(f"Persistent Profile: Saved data for user '{profile.user_id}'")
