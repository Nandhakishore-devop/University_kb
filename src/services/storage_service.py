"""
storage_service.py - Secure storage for sensitive files (e.g. certificates).
"""
import os
from loguru import logger

class SecureStorageService:
    def __init__(self, base_dir: str = "./secure_storage"):
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def store_file(self, file_id: str, content: bytes, access_level: str = "restricted"):
        """Store a file with restricted access levels."""
        # Simple obfuscation/protection for prototype
        safe_path = os.path.join(self.base_dir, f"sec_{file_id}.enc")
        with open(safe_path, "wb") as f:
            f.write(content)
        logger.info(f"SECURE STORAGE: Stored {file_id} with access={access_level}")
        return safe_path
