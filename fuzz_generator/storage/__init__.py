"""Storage module for persistence and caching."""

from fuzz_generator.storage.base import StorageBackend, StorageError
from fuzz_generator.storage.cache import CacheManager
from fuzz_generator.storage.json_storage import JsonStorage

__all__ = [
    "StorageBackend",
    "StorageError",
    "JsonStorage",
    "CacheManager",
]
