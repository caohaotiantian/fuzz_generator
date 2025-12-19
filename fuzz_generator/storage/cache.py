"""Cache management for analysis results."""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fuzz_generator.storage.json_storage import JsonStorage
from fuzz_generator.utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Cache manager for analysis results.

    Provides caching functionality with optional expiration and
    content-based invalidation.
    """

    # Cache categories
    FUNCTIONS = "functions"
    DATAFLOW = "dataflow"
    CALLGRAPH = "callgraph"
    ANALYSIS = "analysis"

    def __init__(
        self,
        base_dir: str | Path,
        expiry_hours: int = 0,
    ):
        """Initialize cache manager.

        Args:
            base_dir: Base directory for cache storage
            expiry_hours: Hours until cache expires (0 = never)
        """
        self.storage = JsonStorage(Path(base_dir) / "cache")
        self.expiry_hours = expiry_hours

    @staticmethod
    def compute_cache_key(
        project_path: str,
        function_name: str,
        source_content: str | None = None,
    ) -> str:
        """Compute a cache key based on content.

        Args:
            project_path: Path to the project
            function_name: Name of the function
            source_content: Optional source code content for invalidation

        Returns:
            Computed cache key (16-char hash)
        """
        content = f"{project_path}:{function_name}"
        if source_content:
            content += f":{source_content}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _is_expired(self, stored_at: datetime) -> bool:
        """Check if a cached item is expired.

        Args:
            stored_at: When the item was stored

        Returns:
            True if expired, False otherwise
        """
        if self.expiry_hours <= 0:
            return False

        expiry_time = stored_at + timedelta(hours=self.expiry_hours)
        return datetime.now() > expiry_time

    async def get(
        self,
        key: str,
        category: str = ANALYSIS,
    ) -> Any | None:
        """Get cached data.

        Args:
            key: Cache key
            category: Cache category

        Returns:
            Cached data or None if not found/expired
        """
        data = await self.storage.load(category, key)

        if data is None:
            logger.debug(f"Cache miss: {category}/{key}")
            return None

        # Check expiration
        metadata = await self.storage.get_metadata(category, key)
        if metadata and self._is_expired(metadata.created_at):
            logger.debug(f"Cache expired: {category}/{key}")
            await self.storage.delete(category, key)
            return None

        logger.debug(f"Cache hit: {category}/{key}")
        return data

    async def set(
        self,
        key: str,
        data: Any,
        category: str = ANALYSIS,
    ) -> None:
        """Set cached data.

        Args:
            key: Cache key
            data: Data to cache
            category: Cache category
        """
        await self.storage.save(category, key, data)
        logger.debug(f"Cache set: {category}/{key}")

    async def invalidate(self, key: str, category: str = ANALYSIS) -> bool:
        """Invalidate a cache entry.

        Args:
            key: Cache key
            category: Cache category

        Returns:
            True if invalidated, False if not found
        """
        result = await self.storage.delete(category, key)
        if result:
            logger.debug(f"Cache invalidated: {category}/{key}")
        return result

    async def invalidate_category(self, category: str) -> int:
        """Invalidate all entries in a category.

        Args:
            category: Category to invalidate

        Returns:
            Number of entries invalidated
        """
        count = await self.storage.clear_category(category)
        logger.info(f"Cache category '{category}' invalidated: {count} entries")
        return count

    async def clear(self) -> int:
        """Clear all cache.

        Returns:
            Number of entries cleared
        """
        count = await self.storage.clear_all()
        logger.info(f"Cache cleared: {count} entries")
        return count

    async def get_or_set(
        self,
        key: str,
        factory: Any,  # Callable[[], Awaitable[Any]]
        category: str = ANALYSIS,
    ) -> Any:
        """Get cached data or compute and cache it.

        Args:
            key: Cache key
            factory: Async function to compute data if not cached
            category: Cache category

        Returns:
            Cached or computed data
        """
        # Try to get from cache
        cached = await self.get(key, category)
        if cached is not None:
            return cached

        # Compute and cache
        data = await factory()
        await self.set(key, data, category)
        return data

    async def list_keys(self, category: str = ANALYSIS) -> list[str]:
        """List all cache keys in a category.

        Args:
            category: Cache category

        Returns:
            List of keys
        """
        return await self.storage.list_keys(category)

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        categories = await self.storage.list_categories()
        stats: dict[str, Any] = {
            "categories": {},
            "total_entries": 0,
            "total_size_bytes": 0,
        }

        for category in categories:
            keys = await self.storage.list_keys(category)
            category_size = 0

            for key in keys:
                metadata = await self.storage.get_metadata(category, key)
                if metadata:
                    category_size += metadata.size_bytes

            stats["categories"][category] = {
                "entries": len(keys),
                "size_bytes": category_size,
            }
            stats["total_entries"] += len(keys)
            stats["total_size_bytes"] += category_size

        return stats
