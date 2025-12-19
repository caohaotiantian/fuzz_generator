"""Test storage module."""

from pathlib import Path

import pytest

from fuzz_generator.storage import CacheManager, JsonStorage


class TestJsonStorage:
    """Test JSON storage backend."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> JsonStorage:
        """Create test storage."""
        return JsonStorage(base_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_save_and_load(self, storage: JsonStorage):
        """Test saving and loading data."""
        data = {"key": "value", "number": 42, "nested": {"a": 1}}

        await storage.save("test_category", "test_key", data)
        loaded = await storage.load("test_category", "test_key")

        assert loaded == data

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, storage: JsonStorage):
        """Test loading non-existent key."""
        result = await storage.load("category", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_exists(self, storage: JsonStorage):
        """Test exists check."""
        assert not await storage.exists("category", "key")

        await storage.save("category", "key", {"data": 1})

        assert await storage.exists("category", "key")

    @pytest.mark.asyncio
    async def test_delete(self, storage: JsonStorage):
        """Test deleting data."""
        await storage.save("category", "key", {"data": 1})
        assert await storage.exists("category", "key")

        result = await storage.delete("category", "key")
        assert result is True
        assert not await storage.exists("category", "key")

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, storage: JsonStorage):
        """Test deleting non-existent key."""
        result = await storage.delete("category", "nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_keys(self, storage: JsonStorage):
        """Test listing keys."""
        await storage.save("category", "key1", {"data": 1})
        await storage.save("category", "key2", {"data": 2})
        await storage.save("category", "key3", {"data": 3})

        keys = await storage.list_keys("category")
        assert set(keys) == {"key1", "key2", "key3"}

    @pytest.mark.asyncio
    async def test_list_keys_empty(self, storage: JsonStorage):
        """Test listing keys in empty category."""
        keys = await storage.list_keys("empty_category")
        assert keys == []

    @pytest.mark.asyncio
    async def test_list_categories(self, storage: JsonStorage):
        """Test listing categories."""
        await storage.save("cat1", "key1", {"data": 1})
        await storage.save("cat2", "key2", {"data": 2})

        categories = await storage.list_categories()
        assert set(categories) == {"cat1", "cat2"}

    @pytest.mark.asyncio
    async def test_get_metadata(self, storage: JsonStorage):
        """Test getting metadata."""
        await storage.save("category", "key", {"data": "test"})

        metadata = await storage.get_metadata("category", "key")

        assert metadata is not None
        assert metadata.key == "key"
        assert metadata.category == "category"
        assert metadata.size_bytes > 0

    @pytest.mark.asyncio
    async def test_clear_category(self, storage: JsonStorage):
        """Test clearing a category."""
        await storage.save("category", "key1", {"data": 1})
        await storage.save("category", "key2", {"data": 2})

        count = await storage.clear_category("category")

        assert count == 2
        assert await storage.list_keys("category") == []

    @pytest.mark.asyncio
    async def test_clear_all(self, storage: JsonStorage):
        """Test clearing all data."""
        await storage.save("cat1", "key1", {"data": 1})
        await storage.save("cat2", "key2", {"data": 2})

        count = await storage.clear_all()

        assert count == 2
        assert await storage.list_categories() == []

    @pytest.mark.asyncio
    async def test_update_preserves_created_at(self, storage: JsonStorage):
        """Test that update preserves created_at."""
        await storage.save("category", "key", {"version": 1})
        meta1 = await storage.get_metadata("category", "key")

        await storage.save("category", "key", {"version": 2})
        meta2 = await storage.get_metadata("category", "key")

        assert meta1.created_at == meta2.created_at
        assert meta2.updated_at >= meta1.updated_at


class TestCacheManager:
    """Test cache manager."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> CacheManager:
        """Create test cache manager."""
        return CacheManager(base_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_cache_hit(self, cache: CacheManager):
        """Test cache hit."""
        await cache.set("key123", {"result": "data"})
        result = await cache.get("key123")
        assert result == {"result": "data"}

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache: CacheManager):
        """Test cache miss."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate(self, cache: CacheManager):
        """Test cache invalidation."""
        await cache.set("key", {"data": 1})
        assert await cache.get("key") is not None

        result = await cache.invalidate("key")
        assert result is True
        assert await cache.get("key") is None

    @pytest.mark.asyncio
    async def test_invalidate_category(self, cache: CacheManager):
        """Test invalidating a category."""
        await cache.set("key1", {"data": 1}, category=CacheManager.FUNCTIONS)
        await cache.set("key2", {"data": 2}, category=CacheManager.FUNCTIONS)

        count = await cache.invalidate_category(CacheManager.FUNCTIONS)
        assert count == 2

    @pytest.mark.asyncio
    async def test_clear(self, cache: CacheManager):
        """Test clearing all cache."""
        await cache.set("key1", {"data": 1})
        await cache.set("key2", {"data": 2})

        count = await cache.clear()
        assert count >= 2

    def test_compute_cache_key(self):
        """Test cache key computation."""
        key1 = CacheManager.compute_cache_key("/path", "func1", "code")
        key2 = CacheManager.compute_cache_key("/path", "func1", "code")
        key3 = CacheManager.compute_cache_key("/path", "func1", "different")

        assert key1 == key2  # Same inputs should produce same key
        assert key1 != key3  # Different content should produce different key
        assert len(key1) == 16  # Should be 16 chars

    @pytest.mark.asyncio
    async def test_get_or_set(self, cache: CacheManager):
        """Test get_or_set functionality."""
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return {"computed": True}

        # First call should invoke factory
        result1 = await cache.get_or_set("key", factory)
        assert result1 == {"computed": True}
        assert call_count == 1

        # Second call should use cache
        result2 = await cache.get_or_set("key", factory)
        assert result2 == {"computed": True}
        assert call_count == 1  # Factory not called again

    @pytest.mark.asyncio
    async def test_list_keys(self, cache: CacheManager):
        """Test listing cache keys."""
        await cache.set("key1", {"data": 1})
        await cache.set("key2", {"data": 2})

        keys = await cache.list_keys()
        assert "key1" in keys
        assert "key2" in keys

    @pytest.mark.asyncio
    async def test_get_stats(self, cache: CacheManager):
        """Test getting cache statistics."""
        await cache.set("key1", {"data": 1})
        await cache.set("key2", {"data": 2})

        stats = await cache.get_stats()

        assert "categories" in stats
        assert "total_entries" in stats
        assert "total_size_bytes" in stats
        assert stats["total_entries"] >= 2
