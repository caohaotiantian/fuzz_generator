"""Base storage interface definitions."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class StorageError(Exception):
    """Storage related errors."""

    pass


class StorageMetadata(BaseModel):
    """Metadata for stored items."""

    key: str
    category: str
    created_at: datetime
    updated_at: datetime
    size_bytes: int = 0
    expires_at: datetime | None = None


class StorageBackend(ABC):
    """Abstract storage backend interface."""

    @abstractmethod
    async def save(self, category: str, key: str, data: Any) -> None:
        """Save data to storage.

        Args:
            category: Category/namespace for the data
            key: Unique key within the category
            data: Data to store (must be JSON serializable)

        Raises:
            StorageError: If save fails
        """
        pass

    @abstractmethod
    async def load(self, category: str, key: str) -> Any | None:
        """Load data from storage.

        Args:
            category: Category/namespace for the data
            key: Key to load

        Returns:
            Stored data or None if not found
        """
        pass

    @abstractmethod
    async def exists(self, category: str, key: str) -> bool:
        """Check if data exists in storage.

        Args:
            category: Category/namespace
            key: Key to check

        Returns:
            True if exists, False otherwise
        """
        pass

    @abstractmethod
    async def delete(self, category: str, key: str) -> bool:
        """Delete data from storage.

        Args:
            category: Category/namespace
            key: Key to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def list_keys(self, category: str) -> list[str]:
        """List all keys in a category.

        Args:
            category: Category/namespace

        Returns:
            List of keys
        """
        pass

    @abstractmethod
    async def list_categories(self) -> list[str]:
        """List all categories.

        Returns:
            List of category names
        """
        pass

    @abstractmethod
    async def get_metadata(self, category: str, key: str) -> StorageMetadata | None:
        """Get metadata for a stored item.

        Args:
            category: Category/namespace
            key: Key to get metadata for

        Returns:
            Metadata or None if not found
        """
        pass

    @abstractmethod
    async def clear_category(self, category: str) -> int:
        """Clear all data in a category.

        Args:
            category: Category to clear

        Returns:
            Number of items deleted
        """
        pass

    @abstractmethod
    async def clear_all(self) -> int:
        """Clear all stored data.

        Returns:
            Number of items deleted
        """
        pass
