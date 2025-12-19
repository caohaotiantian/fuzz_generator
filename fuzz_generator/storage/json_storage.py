"""JSON file-based storage implementation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from fuzz_generator.storage.base import StorageBackend, StorageError, StorageMetadata
from fuzz_generator.utils.logger import get_logger

logger = get_logger(__name__)


class JsonStorage(StorageBackend):
    """JSON file-based storage backend.

    Stores data in JSON files organized by category directories.
    Structure:
        base_dir/
        ├── category1/
        │   ├── key1.json
        │   └── key2.json
        └── category2/
            └── key3.json
    """

    def __init__(self, base_dir: str | Path):
        """Initialize JSON storage.

        Args:
            base_dir: Base directory for storage
        """
        self.base_dir = Path(base_dir)
        self._ensure_base_dir()

    def _ensure_base_dir(self) -> None:
        """Ensure base directory exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_category_path(self, category: str) -> Path:
        """Get path for a category directory."""
        # Sanitize category name
        safe_category = category.replace("/", "_").replace("\\", "_")
        return self.base_dir / safe_category

    def _get_file_path(self, category: str, key: str) -> Path:
        """Get path for a data file."""
        # Sanitize key name
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._get_category_path(category) / f"{safe_key}.json"

    async def save(self, category: str, key: str, data: Any) -> None:
        """Save data to a JSON file."""
        file_path = self._get_file_path(category, key)

        try:
            # Ensure category directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data with metadata
            stored_data = {
                "_metadata": {
                    "key": key,
                    "category": category,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                },
                "data": data,
            }

            # Check if file exists to preserve created_at
            if file_path.exists():
                try:
                    async with aiofiles.open(file_path, encoding="utf-8") as f:
                        existing = json.loads(await f.read())
                        if "_metadata" in existing:
                            stored_data["_metadata"]["created_at"] = existing["_metadata"].get(
                                "created_at", datetime.now().isoformat()
                            )
                except Exception:
                    pass  # Use new created_at if read fails

            # Write data
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(stored_data, indent=2, ensure_ascii=False, default=str))

            logger.debug(f"Saved data to {file_path}")

        except Exception as e:
            raise StorageError(f"Failed to save {category}/{key}: {e}") from e

    async def load(self, category: str, key: str) -> Any | None:
        """Load data from a JSON file."""
        file_path = self._get_file_path(category, key)

        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, encoding="utf-8") as f:
                content = await f.read()
                stored_data = json.loads(content)
                return stored_data.get("data")

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load {category}/{key}: {e}")
            return None

    async def exists(self, category: str, key: str) -> bool:
        """Check if a data file exists."""
        file_path = self._get_file_path(category, key)
        return file_path.exists()

    async def delete(self, category: str, key: str) -> bool:
        """Delete a data file."""
        file_path = self._get_file_path(category, key)

        if not file_path.exists():
            return False

        try:
            await aiofiles.os.remove(file_path)
            logger.debug(f"Deleted {file_path}")

            # Try to remove empty category directory
            category_path = self._get_category_path(category)
            if category_path.exists() and not any(category_path.iterdir()):
                category_path.rmdir()

            return True

        except Exception as e:
            logger.error(f"Failed to delete {category}/{key}: {e}")
            return False

    async def list_keys(self, category: str) -> list[str]:
        """List all keys in a category."""
        category_path = self._get_category_path(category)

        if not category_path.exists():
            return []

        keys = []
        for file_path in category_path.iterdir():
            if file_path.is_file() and file_path.suffix == ".json":
                keys.append(file_path.stem)

        return sorted(keys)

    async def list_categories(self) -> list[str]:
        """List all categories."""
        if not self.base_dir.exists():
            return []

        categories = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                categories.append(path.name)

        return sorted(categories)

    async def get_metadata(self, category: str, key: str) -> StorageMetadata | None:
        """Get metadata for a stored item."""
        file_path = self._get_file_path(category, key)

        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, encoding="utf-8") as f:
                content = await f.read()
                stored_data = json.loads(content)

            metadata_dict = stored_data.get("_metadata", {})
            stat = file_path.stat()

            return StorageMetadata(
                key=key,
                category=category,
                created_at=datetime.fromisoformat(
                    metadata_dict.get("created_at", datetime.now().isoformat())
                ),
                updated_at=datetime.fromisoformat(
                    metadata_dict.get("updated_at", datetime.now().isoformat())
                ),
                size_bytes=stat.st_size,
                expires_at=(
                    datetime.fromisoformat(metadata_dict["expires_at"])
                    if metadata_dict.get("expires_at")
                    else None
                ),
            )

        except Exception as e:
            logger.error(f"Failed to get metadata for {category}/{key}: {e}")
            return None

    async def clear_category(self, category: str) -> int:
        """Clear all data in a category."""
        category_path = self._get_category_path(category)

        if not category_path.exists():
            return 0

        count = 0
        for file_path in list(category_path.iterdir()):
            if file_path.is_file() and file_path.suffix == ".json":
                try:
                    await aiofiles.os.remove(file_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")

        # Remove empty directory
        if category_path.exists() and not any(category_path.iterdir()):
            category_path.rmdir()

        logger.info(f"Cleared {count} items from category '{category}'")
        return count

    async def clear_all(self) -> int:
        """Clear all stored data."""
        categories = await self.list_categories()
        total_count = 0

        for category in categories:
            count = await self.clear_category(category)
            total_count += count

        logger.info(f"Cleared total of {total_count} items")
        return total_count
