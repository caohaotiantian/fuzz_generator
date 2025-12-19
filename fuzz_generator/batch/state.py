"""Batch state management for resume support.

Provides persistent state tracking and resume capabilities for batch execution.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from fuzz_generator.models import BatchTask
from fuzz_generator.storage import StorageBackend
from fuzz_generator.utils import get_logger

logger = get_logger(__name__)


class BatchState(BaseModel):
    """Persistent state for a batch execution."""

    batch_id: str = Field(description="Batch identifier")
    project_path: str = Field(description="Project source path")
    completed: list[str] = Field(
        default_factory=list,
        description="Completed task IDs",
    )
    failed: list[str] = Field(
        default_factory=list,
        description="Failed task IDs",
    )
    pending: list[str] = Field(
        default_factory=list,
        description="Pending task IDs",
    )
    running: list[str] = Field(
        default_factory=list,
        description="Currently running task IDs",
    )
    task_results: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Task results by task ID",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="State creation time",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update time",
    )

    @property
    def total_count(self) -> int:
        """Total number of tasks."""
        return len(self.completed) + len(self.failed) + len(self.pending) + len(self.running)

    @property
    def is_complete(self) -> bool:
        """Check if all tasks are finished."""
        return len(self.pending) == 0 and len(self.running) == 0


class BatchStateManager:
    """Manager for batch execution state."""

    CATEGORY = "batch_state"

    def __init__(self, storage: StorageBackend):
        """Initialize state manager.

        Args:
            storage: Storage backend for persistence.
        """
        self.storage = storage

    async def create_state(self, batch: BatchTask) -> BatchState:
        """Create initial state for a batch.

        Args:
            batch: BatchTask to create state for.

        Returns:
            Initial BatchState.
        """
        state = BatchState(
            batch_id=batch.batch_id,
            project_path=batch.project_path,
            pending=[task.task_id for task in batch.tasks],
        )

        await self.save_state(batch.batch_id, state)
        logger.debug(f"Created state for batch {batch.batch_id}")

        return state

    async def save_state(self, batch_id: str, state: BatchState) -> None:
        """Save batch state to storage.

        Args:
            batch_id: Batch identifier.
            state: State to save.
        """
        state.updated_at = datetime.now()
        await self.storage.save(
            category=self.CATEGORY,
            key=batch_id,
            data=state.model_dump(mode="json"),
        )

    async def load_state(self, batch_id: str) -> BatchState | None:
        """Load batch state from storage.

        Args:
            batch_id: Batch identifier.

        Returns:
            BatchState if exists, None otherwise.
        """
        data = await self.storage.load(category=self.CATEGORY, key=batch_id)
        if data is None:
            return None

        return BatchState(**data)

    async def has_state(self, batch_id: str) -> bool:
        """Check if state exists for a batch.

        Args:
            batch_id: Batch identifier.

        Returns:
            True if state exists.
        """
        return await self.storage.exists(category=self.CATEGORY, key=batch_id)

    async def delete_state(self, batch_id: str) -> bool:
        """Delete batch state.

        Args:
            batch_id: Batch identifier.

        Returns:
            True if deleted.
        """
        return await self.storage.delete(category=self.CATEGORY, key=batch_id)

    async def get_pending_tasks(self, batch_id: str) -> list[str]:
        """Get list of pending task IDs.

        Args:
            batch_id: Batch identifier.

        Returns:
            List of pending task IDs.
        """
        state = await self.load_state(batch_id)
        if state is None:
            return []
        return state.pending

    async def get_completed_tasks(self, batch_id: str) -> list[str]:
        """Get list of completed task IDs.

        Args:
            batch_id: Batch identifier.

        Returns:
            List of completed task IDs.
        """
        state = await self.load_state(batch_id)
        if state is None:
            return []
        return state.completed

    async def get_failed_tasks(self, batch_id: str) -> list[str]:
        """Get list of failed task IDs.

        Args:
            batch_id: Batch identifier.

        Returns:
            List of failed task IDs.
        """
        state = await self.load_state(batch_id)
        if state is None:
            return []
        return state.failed

    async def mark_running(self, batch_id: str, task_id: str) -> None:
        """Mark a task as running.

        Args:
            batch_id: Batch identifier.
            task_id: Task identifier.
        """
        state = await self.load_state(batch_id)
        if state is None:
            logger.warning(f"No state found for batch {batch_id}")
            return

        if task_id in state.pending:
            state.pending.remove(task_id)
        if task_id not in state.running:
            state.running.append(task_id)

        await self.save_state(batch_id, state)

    async def mark_completed(
        self,
        batch_id: str,
        task_id: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Mark a task as completed.

        Args:
            batch_id: Batch identifier.
            task_id: Task identifier.
            result: Optional task result to store.
        """
        state = await self.load_state(batch_id)
        if state is None:
            logger.warning(f"No state found for batch {batch_id}")
            return

        # Remove from other lists
        if task_id in state.pending:
            state.pending.remove(task_id)
        if task_id in state.running:
            state.running.remove(task_id)
        if task_id in state.failed:
            state.failed.remove(task_id)

        # Add to completed
        if task_id not in state.completed:
            state.completed.append(task_id)

        # Store result
        if result is not None:
            state.task_results[task_id] = result

        await self.save_state(batch_id, state)

    async def mark_failed(
        self,
        batch_id: str,
        task_id: str,
        error: str | None = None,
    ) -> None:
        """Mark a task as failed.

        Args:
            batch_id: Batch identifier.
            task_id: Task identifier.
            error: Optional error message.
        """
        state = await self.load_state(batch_id)
        if state is None:
            logger.warning(f"No state found for batch {batch_id}")
            return

        # Remove from other lists
        if task_id in state.pending:
            state.pending.remove(task_id)
        if task_id in state.running:
            state.running.remove(task_id)

        # Add to failed
        if task_id not in state.failed:
            state.failed.append(task_id)

        # Store error
        if error is not None:
            state.task_results[task_id] = {"error": error}

        await self.save_state(batch_id, state)

    async def reset_running(self, batch_id: str) -> None:
        """Reset running tasks back to pending (for resume).

        Args:
            batch_id: Batch identifier.
        """
        state = await self.load_state(batch_id)
        if state is None:
            return

        # Move running tasks back to pending
        state.pending.extend(state.running)
        state.running.clear()

        await self.save_state(batch_id, state)

    async def get_resumable_batch(
        self,
        batch_id: str,
        original_batch: BatchTask,
    ) -> BatchTask:
        """Get a batch with only pending tasks for resume.

        Args:
            batch_id: Batch identifier.
            original_batch: Original batch with all tasks.

        Returns:
            BatchTask with only pending/failed tasks.
        """
        state = await self.load_state(batch_id)
        if state is None:
            return original_batch

        # Reset any running tasks
        await self.reset_running(batch_id)
        state = await self.load_state(batch_id)
        if state is None:
            return original_batch

        # Filter to only pending tasks
        pending_ids = set(state.pending)
        pending_tasks = [task for task in original_batch.tasks if task.task_id in pending_ids]

        return BatchTask(
            batch_id=original_batch.batch_id,
            project_path=original_batch.project_path,
            description=original_batch.description,
            tasks=pending_tasks,
            config_overrides=original_batch.config_overrides,
        )

    async def list_batches(self) -> list[str]:
        """List all batch IDs with saved state.

        Returns:
            List of batch IDs.
        """
        return await self.storage.list_keys(category=self.CATEGORY)

    async def get_batch_summary(self, batch_id: str) -> dict[str, Any] | None:
        """Get a summary of batch state.

        Args:
            batch_id: Batch identifier.

        Returns:
            Summary dictionary or None if not found.
        """
        state = await self.load_state(batch_id)
        if state is None:
            return None

        return {
            "batch_id": state.batch_id,
            "project_path": state.project_path,
            "total": state.total_count,
            "completed": len(state.completed),
            "failed": len(state.failed),
            "pending": len(state.pending),
            "running": len(state.running),
            "is_complete": state.is_complete,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
        }

    async def clear_all(self) -> None:
        """Clear all batch states."""
        await self.storage.clear_category(category=self.CATEGORY)
