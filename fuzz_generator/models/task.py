"""Task models for analysis management."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisTask(BaseModel):
    """Single analysis task definition."""

    task_id: str = Field(description="Unique task identifier")
    source_file: str = Field(description="Source file path (relative to project)")
    function_name: str = Field(description="Function name to analyze")
    output_name: str | None = Field(
        default=None,
        description="Custom name for generated DataModel",
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current task status",
    )
    priority: int = Field(
        default=0,
        description="Task priority (higher = more urgent)",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of task IDs this task depends on",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Task creation timestamp",
    )
    started_at: datetime | None = Field(
        default=None,
        description="Task start timestamp",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Task completion timestamp",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if task failed",
    )
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts",
    )

    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()

    def fail(self, error_message: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message

    def cancel(self) -> None:
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()

    @property
    def duration_seconds(self) -> float | None:
        """Get task duration in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()


class BatchTask(BaseModel):
    """Batch of analysis tasks."""

    batch_id: str = Field(description="Unique batch identifier")
    project_path: str = Field(description="Project source path")
    description: str | None = Field(
        default=None,
        description="Batch description",
    )
    tasks: list[AnalysisTask] = Field(
        default_factory=list,
        description="List of tasks in this batch",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Batch creation timestamp",
    )
    config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration overrides for this batch",
    )

    @property
    def total_count(self) -> int:
        """Get total task count."""
        return len(self.tasks)

    @property
    def pending_count(self) -> int:
        """Get pending task count."""
        return sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)

    @property
    def running_count(self) -> int:
        """Get running task count."""
        return sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING)

    @property
    def completed_count(self) -> int:
        """Get completed task count."""
        return sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        """Get failed task count."""
        return sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)

    @property
    def cancelled_count(self) -> int:
        """Get cancelled task count."""
        return sum(1 for t in self.tasks if t.status == TaskStatus.CANCELLED)

    @property
    def is_complete(self) -> bool:
        """Check if all tasks are finished."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            for t in self.tasks
        )

    def get_pending_tasks(self) -> list[AnalysisTask]:
        """Get all pending tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_next_task(self) -> AnalysisTask | None:
        """Get next task to execute based on priority and dependencies."""
        pending = self.get_pending_tasks()
        if not pending:
            return None

        # Check dependencies
        completed_ids = {t.task_id for t in self.tasks if t.status == TaskStatus.COMPLETED}

        for task in sorted(pending, key=lambda t: -t.priority):
            if all(dep in completed_ids for dep in task.depends_on):
                return task

        return None


class TaskResult(BaseModel):
    """Result of a completed task."""

    task_id: str = Field(description="Task identifier")
    success: bool = Field(description="Whether task succeeded")
    xml_content: str | None = Field(
        default=None,
        description="Generated XML content",
    )
    data_models: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Generated DataModel structures",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warning messages",
    )
    duration_seconds: float = Field(
        default=0.0,
        description="Task duration in seconds",
    )
    output_path: str | None = Field(
        default=None,
        description="Path to output file",
    )


class IntermediateResult(BaseModel):
    """Intermediate result from a processing stage."""

    task_id: str = Field(description="Task identifier")
    stage: str = Field(description="Processing stage name")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Stage output data",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Result timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
