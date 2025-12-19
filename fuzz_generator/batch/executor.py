"""Batch task executor for parallel and sequential execution.

Supports concurrent execution, progress reporting, and partial failure handling.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from fuzz_generator.agents.orchestrator import OrchestratorAgent
from fuzz_generator.config import Settings
from fuzz_generator.exceptions import TaskError
from fuzz_generator.models import AnalysisTask, BatchTask, TaskResult
from fuzz_generator.storage import StorageBackend
from fuzz_generator.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionProgress:
    """Progress information for batch execution."""

    completed: int = 0
    failed: int = 0
    total: int = 0
    current_task: AnalysisTask | None = None
    elapsed_seconds: float = 0.0

    @property
    def pending(self) -> int:
        """Number of pending tasks."""
        return self.total - self.completed - self.failed

    @property
    def progress_percent(self) -> float:
        """Completion percentage."""
        if self.total == 0:
            return 100.0
        return (self.completed + self.failed) / self.total * 100


@dataclass
class BatchResult:
    """Result of batch execution."""

    batch_id: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    results: list[TaskResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def success(self) -> bool:
        """Whether all tasks completed successfully."""
        return self.failed_tasks == 0 and self.cancelled_tasks == 0

    @property
    def duration_seconds(self) -> float | None:
        """Total execution duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()


# Type alias for progress callback
ProgressCallback = Callable[[int, int, AnalysisTask | None], None]


class BatchExecutor:
    """Executor for batch analysis tasks."""

    def __init__(
        self,
        orchestrator: OrchestratorAgent | None = None,
        settings: Settings | None = None,
        storage: StorageBackend | None = None,
    ):
        """Initialize executor.

        Args:
            orchestrator: OrchestratorAgent instance for task execution.
            settings: Application settings.
            storage: Storage backend for state management.
        """
        self.orchestrator = orchestrator
        self.settings = settings
        self.storage = storage
        self._cancelled = False

    async def execute(
        self,
        batch: BatchTask,
        *,
        fail_fast: bool | None = None,
        max_concurrent: int | None = None,
        on_progress: ProgressCallback | None = None,
        on_task_complete: Callable[[TaskResult], None] | None = None,
    ) -> BatchResult:
        """Execute all tasks in a batch.

        Args:
            batch: BatchTask containing tasks to execute.
            fail_fast: Stop on first failure (overrides settings).
            max_concurrent: Max concurrent tasks (overrides settings).
            on_progress: Callback for progress updates.
            on_task_complete: Callback when a task completes.

        Returns:
            BatchResult with execution summary.
        """
        self._cancelled = False

        # Get settings
        _fail_fast = fail_fast
        _max_concurrent = max_concurrent

        if self.settings:
            if _fail_fast is None:
                _fail_fast = self.settings.batch.fail_fast
            if _max_concurrent is None:
                _max_concurrent = self.settings.batch.max_concurrent

        _fail_fast = _fail_fast if _fail_fast is not None else False
        _max_concurrent = _max_concurrent if _max_concurrent is not None else 1

        logger.info(
            f"Starting batch execution: {batch.batch_id} "
            f"({len(batch.tasks)} tasks, max_concurrent={_max_concurrent})"
        )

        result = BatchResult(
            batch_id=batch.batch_id,
            total_tasks=len(batch.tasks),
            start_time=datetime.now(),
        )

        try:
            if _max_concurrent <= 1:
                # Sequential execution
                await self._execute_sequential(
                    batch, result, _fail_fast, on_progress, on_task_complete
                )
            else:
                # Concurrent execution
                await self._execute_concurrent(
                    batch, result, _fail_fast, _max_concurrent, on_progress, on_task_complete
                )
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            result.errors.append(str(e))
        finally:
            result.end_time = datetime.now()

        logger.info(
            f"Batch execution completed: {result.completed_tasks} succeeded, "
            f"{result.failed_tasks} failed, {result.cancelled_tasks} cancelled"
        )

        return result

    async def _execute_sequential(
        self,
        batch: BatchTask,
        result: BatchResult,
        fail_fast: bool,
        on_progress: ProgressCallback | None,
        on_task_complete: Callable[[TaskResult], None] | None,
    ) -> None:
        """Execute tasks sequentially."""
        for i, task in enumerate(batch.tasks):
            if self._cancelled:
                self._cancel_remaining(batch.tasks[i:], result)
                break

            # Check dependencies
            if not self._dependencies_satisfied(task, batch, result):
                logger.warning(f"Skipping task {task.task_id} due to failed dependencies")
                task_result = TaskResult(
                    task_id=task.task_id,
                    success=False,
                    errors=["Skipped due to failed dependencies"],
                )
                result.results.append(task_result)
                result.failed_tasks += 1
                continue

            # Report progress
            if on_progress:
                on_progress(result.completed_tasks + result.failed_tasks, result.total_tasks, task)

            # Execute task
            task_result = await self._execute_task(batch, task)
            result.results.append(task_result)

            if task_result.success:
                result.completed_tasks += 1
            else:
                result.failed_tasks += 1
                if fail_fast:
                    self._cancel_remaining(batch.tasks[i + 1 :], result)
                    break

            if on_task_complete:
                on_task_complete(task_result)

        # Final progress report
        if on_progress:
            on_progress(
                result.completed_tasks + result.failed_tasks,
                result.total_tasks,
                None,
            )

    async def _execute_concurrent(
        self,
        batch: BatchTask,
        result: BatchResult,
        fail_fast: bool,
        max_concurrent: int,
        on_progress: ProgressCallback | None,
        on_task_complete: Callable[[TaskResult], None] | None,
    ) -> None:
        """Execute tasks concurrently with semaphore."""
        semaphore = asyncio.Semaphore(max_concurrent)
        completed_ids: set[str] = set()
        failed_ids: set[str] = set()
        lock = asyncio.Lock()

        async def execute_with_semaphore(task: AnalysisTask) -> TaskResult:
            async with semaphore:
                if self._cancelled:
                    return TaskResult(
                        task_id=task.task_id,
                        success=False,
                        errors=["Cancelled"],
                    )

                # Check dependencies
                async with lock:
                    deps_ok = all(dep in completed_ids for dep in task.depends_on)
                    deps_failed = any(dep in failed_ids for dep in task.depends_on)

                if deps_failed:
                    return TaskResult(
                        task_id=task.task_id,
                        success=False,
                        errors=["Skipped due to failed dependencies"],
                    )

                # Wait for dependencies if not satisfied
                while not deps_ok and not self._cancelled:
                    await asyncio.sleep(0.1)
                    async with lock:
                        deps_ok = all(dep in completed_ids for dep in task.depends_on)
                        deps_failed = any(dep in failed_ids for dep in task.depends_on)
                        if deps_failed:
                            return TaskResult(
                                task_id=task.task_id,
                                success=False,
                                errors=["Skipped due to failed dependencies"],
                            )

                if on_progress:
                    async with lock:
                        on_progress(
                            len(completed_ids) + len(failed_ids),
                            result.total_tasks,
                            task,
                        )

                return await self._execute_task(batch, task)

        # Create all task coroutines
        coroutines = [execute_with_semaphore(task) for task in batch.tasks]

        # Execute and collect results
        for coro in asyncio.as_completed(coroutines):
            task_result = await coro
            result.results.append(task_result)

            async with lock:
                if task_result.success:
                    result.completed_tasks += 1
                    completed_ids.add(task_result.task_id)
                else:
                    result.failed_tasks += 1
                    failed_ids.add(task_result.task_id)

            if on_task_complete:
                on_task_complete(task_result)

            if fail_fast and not task_result.success:
                self._cancelled = True

        # Final progress report
        if on_progress:
            on_progress(
                result.completed_tasks + result.failed_tasks,
                result.total_tasks,
                None,
            )

    async def _execute_task(
        self,
        batch: BatchTask,
        task: AnalysisTask,
    ) -> TaskResult:
        """Execute a single task."""
        logger.info(f"Executing task: {task.task_id} ({task.function_name})")

        task.start()

        try:
            if self.orchestrator is None:
                raise TaskError("No orchestrator configured")

            # Run the orchestrator
            agent_result = await self.orchestrator.run(
                project_path=batch.project_path,
                source_file=task.source_file,
                function_name=task.function_name,
                output_name=task.output_name,
                task_id=task.task_id,
            )

            if agent_result.success:
                task.complete()
                # Extract TaskResult from agent result
                if isinstance(agent_result.data, TaskResult):
                    return agent_result.data
                else:
                    return TaskResult(
                        task_id=task.task_id,
                        success=True,
                        xml_content=agent_result.data.get("xml_content")
                        if isinstance(agent_result.data, dict)
                        else None,
                        duration_seconds=task.duration_seconds or 0.0,
                    )
            else:
                task.fail(agent_result.error or "Unknown error")
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    errors=[agent_result.error or "Unknown error"],
                    duration_seconds=task.duration_seconds or 0.0,
                )

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.fail(str(e))
            return TaskResult(
                task_id=task.task_id,
                success=False,
                errors=[str(e)],
                duration_seconds=task.duration_seconds or 0.0,
            )

    def _dependencies_satisfied(
        self,
        task: AnalysisTask,
        batch: BatchTask,
        result: BatchResult,
    ) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.depends_on:
            return True

        completed_ids = {r.task_id for r in result.results if r.success}

        return all(dep in completed_ids for dep in task.depends_on)

    def _cancel_remaining(
        self,
        tasks: list[AnalysisTask],
        result: BatchResult,
    ) -> None:
        """Mark remaining tasks as cancelled."""
        for task in tasks:
            task.cancel()
            result.results.append(
                TaskResult(
                    task_id=task.task_id,
                    success=False,
                    errors=["Cancelled"],
                )
            )
            result.cancelled_tasks += 1

    def cancel(self) -> None:
        """Cancel batch execution."""
        self._cancelled = True
        logger.info("Batch execution cancellation requested")

    @property
    def is_cancelled(self) -> bool:
        """Check if execution is cancelled."""
        return self._cancelled
