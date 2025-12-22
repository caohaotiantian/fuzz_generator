"""Tests for batch task executor."""

from unittest.mock import AsyncMock

import pytest

from fuzz_generator.batch.executor import BatchExecutor, BatchResult, ExecutionProgress
from fuzz_generator.models import AnalysisTask, BatchTask, TaskResult


class TestExecutionProgress:
    """Tests for ExecutionProgress dataclass."""

    def test_default_values(self):
        """Test default values."""
        progress = ExecutionProgress()
        assert progress.completed == 0
        assert progress.failed == 0
        assert progress.total == 0
        assert progress.current_task is None

    def test_pending_calculation(self):
        """Test pending calculation."""
        progress = ExecutionProgress(completed=2, failed=1, total=10)
        assert progress.pending == 7

    def test_progress_percent(self):
        """Test progress percentage calculation."""
        progress = ExecutionProgress(completed=3, failed=2, total=10)
        assert progress.progress_percent == 50.0

    def test_progress_percent_zero_total(self):
        """Test progress percentage with zero total."""
        progress = ExecutionProgress(total=0)
        assert progress.progress_percent == 100.0


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = BatchResult(batch_id="test")
        assert result.batch_id == "test"
        assert result.total_tasks == 0
        assert result.completed_tasks == 0
        assert result.failed_tasks == 0
        assert result.results == []

    def test_success_all_completed(self):
        """Test success when all tasks completed."""
        result = BatchResult(
            batch_id="test",
            total_tasks=3,
            completed_tasks=3,
            failed_tasks=0,
        )
        assert result.success is True

    def test_success_with_failures(self):
        """Test success is False with failures."""
        result = BatchResult(
            batch_id="test",
            total_tasks=3,
            completed_tasks=2,
            failed_tasks=1,
        )
        assert result.success is False

    def test_success_with_cancelled(self):
        """Test success is False with cancelled tasks."""
        result = BatchResult(
            batch_id="test",
            total_tasks=3,
            completed_tasks=2,
            cancelled_tasks=1,
        )
        assert result.success is False


class TestBatchExecutor:
    """Tests for BatchExecutor class."""

    @pytest.fixture
    def mock_task_executor(self):
        """Create mock task executor."""

        async def executor(task: AnalysisTask) -> TaskResult:
            return TaskResult(
                task_id=task.task_id,
                success=True,
                xml_content="<DataModel>...</DataModel>",
            )

        return AsyncMock(side_effect=executor)

    @pytest.fixture
    def sample_batch(self) -> BatchTask:
        """Create sample batch for testing."""
        return BatchTask(
            batch_id="test_batch",
            project_path="/path/to/project",
            tasks=[
                AnalysisTask(
                    task_id="task_1",
                    source_file="a.c",
                    function_name="func_a",
                ),
                AnalysisTask(
                    task_id="task_2",
                    source_file="b.c",
                    function_name="func_b",
                ),
            ],
        )

    def test_executor_creation(self):
        """Test executor can be created."""
        executor = BatchExecutor()
        assert executor.task_executor is None
        assert executor.settings is None

    def test_executor_with_task_executor(self, mock_task_executor):
        """Test executor with task executor."""
        executor = BatchExecutor(task_executor=mock_task_executor)
        assert executor.task_executor == mock_task_executor

    @pytest.mark.asyncio
    async def test_execute_batch(self, mock_task_executor, sample_batch):
        """Test executing a batch."""
        executor = BatchExecutor(task_executor=mock_task_executor)
        result = await executor.execute(sample_batch)

        assert result.batch_id == "test_batch"
        assert result.total_tasks == 2
        assert result.completed_tasks == 2
        assert result.failed_tasks == 0
        assert len(result.results) == 2
        assert mock_task_executor.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_sequential(self, mock_task_executor, sample_batch):
        """Test sequential execution."""
        executor = BatchExecutor(task_executor=mock_task_executor)
        result = await executor.execute(sample_batch, max_concurrent=1)

        assert result.completed_tasks == 2
        assert mock_task_executor.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_concurrent(self, mock_task_executor, sample_batch):
        """Test concurrent execution."""
        executor = BatchExecutor(task_executor=mock_task_executor)
        result = await executor.execute(sample_batch, max_concurrent=2)

        assert result.completed_tasks == 2
        assert mock_task_executor.call_count == 2

    @pytest.mark.asyncio
    async def test_partial_failure(self, sample_batch):
        """Test partial task failure."""
        call_count = 0

        async def executor(task: AnalysisTask) -> TaskResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return TaskResult(task_id=task.task_id, success=True)
            else:
                return TaskResult(task_id=task.task_id, success=False, errors=["Task failed"])

        mock_executor = AsyncMock(side_effect=executor)
        batch_executor = BatchExecutor(task_executor=mock_executor)
        result = await batch_executor.execute(sample_batch, fail_fast=False)

        assert result.completed_tasks == 1
        assert result.failed_tasks == 1
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_fail_fast(self, sample_batch):
        """Test fail-fast behavior."""

        async def failing_executor(task: AnalysisTask) -> TaskResult:
            return TaskResult(task_id=task.task_id, success=False, errors=["Task failed"])

        mock_executor = AsyncMock(side_effect=failing_executor)

        # Add a third task
        sample_batch.tasks.append(
            AnalysisTask(
                task_id="task_3",
                source_file="c.c",
                function_name="func_c",
            )
        )

        executor = BatchExecutor(task_executor=mock_executor)
        result = await executor.execute(sample_batch, fail_fast=True)

        assert result.failed_tasks == 1
        assert result.cancelled_tasks == 2
        assert mock_executor.call_count == 1  # Only first task was executed

    @pytest.mark.asyncio
    async def test_progress_callback(self, mock_task_executor, sample_batch):
        """Test progress callback."""
        progress_calls = []

        def on_progress(completed, total, task):
            progress_calls.append((completed, total, task))

        executor = BatchExecutor(task_executor=mock_task_executor)
        await executor.execute(sample_batch, on_progress=on_progress)

        # Should have calls for each task + final call
        assert len(progress_calls) >= 2
        # Last call should have completed == total
        assert progress_calls[-1][0] == progress_calls[-1][1]

    @pytest.mark.asyncio
    async def test_task_complete_callback(self, mock_task_executor, sample_batch):
        """Test task completion callback."""
        completed_tasks = []

        def on_task_complete(result):
            completed_tasks.append(result)

        executor = BatchExecutor(task_executor=mock_task_executor)
        await executor.execute(sample_batch, on_task_complete=on_task_complete)

        assert len(completed_tasks) == 2
        assert all(isinstance(t, TaskResult) for t in completed_tasks)

    @pytest.mark.asyncio
    async def test_task_dependencies(self, mock_task_executor):
        """Test task dependencies are respected."""
        batch = BatchTask(
            batch_id="test_batch",
            project_path="/path",
            tasks=[
                AnalysisTask(
                    task_id="task_1",
                    source_file="a.c",
                    function_name="func_a",
                ),
                AnalysisTask(
                    task_id="task_2",
                    source_file="b.c",
                    function_name="func_b",
                    depends_on=["task_1"],
                ),
            ],
        )

        executor = BatchExecutor(task_executor=mock_task_executor)
        result = await executor.execute(batch)

        assert result.completed_tasks == 2
        # task_2 should only run after task_1

    @pytest.mark.asyncio
    async def test_dependency_failure_skips_dependent(self, sample_batch):
        """Test that failed dependency skips dependent task."""
        call_count = 0

        async def executor(task: AnalysisTask) -> TaskResult:
            nonlocal call_count
            call_count += 1
            if task.task_id == "task_1":
                return TaskResult(task_id=task.task_id, success=False, errors=["Failed"])
            return TaskResult(task_id=task.task_id, success=True)

        # Make task_2 depend on task_1
        sample_batch.tasks[1].depends_on = ["task_1"]

        mock_executor = AsyncMock(side_effect=executor)
        batch_executor = BatchExecutor(task_executor=mock_executor)
        result = await batch_executor.execute(sample_batch)

        assert result.failed_tasks == 2  # Both failed (task_1 failed, task_2 skipped)
        assert mock_executor.call_count == 1  # Only task_1 was executed

    @pytest.mark.asyncio
    async def test_no_task_executor(self, sample_batch):
        """Test error when no task executor configured."""
        executor = BatchExecutor()
        result = await executor.execute(sample_batch)

        assert result.failed_tasks == 2
        assert all("No task executor configured" in str(r.errors) for r in result.results)

    def test_cancel(self, mock_task_executor):
        """Test cancellation."""
        executor = BatchExecutor(task_executor=mock_task_executor)
        assert not executor.is_cancelled

        executor.cancel()
        assert executor.is_cancelled

    @pytest.mark.asyncio
    async def test_duration_tracking(self, mock_task_executor, sample_batch):
        """Test duration tracking."""
        executor = BatchExecutor(task_executor=mock_task_executor)
        result = await executor.execute(sample_batch)

        assert result.start_time is not None
        assert result.end_time is not None
        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0


class TestBatchExecutorEdgeCases:
    """Tests for edge cases in BatchExecutor."""

    @pytest.fixture
    def sample_batch(self) -> BatchTask:
        """Create sample batch for testing."""
        return BatchTask(
            batch_id="test_batch",
            project_path="/path/to/project",
            tasks=[
                AnalysisTask(
                    task_id="task_1",
                    source_file="a.c",
                    function_name="func_a",
                ),
                AnalysisTask(
                    task_id="task_2",
                    source_file="b.c",
                    function_name="func_b",
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Test executing empty batch."""
        batch = BatchTask(
            batch_id="empty",
            project_path="/path",
            tasks=[],
        )

        executor = BatchExecutor()
        result = await executor.execute(batch)

        assert result.total_tasks == 0
        assert result.completed_tasks == 0
        assert result.success is True

    @pytest.mark.asyncio
    async def test_exception_in_executor(self, sample_batch):
        """Test exception handling in task executor."""

        async def failing_executor(task: AnalysisTask) -> TaskResult:
            raise RuntimeError("Unexpected error")

        mock_executor = AsyncMock(side_effect=failing_executor)
        executor = BatchExecutor(task_executor=mock_executor)
        result = await executor.execute(sample_batch)

        assert result.failed_tasks == 2
        assert all("Unexpected error" in str(r.errors) for r in result.results)

    @pytest.mark.asyncio
    async def test_single_task_batch(self):
        """Test batch with single task."""
        batch = BatchTask(
            batch_id="single",
            project_path="/path",
            tasks=[
                AnalysisTask(
                    task_id="only_task",
                    source_file="a.c",
                    function_name="func",
                ),
            ],
        )

        async def executor(task: AnalysisTask) -> TaskResult:
            return TaskResult(task_id=task.task_id, success=True)

        mock_executor = AsyncMock(side_effect=executor)
        batch_executor = BatchExecutor(task_executor=mock_executor)
        result = await batch_executor.execute(batch)

        assert result.total_tasks == 1
        assert result.completed_tasks == 1
        assert result.success is True
