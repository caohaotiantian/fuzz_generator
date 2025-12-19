"""Tests for batch task executor."""

from unittest.mock import AsyncMock, MagicMock

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
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = AsyncMock()
        orchestrator.run = AsyncMock(
            return_value=MagicMock(
                success=True,
                data=TaskResult(
                    task_id="test",
                    success=True,
                    xml_content="<DataModel>...</DataModel>",
                ),
                error=None,
            )
        )
        return orchestrator

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
        assert executor.orchestrator is None
        assert executor.settings is None

    def test_executor_with_orchestrator(self, mock_orchestrator):
        """Test executor with orchestrator."""
        executor = BatchExecutor(orchestrator=mock_orchestrator)
        assert executor.orchestrator == mock_orchestrator

    @pytest.mark.asyncio
    async def test_execute_batch(self, mock_orchestrator, sample_batch):
        """Test executing a batch."""
        executor = BatchExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(sample_batch)

        assert result.batch_id == "test_batch"
        assert result.total_tasks == 2
        assert result.completed_tasks == 2
        assert result.failed_tasks == 0
        assert len(result.results) == 2
        assert mock_orchestrator.run.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_sequential(self, mock_orchestrator, sample_batch):
        """Test sequential execution."""
        executor = BatchExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(sample_batch, max_concurrent=1)

        assert result.completed_tasks == 2
        assert mock_orchestrator.run.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_concurrent(self, mock_orchestrator, sample_batch):
        """Test concurrent execution."""
        executor = BatchExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(sample_batch, max_concurrent=2)

        assert result.completed_tasks == 2
        assert mock_orchestrator.run.call_count == 2

    @pytest.mark.asyncio
    async def test_partial_failure(self, mock_orchestrator, sample_batch):
        """Test partial task failure."""
        mock_orchestrator.run = AsyncMock(
            side_effect=[
                MagicMock(
                    success=True,
                    data=TaskResult(task_id="task_1", success=True),
                    error=None,
                ),
                MagicMock(
                    success=False,
                    data=None,
                    error="Task failed",
                ),
            ]
        )

        executor = BatchExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(sample_batch, fail_fast=False)

        assert result.completed_tasks == 1
        assert result.failed_tasks == 1
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_fail_fast(self, mock_orchestrator, sample_batch):
        """Test fail-fast behavior."""
        mock_orchestrator.run = AsyncMock(
            side_effect=[
                MagicMock(success=False, error="Task failed"),
                MagicMock(success=True, data=TaskResult(task_id="task_2", success=True)),
            ]
        )

        # Add a third task
        sample_batch.tasks.append(
            AnalysisTask(
                task_id="task_3",
                source_file="c.c",
                function_name="func_c",
            )
        )

        executor = BatchExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(sample_batch, fail_fast=True)

        assert result.failed_tasks == 1
        assert result.cancelled_tasks == 2
        assert mock_orchestrator.run.call_count == 1  # Only first task was executed

    @pytest.mark.asyncio
    async def test_progress_callback(self, mock_orchestrator, sample_batch):
        """Test progress callback."""
        progress_calls = []

        def on_progress(completed, total, task):
            progress_calls.append((completed, total, task))

        executor = BatchExecutor(orchestrator=mock_orchestrator)
        await executor.execute(sample_batch, on_progress=on_progress)

        # Should have calls for each task + final call
        assert len(progress_calls) >= 2
        # Last call should have completed == total
        assert progress_calls[-1][0] == progress_calls[-1][1]

    @pytest.mark.asyncio
    async def test_task_complete_callback(self, mock_orchestrator, sample_batch):
        """Test task completion callback."""
        completed_tasks = []

        def on_task_complete(result):
            completed_tasks.append(result)

        executor = BatchExecutor(orchestrator=mock_orchestrator)
        await executor.execute(sample_batch, on_task_complete=on_task_complete)

        assert len(completed_tasks) == 2
        assert all(isinstance(t, TaskResult) for t in completed_tasks)

    @pytest.mark.asyncio
    async def test_task_dependencies(self, mock_orchestrator):
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

        call_order = []

        async def track_call(*args, **kwargs):
            call_order.append(kwargs.get("task_id"))
            return MagicMock(
                success=True,
                data=TaskResult(task_id=kwargs.get("task_id", "test"), success=True),
                error=None,
            )

        mock_orchestrator.run = track_call

        executor = BatchExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(batch, max_concurrent=1)

        assert result.completed_tasks == 2
        # task_1 should execute before task_2 due to dependency
        assert call_order.index("task_1") < call_order.index("task_2")

    @pytest.mark.asyncio
    async def test_dependency_failure_skips_dependent(self, mock_orchestrator):
        """Test that dependent tasks are skipped when dependency fails."""
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

        mock_orchestrator.run = AsyncMock(
            return_value=MagicMock(success=False, error="Task failed")
        )

        executor = BatchExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(batch, fail_fast=False)

        # task_1 fails, task_2 should be skipped due to failed dependency
        assert result.failed_tasks == 2
        # Only task_1 should have been executed
        assert mock_orchestrator.run.call_count == 1

    @pytest.mark.asyncio
    async def test_cancel_execution(self, mock_orchestrator, sample_batch):
        """Test cancelling execution."""
        # Add more tasks
        for i in range(3, 10):
            sample_batch.tasks.append(
                AnalysisTask(
                    task_id=f"task_{i}",
                    source_file=f"{i}.c",
                    function_name=f"func_{i}",
                )
            )

        call_count = 0

        async def slow_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                executor.cancel()
            return MagicMock(
                success=True,
                data=TaskResult(task_id=kwargs.get("task_id", "test"), success=True),
                error=None,
            )

        mock_orchestrator.run = slow_run

        executor = BatchExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(sample_batch, max_concurrent=1)

        # Should have some cancelled tasks
        assert result.cancelled_tasks > 0
        assert executor.is_cancelled is True

    @pytest.mark.asyncio
    async def test_no_orchestrator_error(self, sample_batch):
        """Test error when no orchestrator is configured."""
        executor = BatchExecutor()
        result = await executor.execute(sample_batch)

        assert result.failed_tasks == 2
        assert all("No orchestrator" in r.errors[0] for r in result.results)

    @pytest.mark.asyncio
    async def test_exception_handling(self, mock_orchestrator, sample_batch):
        """Test exception handling during task execution."""
        mock_orchestrator.run = AsyncMock(side_effect=Exception("Unexpected error"))

        executor = BatchExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(sample_batch)

        assert result.failed_tasks == 2
        assert all("Unexpected error" in r.errors[0] for r in result.results)

    @pytest.mark.asyncio
    async def test_empty_batch(self, mock_orchestrator):
        """Test executing empty batch."""
        batch = BatchTask(
            batch_id="empty_batch",
            project_path="/path",
            tasks=[],
        )

        executor = BatchExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(batch)

        assert result.total_tasks == 0
        assert result.completed_tasks == 0
        assert result.success is True

    @pytest.mark.asyncio
    async def test_batch_result_duration(self, mock_orchestrator, sample_batch):
        """Test batch result has duration."""
        executor = BatchExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(sample_batch)

        assert result.start_time is not None
        assert result.end_time is not None
        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_execute_with_settings(self, mock_orchestrator, sample_batch):
        """Test execution with settings."""
        from fuzz_generator.config import Settings

        settings = Settings()
        settings.batch.fail_fast = True
        settings.batch.max_concurrent = 4

        executor = BatchExecutor(
            orchestrator=mock_orchestrator,
            settings=settings,
        )
        result = await executor.execute(sample_batch)

        assert result.completed_tasks == 2
