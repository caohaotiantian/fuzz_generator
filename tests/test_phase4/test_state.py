"""Tests for batch state management."""

import pytest

from fuzz_generator.batch.state import BatchState, BatchStateManager
from fuzz_generator.models import AnalysisTask, BatchTask
from fuzz_generator.storage import JsonStorage


class TestBatchState:
    """Tests for BatchState model."""

    def test_default_values(self):
        """Test default values."""
        state = BatchState(
            batch_id="test",
            project_path="/path",
        )
        assert state.batch_id == "test"
        assert state.completed == []
        assert state.failed == []
        assert state.pending == []
        assert state.running == []

    def test_total_count(self):
        """Test total_count calculation."""
        state = BatchState(
            batch_id="test",
            project_path="/path",
            completed=["task_1", "task_2"],
            failed=["task_3"],
            pending=["task_4", "task_5"],
            running=["task_6"],
        )
        assert state.total_count == 6

    def test_is_complete_true(self):
        """Test is_complete when all tasks finished."""
        state = BatchState(
            batch_id="test",
            project_path="/path",
            completed=["task_1", "task_2"],
            failed=["task_3"],
            pending=[],
            running=[],
        )
        assert state.is_complete is True

    def test_is_complete_false_pending(self):
        """Test is_complete with pending tasks."""
        state = BatchState(
            batch_id="test",
            project_path="/path",
            completed=["task_1"],
            pending=["task_2"],
        )
        assert state.is_complete is False

    def test_is_complete_false_running(self):
        """Test is_complete with running tasks."""
        state = BatchState(
            batch_id="test",
            project_path="/path",
            completed=["task_1"],
            running=["task_2"],
        )
        assert state.is_complete is False


class TestBatchStateManager:
    """Tests for BatchStateManager class."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create temporary storage."""
        return JsonStorage(base_dir=tmp_path)

    @pytest.fixture
    def manager(self, storage):
        """Create state manager."""
        return BatchStateManager(storage=storage)

    @pytest.fixture
    def sample_batch(self) -> BatchTask:
        """Create sample batch."""
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
                AnalysisTask(
                    task_id="task_3",
                    source_file="c.c",
                    function_name="func_c",
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_create_state(self, manager, sample_batch):
        """Test creating initial state."""
        state = await manager.create_state(sample_batch)

        assert state.batch_id == "test_batch"
        assert state.project_path == "/path/to/project"
        assert len(state.pending) == 3
        assert len(state.completed) == 0

    @pytest.mark.asyncio
    async def test_save_and_load_state(self, manager):
        """Test saving and loading state."""
        batch_id = "batch_001"
        state = BatchState(
            batch_id=batch_id,
            project_path="/path",
            completed=["task_1", "task_2"],
            failed=["task_3"],
            pending=["task_4", "task_5"],
        )

        await manager.save_state(batch_id, state)
        loaded = await manager.load_state(batch_id)

        assert loaded is not None
        assert loaded.batch_id == batch_id
        assert loaded.completed == ["task_1", "task_2"]
        assert loaded.failed == ["task_3"]
        assert loaded.pending == ["task_4", "task_5"]

    @pytest.mark.asyncio
    async def test_load_nonexistent_state(self, manager):
        """Test loading non-existent state."""
        loaded = await manager.load_state("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_has_state(self, manager):
        """Test checking if state exists."""
        batch_id = "batch_001"

        assert await manager.has_state(batch_id) is False

        state = BatchState(batch_id=batch_id, project_path="/path")
        await manager.save_state(batch_id, state)

        assert await manager.has_state(batch_id) is True

    @pytest.mark.asyncio
    async def test_delete_state(self, manager):
        """Test deleting state."""
        batch_id = "batch_001"
        state = BatchState(batch_id=batch_id, project_path="/path")
        await manager.save_state(batch_id, state)

        assert await manager.has_state(batch_id) is True

        result = await manager.delete_state(batch_id)
        assert result is True

        assert await manager.has_state(batch_id) is False

    @pytest.mark.asyncio
    async def test_get_pending_tasks(self, manager):
        """Test getting pending tasks."""
        batch_id = "batch_001"
        state = BatchState(
            batch_id=batch_id,
            project_path="/path",
            pending=["task_2", "task_3"],
            completed=["task_1"],
        )
        await manager.save_state(batch_id, state)

        pending = await manager.get_pending_tasks(batch_id)
        assert pending == ["task_2", "task_3"]

    @pytest.mark.asyncio
    async def test_get_pending_tasks_nonexistent(self, manager):
        """Test getting pending tasks for non-existent batch."""
        pending = await manager.get_pending_tasks("nonexistent")
        assert pending == []

    @pytest.mark.asyncio
    async def test_get_completed_tasks(self, manager):
        """Test getting completed tasks."""
        batch_id = "batch_001"
        state = BatchState(
            batch_id=batch_id,
            project_path="/path",
            completed=["task_1", "task_2"],
        )
        await manager.save_state(batch_id, state)

        completed = await manager.get_completed_tasks(batch_id)
        assert completed == ["task_1", "task_2"]

    @pytest.mark.asyncio
    async def test_get_failed_tasks(self, manager):
        """Test getting failed tasks."""
        batch_id = "batch_001"
        state = BatchState(
            batch_id=batch_id,
            project_path="/path",
            failed=["task_3"],
        )
        await manager.save_state(batch_id, state)

        failed = await manager.get_failed_tasks(batch_id)
        assert failed == ["task_3"]

    @pytest.mark.asyncio
    async def test_mark_running(self, manager):
        """Test marking task as running."""
        batch_id = "batch_001"
        state = BatchState(
            batch_id=batch_id,
            project_path="/path",
            pending=["task_1", "task_2"],
        )
        await manager.save_state(batch_id, state)

        await manager.mark_running(batch_id, "task_1")

        updated = await manager.load_state(batch_id)
        assert updated is not None
        assert "task_1" not in updated.pending
        assert "task_1" in updated.running

    @pytest.mark.asyncio
    async def test_mark_completed(self, manager):
        """Test marking task as completed."""
        batch_id = "batch_001"
        state = BatchState(
            batch_id=batch_id,
            project_path="/path",
            pending=["task_1", "task_2"],
        )
        await manager.save_state(batch_id, state)

        await manager.mark_completed(batch_id, "task_1", result={"xml": "<DataModel/>"})

        updated = await manager.load_state(batch_id)
        assert updated is not None
        assert "task_1" in updated.completed
        assert "task_1" not in updated.pending
        assert updated.task_results.get("task_1") == {"xml": "<DataModel/>"}

    @pytest.mark.asyncio
    async def test_mark_completed_from_running(self, manager):
        """Test marking running task as completed."""
        batch_id = "batch_001"
        state = BatchState(
            batch_id=batch_id,
            project_path="/path",
            running=["task_1"],
        )
        await manager.save_state(batch_id, state)

        await manager.mark_completed(batch_id, "task_1")

        updated = await manager.load_state(batch_id)
        assert updated is not None
        assert "task_1" in updated.completed
        assert "task_1" not in updated.running

    @pytest.mark.asyncio
    async def test_mark_failed(self, manager):
        """Test marking task as failed."""
        batch_id = "batch_001"
        state = BatchState(
            batch_id=batch_id,
            project_path="/path",
            pending=["task_1", "task_2"],
        )
        await manager.save_state(batch_id, state)

        await manager.mark_failed(batch_id, "task_1", error="Something went wrong")

        updated = await manager.load_state(batch_id)
        assert updated is not None
        assert "task_1" in updated.failed
        assert "task_1" not in updated.pending
        assert updated.task_results.get("task_1") == {"error": "Something went wrong"}

    @pytest.mark.asyncio
    async def test_reset_running(self, manager):
        """Test resetting running tasks."""
        batch_id = "batch_001"
        state = BatchState(
            batch_id=batch_id,
            project_path="/path",
            running=["task_1", "task_2"],
            pending=["task_3"],
        )
        await manager.save_state(batch_id, state)

        await manager.reset_running(batch_id)

        updated = await manager.load_state(batch_id)
        assert updated is not None
        assert len(updated.running) == 0
        assert set(updated.pending) == {"task_1", "task_2", "task_3"}

    @pytest.mark.asyncio
    async def test_get_resumable_batch(self, manager, sample_batch):
        """Test getting resumable batch."""
        batch_id = sample_batch.batch_id

        # Create state with some completed tasks
        state = BatchState(
            batch_id=batch_id,
            project_path=sample_batch.project_path,
            completed=["task_1"],
            pending=["task_2", "task_3"],
        )
        await manager.save_state(batch_id, state)

        resumable = await manager.get_resumable_batch(batch_id, sample_batch)

        assert len(resumable.tasks) == 2
        assert all(t.task_id in ["task_2", "task_3"] for t in resumable.tasks)

    @pytest.mark.asyncio
    async def test_get_resumable_batch_with_running(self, manager, sample_batch):
        """Test getting resumable batch resets running tasks."""
        batch_id = sample_batch.batch_id

        state = BatchState(
            batch_id=batch_id,
            project_path=sample_batch.project_path,
            completed=["task_1"],
            running=["task_2"],
            pending=["task_3"],
        )
        await manager.save_state(batch_id, state)

        resumable = await manager.get_resumable_batch(batch_id, sample_batch)

        # task_2 should be back in pending and included
        assert len(resumable.tasks) == 2
        task_ids = [t.task_id for t in resumable.tasks]
        assert "task_2" in task_ids
        assert "task_3" in task_ids

    @pytest.mark.asyncio
    async def test_get_resumable_batch_no_state(self, manager, sample_batch):
        """Test getting resumable batch when no state exists."""
        resumable = await manager.get_resumable_batch("nonexistent", sample_batch)

        assert resumable == sample_batch

    @pytest.mark.asyncio
    async def test_list_batches(self, manager):
        """Test listing all batches."""
        # Create multiple batch states
        for i in range(3):
            state = BatchState(batch_id=f"batch_{i}", project_path="/path")
            await manager.save_state(f"batch_{i}", state)

        batches = await manager.list_batches()

        assert len(batches) == 3
        assert set(batches) == {"batch_0", "batch_1", "batch_2"}

    @pytest.mark.asyncio
    async def test_get_batch_summary(self, manager):
        """Test getting batch summary."""
        batch_id = "batch_001"
        state = BatchState(
            batch_id=batch_id,
            project_path="/path/to/project",
            completed=["task_1", "task_2"],
            failed=["task_3"],
            pending=["task_4"],
            running=["task_5"],
        )
        await manager.save_state(batch_id, state)

        summary = await manager.get_batch_summary(batch_id)

        assert summary is not None
        assert summary["batch_id"] == batch_id
        assert summary["project_path"] == "/path/to/project"
        assert summary["total"] == 5
        assert summary["completed"] == 2
        assert summary["failed"] == 1
        assert summary["pending"] == 1
        assert summary["running"] == 1
        assert summary["is_complete"] is False

    @pytest.mark.asyncio
    async def test_get_batch_summary_nonexistent(self, manager):
        """Test getting summary for non-existent batch."""
        summary = await manager.get_batch_summary("nonexistent")
        assert summary is None

    @pytest.mark.asyncio
    async def test_clear_all(self, manager):
        """Test clearing all batch states."""
        # Create multiple states
        for i in range(3):
            state = BatchState(batch_id=f"batch_{i}", project_path="/path")
            await manager.save_state(f"batch_{i}", state)

        batches = await manager.list_batches()
        assert len(batches) == 3

        await manager.clear_all()

        batches = await manager.list_batches()
        assert len(batches) == 0

    @pytest.mark.asyncio
    async def test_mark_operations_on_nonexistent(self, manager):
        """Test mark operations on non-existent batch don't raise."""
        # These should not raise exceptions
        await manager.mark_running("nonexistent", "task_1")
        await manager.mark_completed("nonexistent", "task_1")
        await manager.mark_failed("nonexistent", "task_1")

    @pytest.mark.asyncio
    async def test_state_updated_at(self, manager):
        """Test state updated_at is updated on save."""
        batch_id = "batch_001"
        state = BatchState(batch_id=batch_id, project_path="/path")

        await manager.save_state(batch_id, state)
        loaded1 = await manager.load_state(batch_id)
        time1 = loaded1.updated_at

        # Make a change
        state.completed.append("task_1")
        await manager.save_state(batch_id, state)
        loaded2 = await manager.load_state(batch_id)
        time2 = loaded2.updated_at

        assert time2 >= time1

    @pytest.mark.asyncio
    async def test_task_results_persistence(self, manager):
        """Test task results are persisted."""
        batch_id = "batch_001"
        state = BatchState(
            batch_id=batch_id,
            project_path="/path",
            pending=["task_1"],
        )
        await manager.save_state(batch_id, state)

        await manager.mark_completed(
            batch_id,
            "task_1",
            result={
                "xml_content": "<DataModel name='Test'/>",
                "duration": 1.5,
            },
        )

        loaded = await manager.load_state(batch_id)
        assert loaded is not None
        assert "task_1" in loaded.task_results
        assert loaded.task_results["task_1"]["xml_content"] == "<DataModel name='Test'/>"
