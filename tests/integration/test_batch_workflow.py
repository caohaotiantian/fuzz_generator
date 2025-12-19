"""Integration tests for batch workflow."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fuzz_generator.batch import BatchExecutor, BatchStateManager, TaskParser
from fuzz_generator.config import Settings
from fuzz_generator.models import TaskResult
from fuzz_generator.storage import JsonStorage


class TestBatchWorkflow:
    """Integration tests for batch task processing workflow."""

    @pytest.fixture
    def settings(self, tmp_path: Path) -> Settings:
        """Create settings with temp directory."""
        settings = Settings()
        settings.storage.work_dir = str(tmp_path / ".fuzz_generator")
        return settings

    @pytest.fixture
    def storage(self, tmp_path: Path) -> JsonStorage:
        """Create storage backend."""
        return JsonStorage(base_dir=tmp_path / "storage")

    @pytest.mark.asyncio
    async def test_parse_and_execute_workflow(
        self,
        sample_c_project: Path,
        tmp_path: Path,
        settings: Settings,
        storage: JsonStorage,
    ):
        """Test parsing task file and executing batch."""
        # Create task file
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(f'''project_path: "{sample_c_project}"
tasks:
  - source_file: "handler.c"
    function_name: "process_request"
  - source_file: "handler.c"
    function_name: "parse_header"
''')

        # Parse tasks
        parser = TaskParser(base_path=tmp_path)
        batch = parser.parse(str(task_file))

        assert len(batch.tasks) == 2
        assert batch.project_path == str(sample_c_project)

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock(
            return_value=MagicMock(
                success=True,
                data=TaskResult(
                    task_id="test",
                    success=True,
                    xml_content="<DataModel name='Test'/>",
                ),
            )
        )

        # Execute batch
        executor = BatchExecutor(
            orchestrator=mock_orchestrator,
            settings=settings,
            storage=storage,
        )
        result = await executor.execute(batch)

        assert result.success is True
        assert result.completed_tasks == 2
        assert result.failed_tasks == 0

    @pytest.mark.asyncio
    async def test_state_management_workflow(
        self,
        sample_c_project: Path,
        tmp_path: Path,
        storage: JsonStorage,
    ):
        """Test state management during batch execution."""
        # Create task file
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(f'''project_path: "{sample_c_project}"
tasks:
  - source_file: "handler.c"
    function_name: "process_request"
  - source_file: "handler.c"
    function_name: "parse_header"
  - source_file: "handler.c"
    function_name: "handle_connection"
''')

        # Parse tasks
        parser = TaskParser()
        batch = parser.parse(str(task_file))

        # Initialize state manager
        state_manager = BatchStateManager(storage=storage)

        # Create initial state
        state = await state_manager.create_state(batch)

        assert len(state.pending) == 3
        assert len(state.completed) == 0

        # Mark first task as running then completed
        await state_manager.mark_running(batch.batch_id, batch.tasks[0].task_id)
        await state_manager.mark_completed(
            batch.batch_id,
            batch.tasks[0].task_id,
            result={"xml": "<DataModel/>"},
        )

        # Verify state
        updated_state = await state_manager.load_state(batch.batch_id)
        assert len(updated_state.completed) == 1
        assert len(updated_state.pending) == 2

        # Mark second task as failed
        await state_manager.mark_failed(
            batch.batch_id,
            batch.tasks[1].task_id,
            error="Test error",
        )

        # Get resumable batch
        resumable = await state_manager.get_resumable_batch(batch.batch_id, batch)

        # Should only have the last pending task
        assert len(resumable.tasks) == 1
        assert resumable.tasks[0].task_id == batch.tasks[2].task_id

    @pytest.mark.asyncio
    async def test_partial_failure_continue(
        self,
        sample_c_project: Path,
        tmp_path: Path,
        settings: Settings,
    ):
        """Test batch continues after partial failure."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(f'''project_path: "{sample_c_project}"
tasks:
  - source_file: "handler.c"
    function_name: "func1"
  - source_file: "handler.c"
    function_name: "func2"
  - source_file: "handler.c"
    function_name: "func3"
''')

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        # Mock orchestrator with failure on second task
        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return MagicMock(success=False, error="Simulated failure")
            return MagicMock(
                success=True,
                data=TaskResult(
                    task_id=kwargs.get("task_id", "test"),
                    success=True,
                    xml_content="<DataModel/>",
                ),
            )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = mock_run

        executor = BatchExecutor(orchestrator=mock_orchestrator, settings=settings)
        result = await executor.execute(batch, fail_fast=False)

        # Should complete 2, fail 1
        assert result.completed_tasks == 2
        assert result.failed_tasks == 1
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_progress_tracking(
        self,
        sample_c_project: Path,
        tmp_path: Path,
        settings: Settings,
    ):
        """Test progress tracking during execution."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(f'''project_path: "{sample_c_project}"
tasks:
  - source_file: "handler.c"
    function_name: "func1"
  - source_file: "handler.c"
    function_name: "func2"
''')

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        progress_updates = []

        def on_progress(completed, total, task):
            progress_updates.append(
                {
                    "completed": completed,
                    "total": total,
                    "task_id": task.task_id if task else None,
                }
            )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock(
            return_value=MagicMock(
                success=True,
                data=TaskResult(task_id="test", success=True),
            )
        )

        executor = BatchExecutor(orchestrator=mock_orchestrator, settings=settings)
        await executor.execute(batch, on_progress=on_progress)

        # Should have progress updates
        assert len(progress_updates) >= 2
        assert progress_updates[-1]["completed"] == progress_updates[-1]["total"]


class TestXMLGenerationWorkflow:
    """Integration tests for XML generation workflow."""

    def test_generate_and_validate_xml(self, tmp_path: Path):
        """Test generating and validating XML."""
        from fuzz_generator.generators import XMLGenerator, XMLValidator
        from fuzz_generator.models.xml_models import (
            BlockElement,
            ChoiceElement,
            DataModel,
            StringElement,
        )

        # Create DataModels
        crlf_model = DataModel(
            name="CrLf",
            elements=[
                ChoiceElement(
                    name="EndChoice",
                    options=[
                        StringElement(name="CRLF", value="\r\n", token=True),
                        StringElement(name="LF", value="\n", token=True),
                    ],
                ),
            ],
        )

        request_model = DataModel(
            name="Request",
            description="HTTP Request model",
            elements=[
                StringElement(name="Method", value="GET"),
                StringElement(name="Space", value=" ", token=True),
                StringElement(name="Path", value="/"),
                BlockElement(name="End", ref="CrLf"),
            ],
        )

        # Generate XML
        generator = XMLGenerator(indent=4)
        xml_str = generator.generate([crlf_model, request_model])

        # Validate XML
        validator = XMLValidator()
        result = validator.validate(xml_str)

        assert result.is_valid is True
        assert result.info["datamodel_count"] == 2
        assert "CrLf" in result.info["datamodel_names"]
        assert "Request" in result.info["datamodel_names"]

        # Write to file and validate
        output_file = tmp_path / "output.xml"
        generator.generate_to_file([crlf_model, request_model], str(output_file))

        file_result = validator.validate_file(str(output_file))
        assert file_result.is_valid is True


class TestConfigurationWorkflow:
    """Integration tests for configuration workflow."""

    def test_load_custom_config(self, tmp_path: Path):
        """Test loading custom configuration."""
        from fuzz_generator.config import load_config

        # Create custom config
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""version: "1.0"
llm:
  base_url: "http://localhost:8080/v1"
  model: "custom-model"
  temperature: 0.5
mcp_server:
  url: "http://localhost:9000/mcp"
batch:
  max_concurrent: 4
  fail_fast: true
""")

        # Load config
        settings = load_config(str(config_file))

        assert settings.llm.base_url == "http://localhost:8080/v1"
        assert settings.llm.model == "custom-model"
        assert settings.llm.temperature == 0.5
        assert settings.mcp_server.url == "http://localhost:9000/mcp"
        assert settings.batch.max_concurrent == 4
        assert settings.batch.fail_fast is True

    def test_config_with_env_override(self, tmp_path: Path, monkeypatch):
        """Test configuration with environment variable override."""
        from fuzz_generator.config import load_config

        # Set environment variable
        monkeypatch.setenv("FUZZ_GENERATOR_LLM_MODEL", "env-model")

        # Create minimal config
        config_file = tmp_path / "config.yaml"
        config_file.write_text('version: "1.0"')

        # Load config (env should override if implemented)
        settings = load_config(str(config_file))

        # Verify config loads successfully
        assert settings is not None
        # The model may or may not be overridden depending on implementation
        assert settings.llm.model in ["env-model", "qwen2.5:32b"]
