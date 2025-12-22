"""Tests for TwoPhaseWorkflow and related components."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fuzz_generator.agents.autogen_agents import (
    ConversationRecorder,
    OutputValidator,
    PromptLoader,
    TwoPhaseWorkflow,
    create_analysis_tools,
)
from fuzz_generator.models import AnalysisTask, TaskResult


class TestPromptLoader:
    """Tests for PromptLoader class."""

    def test_loader_creation(self):
        """Test PromptLoader can be created."""
        loader = PromptLoader()
        assert loader is not None

    def test_load_nonexistent(self):
        """Test loading nonexistent prompt returns empty template."""
        loader = PromptLoader()
        template = loader.load("nonexistent_agent")
        assert template is not None
        assert template.system_prompt == ""


class TestOutputValidator:
    """Tests for OutputValidator class."""

    def test_validator_creation(self):
        """Test OutputValidator can be created."""
        validator = OutputValidator()
        assert validator is not None

    def test_extract_json_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        text = '```json\n{"key": "value"}\n```'
        result = OutputValidator.extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_raw(self):
        """Test extracting raw JSON."""
        text = '{"key": "value"}'
        result = OutputValidator.extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_invalid(self):
        """Test extracting invalid JSON returns None."""
        text = "not json at all"
        result = OutputValidator.extract_json(text)
        assert result is None

    def test_validate_analysis_result_valid(self):
        """Test validating valid analysis result."""
        data = {
            "status": "success",
            "function": {"name": "test"},
            "parameters": [],
        }
        assert OutputValidator.validate_analysis_result(data) is True

    def test_validate_analysis_result_missing_field(self):
        """Test validating analysis result with missing field."""
        data = {"status": "success"}  # Missing function and parameters
        assert OutputValidator.validate_analysis_result(data) is False


class TestConversationRecorder:
    """Tests for ConversationRecorder class."""

    def test_recorder_creation(self, tmp_path: Path):
        """Test ConversationRecorder can be created."""
        recorder = ConversationRecorder(tmp_path)
        assert recorder is not None

    def test_record_message(self, tmp_path: Path):
        """Test recording a message."""
        recorder = ConversationRecorder(tmp_path)
        recorder.record("TestAgent", "Hello", role="user")
        assert len(recorder.messages) == 1
        assert recorder.messages[0]["agent"] == "TestAgent"
        assert recorder.messages[0]["content"] == "Hello"
        assert recorder.messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_save_conversation(self, tmp_path: Path):
        """Test saving conversation to file."""
        recorder = ConversationRecorder(tmp_path)
        recorder.record("Agent1", "user", "Message 1")
        recorder.record("Agent2", "assistant", "Message 2")

        await recorder.save("test_task")

        # Check file was created
        result_dir = tmp_path / "results" / "test_task" / "intermediate"
        assert result_dir.exists()
        conversation_file = result_dir / "agent_conversations.json"
        assert conversation_file.exists()


class TestCreateAnalysisTools:
    """Tests for create_analysis_tools function."""

    def test_create_tools(self):
        """Test creating analysis tools."""
        mock_client = MagicMock()
        tools = create_analysis_tools(mock_client, "test_project")

        assert isinstance(tools, list)
        assert len(tools) > 0

        # Check tool names
        tool_names = [t.__name__ for t in tools]
        assert "get_function_code" in tool_names
        assert "list_functions" in tool_names
        assert "track_dataflow" in tool_names
        assert "get_callees" in tool_names


class TestTwoPhaseWorkflow:
    """Tests for TwoPhaseWorkflow class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        # Configure LLM settings
        settings.llm = MagicMock()
        settings.llm.base_url = "http://localhost:1234/v1"
        settings.llm.model = "test-model"
        settings.llm.api_key = "test-key"
        settings.llm.temperature = 0.7
        settings.llm.max_tokens = 4096
        settings.llm.timeout = 60
        # Configure agent settings
        settings.agents = MagicMock()
        settings.agents.code_analyzer = MagicMock()
        settings.agents.code_analyzer.max_iterations = 10
        settings.agents.model_generator = MagicMock()
        settings.agents.model_generator.max_iterations = 5
        return settings

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client."""
        return MagicMock()

    def test_workflow_creation(self, mock_settings, mock_mcp_client, tmp_path: Path):
        """Test TwoPhaseWorkflow can be created."""
        workflow = TwoPhaseWorkflow(
            settings=mock_settings,
            mcp_client=mock_mcp_client,
            project_name="test_project",
            storage_path=tmp_path,
        )
        assert workflow is not None
        assert workflow.project_name == "test_project"

    @pytest.mark.asyncio
    async def test_workflow_close(self, mock_settings, mock_mcp_client, tmp_path: Path):
        """Test workflow close method."""
        workflow = TwoPhaseWorkflow(
            settings=mock_settings,
            mcp_client=mock_mcp_client,
            project_name="test_project",
            storage_path=tmp_path,
        )

        # Mock the model client's close method
        workflow.model_client.close = AsyncMock()

        await workflow.close()
        workflow.model_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_context_manager(self, mock_settings, mock_mcp_client, tmp_path: Path):
        """Test workflow as async context manager."""
        async with TwoPhaseWorkflow(
            settings=mock_settings,
            mcp_client=mock_mcp_client,
            project_name="test_project",
            storage_path=tmp_path,
        ) as workflow:
            assert workflow is not None
            workflow.model_client.close = AsyncMock()

        # close should have been called

    def test_get_prompt(self, mock_settings, mock_mcp_client, tmp_path: Path):
        """Test getting prompt with fallback."""
        workflow = TwoPhaseWorkflow(
            settings=mock_settings,
            mcp_client=mock_mcp_client,
            project_name="test_project",
            storage_path=tmp_path,
        )

        prompt = workflow._get_prompt("nonexistent", "fallback prompt {custom_knowledge}")
        assert prompt == "fallback prompt "  # custom_knowledge replaced with empty

    def test_get_prompt_with_knowledge(self, mock_settings, mock_mcp_client, tmp_path: Path):
        """Test getting prompt with custom knowledge."""
        workflow = TwoPhaseWorkflow(
            settings=mock_settings,
            mcp_client=mock_mcp_client,
            project_name="test_project",
            custom_knowledge="Some custom knowledge",
            storage_path=tmp_path,
        )

        prompt = workflow._get_prompt("nonexistent", "Base prompt {custom_knowledge}")
        assert "背景知识" in prompt
        assert "Some custom knowledge" in prompt

    def test_wrap_xml(self, mock_settings, mock_mcp_client, tmp_path: Path):
        """Test XML wrapping."""
        workflow = TwoPhaseWorkflow(
            settings=mock_settings,
            mcp_client=mock_mcp_client,
            project_name="test_project",
            storage_path=tmp_path,
        )

        xml = workflow._wrap_xml('<DataModel name="Test"/>')
        assert '<?xml version="1.0" encoding="utf-8"?>' in xml
        assert "<Secray>" in xml
        assert "</Secray>" in xml
        assert '<DataModel name="Test"/>' in xml

    def test_wrap_xml_removes_existing_declaration(self, mock_settings, mock_mcp_client, tmp_path: Path):
        """Test XML wrapping removes existing declaration."""
        workflow = TwoPhaseWorkflow(
            settings=mock_settings,
            mcp_client=mock_mcp_client,
            project_name="test_project",
            storage_path=tmp_path,
        )

        xml = workflow._wrap_xml('<?xml version="1.0"?><DataModel name="Test"/>')
        # Should only have one declaration
        assert xml.count('<?xml') == 1


class TestTwoPhaseWorkflowIntegration:
    """Integration tests for TwoPhaseWorkflow (requires mocking LLM)."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        # Configure LLM settings
        settings.llm = MagicMock()
        settings.llm.base_url = "http://localhost:1234/v1"
        settings.llm.model = "test-model"
        settings.llm.api_key = "test-key"
        settings.llm.temperature = 0.7
        settings.llm.max_tokens = 4096
        settings.llm.timeout = 60
        # Configure agent settings
        settings.agents = MagicMock()
        settings.agents.code_analyzer = MagicMock()
        settings.agents.code_analyzer.max_iterations = 10
        settings.agents.model_generator = MagicMock()
        settings.agents.model_generator.max_iterations = 5
        return settings

    @pytest.fixture
    def sample_task(self):
        """Create sample analysis task."""
        return AnalysisTask(
            task_id="test_task",
            source_file="handler.c",
            function_name="process_request",
            output_name="RequestModel",
        )

    @pytest.mark.asyncio
    async def test_run_returns_task_result(
        self, mock_settings, sample_task, tmp_path: Path
    ):
        """Test that run returns TaskResult."""
        mock_mcp_client = MagicMock()

        # Patch CustomModelClient to avoid actual LLM calls
        with patch(
            "fuzz_generator.agents.autogen_agents.CustomModelClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            workflow = TwoPhaseWorkflow(
                settings=mock_settings,
                mcp_client=mock_mcp_client,
                project_name="test_project",
                storage_path=tmp_path,
            )

            # Patch internal methods
            workflow._run_analysis_phase = AsyncMock(return_value={
                "status": "success",
                "function": {"name": "process_request"},
                "parameters": [],
            })
            workflow._run_generation_phase = AsyncMock(
                return_value='<DataModel name="RequestModel"/>'
            )

            result = await workflow.run(sample_task)

            assert isinstance(result, TaskResult)
            assert result.task_id == "test_task"

    @pytest.mark.asyncio
    async def test_run_handles_analysis_failure(
        self, mock_settings, sample_task, tmp_path: Path
    ):
        """Test that run handles analysis phase failure."""
        mock_mcp_client = MagicMock()

        with patch(
            "fuzz_generator.agents.autogen_agents.CustomModelClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            workflow = TwoPhaseWorkflow(
                settings=mock_settings,
                mcp_client=mock_mcp_client,
                project_name="test_project",
                storage_path=tmp_path,
            )

            # Make analysis phase return None (failure)
            workflow._run_analysis_phase = AsyncMock(return_value=None)

            result = await workflow.run(sample_task)

            assert isinstance(result, TaskResult)
            assert result.success is False
            assert len(result.errors) > 0

