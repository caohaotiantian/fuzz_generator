"""Tests for CodeAnalyzer agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fuzz_generator.agents.base import AgentConfig
from fuzz_generator.agents.code_analyzer import DEFAULT_SYSTEM_PROMPT, CodeAnalyzerAgent
from fuzz_generator.exceptions import AnalysisError
from fuzz_generator.models import FunctionInfo, ParameterDirection
from fuzz_generator.tools.query_tools import FunctionCodeResult


class TestCodeAnalyzerAgent:
    """Tests for CodeAnalyzerAgent."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client."""
        return AsyncMock()

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        return client

    @pytest.fixture
    def agent(self, mock_mcp_client):
        """Create CodeAnalyzer agent without LLM."""
        return CodeAnalyzerAgent(mcp_client=mock_mcp_client)

    @pytest.fixture
    def agent_with_llm(self, mock_mcp_client, mock_llm_client):
        """Create CodeAnalyzer agent with LLM."""
        return CodeAnalyzerAgent(
            mcp_client=mock_mcp_client,
            llm_client=mock_llm_client,
        )

    def test_agent_creation(self, agent):
        """Test agent creation with default config."""
        assert agent.name == "CodeAnalyzer"
        assert DEFAULT_SYSTEM_PROMPT in agent.config.system_prompt

    def test_agent_custom_config(self, mock_mcp_client):
        """Test agent with custom config."""
        config = AgentConfig(
            name="CustomAnalyzer",
            system_prompt="Custom prompt",
            max_iterations=20,
        )
        agent = CodeAnalyzerAgent(
            mcp_client=mock_mcp_client,
            config=config,
        )
        assert agent.name == "CustomAnalyzer"
        assert agent.config.max_iterations == 20

    @pytest.mark.asyncio
    async def test_analyze_function_basic(self, agent, mock_mcp_client):
        """Test basic function analysis without LLM."""
        mock_result = FunctionCodeResult(
            success=True,
            function_name="process_request",
            code="int process_request(char* buffer, int length) { return 0; }",
            file="handler.c",
            line_number=10,
            signature="int process_request(char* buffer, int length)",
        )

        # Patch the get_function_code function
        with patch(
            "fuzz_generator.agents.code_analyzer.get_function_code",
            return_value=mock_result,
        ):
            result = await agent.run(
                project_name="test_project",
                function_name="process_request",
            )

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, FunctionInfo)
        assert result.data.name == "process_request"

    @pytest.mark.asyncio
    async def test_analyze_function_not_found(self, agent):
        """Test analysis when function not found."""
        mock_result = FunctionCodeResult(
            success=False,
            error="Function not found",
        )

        with patch(
            "fuzz_generator.agents.code_analyzer.get_function_code",
            return_value=mock_result,
        ):
            result = await agent.run(
                project_name="test_project",
                function_name="nonexistent",
            )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_analyze_with_llm(self, agent_with_llm, mock_llm_client):
        """Test analysis with LLM."""
        mock_code_result = FunctionCodeResult(
            success=True,
            function_name="process_request",
            code="int process_request(char* buffer, int length) { return 0; }",
            file="handler.c",
            line_number=10,
            signature="int process_request(char* buffer, int length)",
        )

        # Mock LLM response
        llm_response = MagicMock()
        llm_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="""
{
    "function_name": "process_request",
    "file_path": "handler.c",
    "return_type": "int",
    "description": "Process incoming request",
    "parameters": [
        {
            "name": "buffer",
            "type": "char*",
            "direction": "in",
            "description": "Input buffer"
        },
        {
            "name": "length",
            "type": "int",
            "direction": "in",
            "description": "Buffer length"
        }
    ]
}
                    """
                )
            )
        ]
        mock_llm_client.chat.completions.create.return_value = llm_response

        with patch(
            "fuzz_generator.agents.code_analyzer.get_function_code",
            return_value=mock_code_result,
        ):
            result = await agent_with_llm.run(
                project_name="test_project",
                function_name="process_request",
            )

        assert result.success is True
        assert result.data.name == "process_request"
        assert len(result.data.parameters) == 2
        assert result.data.parameters[0].name == "buffer"

    @pytest.mark.asyncio
    async def test_analyze_convenience_method(self, agent):
        """Test analyze() convenience method."""
        mock_result = FunctionCodeResult(
            success=True,
            function_name="main",
            code="int main() { return 0; }",
            file="main.c",
            line_number=1,
        )

        with patch(
            "fuzz_generator.agents.code_analyzer.get_function_code",
            return_value=mock_result,
        ):
            func_info = await agent.analyze(
                project_name="test",
                function_name="main",
            )

        assert isinstance(func_info, FunctionInfo)
        assert func_info.name == "main"

    @pytest.mark.asyncio
    async def test_analyze_raises_on_failure(self, agent):
        """Test that analyze() raises AnalysisError on failure."""
        mock_result = FunctionCodeResult(
            success=False,
            error="Function not found",
        )

        with patch(
            "fuzz_generator.agents.code_analyzer.get_function_code",
            return_value=mock_result,
        ):
            with pytest.raises(AnalysisError, match="Failed to analyze"):
                await agent.analyze(
                    project_name="test",
                    function_name="nonexistent",
                )

    def test_basic_analysis(self, agent):
        """Test basic analysis without LLM."""
        function_code = {
            "function_name": "test_func",
            "code": "void test_func(int x) {}",
            "file": "test.c",
            "line_number": 1,
            "signature": "void test_func(int x)",
        }

        result = agent._basic_analysis("test_func", function_code)

        assert isinstance(result, FunctionInfo)
        assert result.name == "test_func"
        assert result.file_path == "test.c"

    def test_build_function_info(self, agent):
        """Test building FunctionInfo from parsed response."""
        parsed = {
            "function_name": "my_func",
            "file_path": "my_file.c",
            "line_number": 100,
            "return_type": "void",
            "description": "My function",
            "parameters": [
                {
                    "name": "param1",
                    "type": "int",
                    "direction": "in",
                    "description": "First param",
                },
                {
                    "name": "param2",
                    "type": "int*",
                    "direction": "out",
                    "description": "Output param",
                },
            ],
        }
        function_code = {"code": "void my_func() {}"}

        result = agent._build_function_info(parsed, function_code)

        assert result.name == "my_func"
        assert result.return_type == "void"
        assert len(result.parameters) == 2
        assert result.parameters[0].direction == ParameterDirection.IN
        assert result.parameters[1].direction == ParameterDirection.OUT

    @pytest.mark.asyncio
    async def test_get_function_code_error(self, agent):
        """Test handling of MCP error."""
        with patch(
            "fuzz_generator.agents.code_analyzer.get_function_code",
            side_effect=Exception("MCP error"),
        ):
            result = await agent._get_function_code("func", None)
            assert result is None
