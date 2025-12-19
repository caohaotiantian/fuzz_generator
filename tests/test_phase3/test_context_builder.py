"""Tests for ContextBuilder agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fuzz_generator.agents.base import AgentConfig
from fuzz_generator.agents.context_builder import ContextBuilderAgent
from fuzz_generator.models import (
    AnalysisContext,
    FunctionInfo,
    ParameterDirection,
    ParameterInfo,
)


class TestContextBuilderAgent:
    """Tests for ContextBuilderAgent."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client."""
        return AsyncMock()

    @pytest.fixture
    def agent(self, mock_mcp_client):
        """Create ContextBuilder agent."""
        return ContextBuilderAgent(mcp_client=mock_mcp_client)

    @pytest.fixture
    def function_info(self):
        """Create sample function info."""
        return FunctionInfo(
            name="process_request",
            file_path="handler.c",
            line_number=10,
            return_type="int",
            parameters=[
                ParameterInfo(
                    name="buffer",
                    type="char*",
                    direction=ParameterDirection.IN,
                    description="Input buffer",
                ),
                ParameterInfo(
                    name="length",
                    type="int",
                    direction=ParameterDirection.IN,
                    description="Buffer length",
                ),
            ],
            description="Process incoming request",
        )

    def test_agent_creation(self, agent):
        """Test agent creation."""
        assert agent.name == "ContextBuilder"
        assert agent.config.max_iterations == 15

    def test_agent_custom_config(self, mock_mcp_client):
        """Test agent with custom config."""
        config = AgentConfig(
            name="CustomBuilder",
            max_iterations=25,
        )
        agent = ContextBuilderAgent(
            mcp_client=mock_mcp_client,
            config=config,
        )
        assert agent.name == "CustomBuilder"
        assert agent.config.max_iterations == 25

    @pytest.mark.asyncio
    async def test_build_context_success(self, agent, function_info):
        """Test successful context building."""
        # Mock MCP tool responses
        mock_dataflow = MagicMock()
        mock_dataflow.success = True
        mock_dataflow.flows = []

        mock_cfg = MagicMock()
        mock_cfg.success = True
        mock_cfg.nodes = []
        mock_cfg.has_loops = False
        mock_cfg.has_conditions = True

        mock_callers = MagicMock()
        mock_callers.success = True
        mock_callers.callers = []

        mock_callees = MagicMock()
        mock_callees.success = True
        mock_callees.callees = []

        with (
            patch(
                "fuzz_generator.agents.context_builder.track_dataflow",
                return_value=mock_dataflow,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_control_flow_graph",
                return_value=mock_cfg,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_callers",
                return_value=mock_callers,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_callees",
                return_value=mock_callees,
            ),
        ):
            result = await agent.run(
                project_name="test_project",
                function_info=function_info,
            )

        assert result.success is True
        assert isinstance(result.data, AnalysisContext)
        assert result.data.function_info.name == "process_request"

    @pytest.mark.asyncio
    async def test_build_context_with_flows(self, agent, function_info):
        """Test context building with data flows."""
        # Create mock flow with proper structure
        mock_flow = MagicMock()
        mock_flow.source = {
            "code": "buffer",
            "file": "handler.c",
            "line": 10,
            "node_type": "source",
        }
        mock_flow.sink = {"code": "strcpy", "file": "handler.c", "line": 15, "node_type": "sink"}
        mock_flow.path_length = 3
        mock_flow.path_details = []

        mock_dataflow = MagicMock()
        mock_dataflow.success = True
        mock_dataflow.flows = [mock_flow]

        mock_cfg = MagicMock()
        mock_cfg.success = True
        mock_cfg.nodes = []
        mock_cfg.has_loops = False
        mock_cfg.has_conditions = False

        mock_callers = MagicMock()
        mock_callers.success = True
        mock_callers.callers = []

        mock_callees = MagicMock()
        mock_callees.success = True
        mock_callees.callees = []

        with (
            patch(
                "fuzz_generator.agents.context_builder.track_dataflow",
                return_value=mock_dataflow,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_control_flow_graph",
                return_value=mock_cfg,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_callers",
                return_value=mock_callers,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_callees",
                return_value=mock_callees,
            ),
        ):
            result = await agent.run(
                project_name="test_project",
                function_info=function_info,
            )

        assert result.success is True
        # Data flows should be populated (2 flows, one per input parameter)
        assert len(result.data.data_flows) == 2

    @pytest.mark.asyncio
    async def test_build_context_with_call_graph(self, agent, function_info):
        """Test context building with call graph."""
        mock_dataflow = MagicMock()
        mock_dataflow.success = True
        mock_dataflow.flows = []

        mock_cfg = MagicMock()
        mock_cfg.success = True
        mock_cfg.nodes = []
        mock_cfg.has_loops = True
        mock_cfg.has_conditions = True

        # Mock caller
        mock_caller = MagicMock()
        mock_caller.name = "main"
        mock_caller.filename = "main.c"
        mock_caller.line_number = 5

        mock_callers = MagicMock()
        mock_callers.success = True
        mock_callers.callers = [mock_caller]

        # Mock callee
        mock_callee = MagicMock()
        mock_callee.name = "helper"
        mock_callee.filename = "utils.c"
        mock_callee.line_number = 20

        mock_callees = MagicMock()
        mock_callees.success = True
        mock_callees.callees = [mock_callee]

        with (
            patch(
                "fuzz_generator.agents.context_builder.track_dataflow",
                return_value=mock_dataflow,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_control_flow_graph",
                return_value=mock_cfg,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_callers",
                return_value=mock_callers,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_callees",
                return_value=mock_callees,
            ),
        ):
            result = await agent.run(
                project_name="test_project",
                function_info=function_info,
            )

        assert result.success is True
        # CallGraphInfo uses list[dict] structure
        assert len(result.data.call_graph.callers) == 1
        assert len(result.data.call_graph.callees) == 1
        # Callers should have name attribute (it's a CallerInfo model)
        assert result.data.call_graph.callers[0].name == "main"

    @pytest.mark.asyncio
    async def test_build_context_convenience_method(self, agent, function_info):
        """Test build_context() convenience method."""
        mock_dataflow = MagicMock()
        mock_dataflow.success = True
        mock_dataflow.flows = []

        mock_cfg = MagicMock()
        mock_cfg.success = True
        mock_cfg.nodes = []
        mock_cfg.has_loops = False
        mock_cfg.has_conditions = False

        mock_callers = MagicMock()
        mock_callers.success = True
        mock_callers.callers = []

        mock_callees = MagicMock()
        mock_callees.success = True
        mock_callees.callees = []

        with (
            patch(
                "fuzz_generator.agents.context_builder.track_dataflow",
                return_value=mock_dataflow,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_control_flow_graph",
                return_value=mock_cfg,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_callers",
                return_value=mock_callers,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_callees",
                return_value=mock_callees,
            ),
        ):
            context = await agent.build_context(
                project_name="test",
                function_info=function_info,
            )

        assert isinstance(context, AnalysisContext)

    @pytest.mark.asyncio
    async def test_handle_dataflow_error(self, agent, function_info):
        """Test handling of dataflow error."""
        # Dataflow fails but other tools succeed
        mock_cfg = MagicMock()
        mock_cfg.success = True
        mock_cfg.nodes = []
        mock_cfg.has_loops = False
        mock_cfg.has_conditions = False

        mock_callers = MagicMock()
        mock_callers.success = True
        mock_callers.callers = []

        mock_callees = MagicMock()
        mock_callees.success = True
        mock_callees.callees = []

        with (
            patch(
                "fuzz_generator.agents.context_builder.track_dataflow",
                side_effect=Exception("Dataflow error"),
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_control_flow_graph",
                return_value=mock_cfg,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_callers",
                return_value=mock_callers,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_callees",
                return_value=mock_callees,
            ),
        ):
            result = await agent.run(
                project_name="test",
                function_info=function_info,
            )

        # Should still succeed with empty dataflows
        assert result.success is True
        assert len(result.data.data_flows) == 0

    @pytest.mark.asyncio
    async def test_handle_cfg_error(self, agent, function_info):
        """Test handling of control flow error."""
        mock_dataflow = MagicMock()
        mock_dataflow.success = True
        mock_dataflow.flows = []

        mock_callers = MagicMock()
        mock_callers.success = True
        mock_callers.callers = []

        mock_callees = MagicMock()
        mock_callees.success = True
        mock_callees.callees = []

        with (
            patch(
                "fuzz_generator.agents.context_builder.track_dataflow",
                return_value=mock_dataflow,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_control_flow_graph",
                side_effect=Exception("CFG error"),
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_callers",
                return_value=mock_callers,
            ),
            patch(
                "fuzz_generator.agents.context_builder.get_callees",
                return_value=mock_callees,
            ),
        ):
            result = await agent.run(
                project_name="test",
                function_info=function_info,
            )

        # Should still succeed with default control flow
        assert result.success is True
        assert result.data.control_flow.has_loops is False
