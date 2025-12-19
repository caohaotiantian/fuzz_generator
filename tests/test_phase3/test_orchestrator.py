"""Tests for Orchestrator agent."""

from unittest.mock import AsyncMock

import pytest

from fuzz_generator.agents.base import AgentConfig, AgentResult
from fuzz_generator.agents.orchestrator import OrchestratorAgent
from fuzz_generator.exceptions import AnalysisError
from fuzz_generator.models import (
    AnalysisContext,
    CallGraphInfo,
    ControlFlowInfo,
    FunctionInfo,
    GenerationResult,
    ParameterDirection,
    ParameterInfo,
    TaskResult,
)


class TestOrchestratorAgent:
    """Tests for OrchestratorAgent."""

    @pytest.fixture
    def mock_code_analyzer(self):
        """Create mock CodeAnalyzer."""
        analyzer = AsyncMock()
        analyzer.run = AsyncMock()
        return analyzer

    @pytest.fixture
    def mock_context_builder(self):
        """Create mock ContextBuilder."""
        builder = AsyncMock()
        builder.run = AsyncMock()
        return builder

    @pytest.fixture
    def mock_model_generator(self):
        """Create mock ModelGenerator."""
        generator = AsyncMock()
        generator.run = AsyncMock()
        return generator

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage backend."""
        storage = AsyncMock()
        storage.save = AsyncMock()
        return storage

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
                ),
            ],
        )

    @pytest.fixture
    def analysis_context(self, function_info):
        """Create sample analysis context."""
        return AnalysisContext(
            function_info=function_info,
            data_flows=[],
            control_flow=ControlFlowInfo(),
            call_graph=CallGraphInfo(),
        )

    @pytest.fixture
    def generation_result(self):
        """Create sample generation result."""
        return GenerationResult(
            success=True,
            xml_content="<DataModel>...</DataModel>",
            data_models=[{"name": "TestModel"}],
        )

    @pytest.fixture
    def orchestrator(
        self,
        mock_code_analyzer,
        mock_context_builder,
        mock_model_generator,
        mock_storage,
    ):
        """Create Orchestrator agent."""
        return OrchestratorAgent(
            code_analyzer=mock_code_analyzer,
            context_builder=mock_context_builder,
            model_generator=mock_model_generator,
            storage=mock_storage,
        )

    def test_agent_creation(self, orchestrator):
        """Test agent creation."""
        assert orchestrator.name == "Orchestrator"
        assert orchestrator.config.max_iterations == 50

    def test_agent_custom_config(
        self,
        mock_code_analyzer,
        mock_context_builder,
        mock_model_generator,
    ):
        """Test agent with custom config."""
        config = AgentConfig(
            name="CustomOrchestrator",
            max_iterations=100,
        )
        agent = OrchestratorAgent(
            code_analyzer=mock_code_analyzer,
            context_builder=mock_context_builder,
            model_generator=mock_model_generator,
            config=config,
        )
        assert agent.name == "CustomOrchestrator"

    @pytest.mark.asyncio
    async def test_run_complete_workflow(
        self,
        orchestrator,
        mock_code_analyzer,
        mock_context_builder,
        mock_model_generator,
        function_info,
        analysis_context,
        generation_result,
    ):
        """Test complete workflow execution."""
        # Setup mock returns
        mock_code_analyzer.run.return_value = AgentResult(
            success=True,
            data=function_info,
        )
        mock_context_builder.run.return_value = AgentResult(
            success=True,
            data=analysis_context,
        )
        mock_model_generator.run.return_value = AgentResult(
            success=True,
            data=generation_result,
        )

        result = await orchestrator.run(
            project_name="test_project",
            source_file="handler.c",
            function_name="process_request",
        )

        assert result.success is True
        assert isinstance(result.data, TaskResult)
        assert result.data.success is True

        # Verify all agents were called
        mock_code_analyzer.run.assert_called_once()
        mock_context_builder.run.assert_called_once()
        mock_model_generator.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_code_analysis_failure(
        self,
        orchestrator,
        mock_code_analyzer,
    ):
        """Test handling of code analysis failure."""
        mock_code_analyzer.run.return_value = AgentResult(
            success=False,
            error="Function not found",
        )

        result = await orchestrator.run(
            project_name="test_project",
            source_file="handler.c",
            function_name="nonexistent",
        )

        assert result.success is False
        assert "Code analysis failed" in result.error

    @pytest.mark.asyncio
    async def test_context_building_failure(
        self,
        orchestrator,
        mock_code_analyzer,
        mock_context_builder,
        function_info,
    ):
        """Test handling of context building failure."""
        mock_code_analyzer.run.return_value = AgentResult(
            success=True,
            data=function_info,
        )
        mock_context_builder.run.return_value = AgentResult(
            success=False,
            error="MCP error",
        )

        result = await orchestrator.run(
            project_name="test_project",
            source_file="handler.c",
            function_name="process_request",
        )

        assert result.success is False
        assert "Context building failed" in result.error

    @pytest.mark.asyncio
    async def test_model_generation_failure(
        self,
        orchestrator,
        mock_code_analyzer,
        mock_context_builder,
        mock_model_generator,
        function_info,
        analysis_context,
    ):
        """Test handling of model generation failure."""
        mock_code_analyzer.run.return_value = AgentResult(
            success=True,
            data=function_info,
        )
        mock_context_builder.run.return_value = AgentResult(
            success=True,
            data=analysis_context,
        )
        mock_model_generator.run.return_value = AgentResult(
            success=False,
            error="Generation failed",
        )

        result = await orchestrator.run(
            project_name="test_project",
            source_file="handler.c",
            function_name="process_request",
        )

        assert result.success is False
        assert "Model generation failed" in result.error

    @pytest.mark.asyncio
    async def test_analyze_function_convenience_method(
        self,
        orchestrator,
        mock_code_analyzer,
        mock_context_builder,
        mock_model_generator,
        function_info,
        analysis_context,
        generation_result,
    ):
        """Test analyze_function() convenience method."""
        mock_code_analyzer.run.return_value = AgentResult(
            success=True,
            data=function_info,
        )
        mock_context_builder.run.return_value = AgentResult(
            success=True,
            data=analysis_context,
        )
        mock_model_generator.run.return_value = AgentResult(
            success=True,
            data=generation_result,
        )

        task_result = await orchestrator.analyze_function(
            project_name="test",
            source_file="test.c",
            function_name="test_func",
        )

        assert isinstance(task_result, TaskResult)

    @pytest.mark.asyncio
    async def test_analyze_function_raises_on_failure(
        self,
        orchestrator,
        mock_code_analyzer,
    ):
        """Test that analyze_function() raises on failure."""
        mock_code_analyzer.run.return_value = AgentResult(
            success=False,
            error="Error",
        )

        with pytest.raises(AnalysisError):
            await orchestrator.analyze_function(
                project_name="test",
                source_file="test.c",
                function_name="test_func",
            )

    @pytest.mark.asyncio
    async def test_intermediate_results_saved(
        self,
        orchestrator,
        mock_code_analyzer,
        mock_context_builder,
        mock_model_generator,
        mock_storage,
        function_info,
        analysis_context,
        generation_result,
    ):
        """Test that intermediate results are saved."""
        mock_code_analyzer.run.return_value = AgentResult(
            success=True,
            data=function_info,
        )
        mock_context_builder.run.return_value = AgentResult(
            success=True,
            data=analysis_context,
        )
        mock_model_generator.run.return_value = AgentResult(
            success=True,
            data=generation_result,
        )

        await orchestrator.run(
            project_name="test_project",
            source_file="handler.c",
            function_name="process_request",
        )

        # Should save intermediate results for each phase + final
        assert mock_storage.save.call_count >= 3

    @pytest.mark.asyncio
    async def test_custom_task_id(
        self,
        orchestrator,
        mock_code_analyzer,
        mock_context_builder,
        mock_model_generator,
        function_info,
        analysis_context,
        generation_result,
    ):
        """Test custom task ID."""
        mock_code_analyzer.run.return_value = AgentResult(
            success=True,
            data=function_info,
        )
        mock_context_builder.run.return_value = AgentResult(
            success=True,
            data=analysis_context,
        )
        mock_model_generator.run.return_value = AgentResult(
            success=True,
            data=generation_result,
        )

        result = await orchestrator.run(
            project_name="test",
            source_file="test.c",
            function_name="func",
            task_id="custom_task_123",
        )

        assert result.data.task_id == "custom_task_123"

    @pytest.mark.asyncio
    async def test_custom_knowledge_passed(
        self,
        orchestrator,
        mock_code_analyzer,
        mock_context_builder,
        mock_model_generator,
        function_info,
        analysis_context,
        generation_result,
    ):
        """Test that custom knowledge is passed through."""
        mock_code_analyzer.run.return_value = AgentResult(
            success=True,
            data=function_info,
        )
        mock_context_builder.run.return_value = AgentResult(
            success=True,
            data=analysis_context,
        )
        mock_model_generator.run.return_value = AgentResult(
            success=True,
            data=generation_result,
        )

        await orchestrator.run(
            project_name="test",
            source_file="test.c",
            function_name="func",
            custom_knowledge="Special rules",
        )

        # Verify custom_knowledge was passed to code_analyzer
        call_kwargs = mock_code_analyzer.run.call_args[1]
        assert call_kwargs.get("custom_knowledge") == "Special rules"

    def test_get_intermediate_result(
        self,
        orchestrator,
    ):
        """Test getting intermediate results."""
        orchestrator._intermediate_results["code_analysis"] = {"test": "data"}

        result = orchestrator.get_intermediate_result("code_analysis")
        assert result == {"test": "data"}

        # Nonexistent stage
        result = orchestrator.get_intermediate_result("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_storage_error_handled(
        self,
        orchestrator,
        mock_code_analyzer,
        mock_context_builder,
        mock_model_generator,
        mock_storage,
        function_info,
        analysis_context,
        generation_result,
    ):
        """Test that storage errors don't break workflow."""
        mock_code_analyzer.run.return_value = AgentResult(
            success=True,
            data=function_info,
        )
        mock_context_builder.run.return_value = AgentResult(
            success=True,
            data=analysis_context,
        )
        mock_model_generator.run.return_value = AgentResult(
            success=True,
            data=generation_result,
        )
        mock_storage.save.side_effect = Exception("Storage error")

        # Should still complete successfully
        result = await orchestrator.run(
            project_name="test",
            source_file="test.c",
            function_name="func",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_without_storage(
        self,
        mock_code_analyzer,
        mock_context_builder,
        mock_model_generator,
        function_info,
        analysis_context,
        generation_result,
    ):
        """Test workflow without storage backend."""
        orchestrator = OrchestratorAgent(
            code_analyzer=mock_code_analyzer,
            context_builder=mock_context_builder,
            model_generator=mock_model_generator,
            storage=None,
        )

        mock_code_analyzer.run.return_value = AgentResult(
            success=True,
            data=function_info,
        )
        mock_context_builder.run.return_value = AgentResult(
            success=True,
            data=analysis_context,
        )
        mock_model_generator.run.return_value = AgentResult(
            success=True,
            data=generation_result,
        )

        result = await orchestrator.run(
            project_name="test",
            source_file="test.c",
            function_name="func",
        )

        assert result.success is True
