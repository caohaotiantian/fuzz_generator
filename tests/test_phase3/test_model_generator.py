"""Tests for ModelGenerator agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from fuzz_generator.agents.base import AgentConfig
from fuzz_generator.agents.model_generator import ModelGeneratorAgent
from fuzz_generator.models import (
    AnalysisContext,
    CallGraphInfo,
    ControlFlowInfo,
    FunctionInfo,
    GenerationResult,
    ParameterDirection,
    ParameterInfo,
)


class TestModelGeneratorAgent:
    """Tests for ModelGeneratorAgent."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        return client

    @pytest.fixture
    def agent(self):
        """Create ModelGenerator agent without LLM."""
        return ModelGeneratorAgent()

    @pytest.fixture
    def agent_with_llm(self, mock_llm_client):
        """Create ModelGenerator agent with LLM."""
        return ModelGeneratorAgent(llm_client=mock_llm_client)

    @pytest.fixture
    def analysis_context(self):
        """Create sample analysis context."""
        function_info = FunctionInfo(
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

        return AnalysisContext(
            function_info=function_info,
            data_flows=[],
            control_flow=ControlFlowInfo(
                has_loops=False,
                has_conditions=True,
                branches=[],
                complexity=5,
            ),
            call_graph=CallGraphInfo(
                callers=[],
                callees=[],
                depth=1,
            ),
        )

    def test_agent_creation(self, agent):
        """Test agent creation."""
        assert agent.name == "ModelGenerator"
        assert agent.config.max_iterations == 5

    def test_agent_custom_config(self):
        """Test agent with custom config."""
        config = AgentConfig(
            name="CustomGenerator",
            max_iterations=10,
        )
        agent = ModelGeneratorAgent(config=config)
        assert agent.name == "CustomGenerator"

    @pytest.mark.asyncio
    async def test_generate_without_llm(self, agent, analysis_context):
        """Test generation without LLM (basic template)."""
        result = await agent.run(context=analysis_context)

        assert result.success is True
        assert isinstance(result.data, GenerationResult)
        assert result.data.xml_content is not None
        assert "<DataModel" in result.data.xml_content
        assert "ProcessRequestModel" in result.data.xml_content

    @pytest.mark.asyncio
    async def test_generate_with_llm(self, agent_with_llm, mock_llm_client, analysis_context):
        """Test generation with LLM."""
        # Mock LLM response
        llm_response = MagicMock()
        llm_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="""
```xml
<?xml version="1.0" encoding="utf-8"?>
<Secray>
    <DataModel name="ProcessRequestModel">
        <String name="buffer" />
        <Number name="length" size="32" signed="true" />
    </DataModel>
</Secray>
```
                    """
                )
            )
        ]
        mock_llm_client.chat.completions.create.return_value = llm_response

        result = await agent_with_llm.run(context=analysis_context)

        assert result.success is True
        assert "<DataModel" in result.data.xml_content
        assert "ProcessRequestModel" in result.data.xml_content

    @pytest.mark.asyncio
    async def test_generate_with_custom_name(self, agent, analysis_context):
        """Test generation with custom output name."""
        result = await agent.run(
            context=analysis_context,
            output_name="CustomModel",
        )

        assert result.success is True
        assert "CustomModel" in result.data.xml_content

    @pytest.mark.asyncio
    async def test_generate_with_custom_knowledge(
        self, agent_with_llm, mock_llm_client, analysis_context
    ):
        """Test generation with custom knowledge."""
        llm_response = MagicMock()
        llm_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='<DataModel name="TestModel"><String name="test" /></DataModel>'
                )
            )
        ]
        mock_llm_client.chat.completions.create.return_value = llm_response

        result = await agent_with_llm.run(
            context=analysis_context,
            custom_knowledge="Special modeling rules",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_generate_convenience_method(self, agent, analysis_context):
        """Test generate() convenience method."""
        gen_result = await agent.generate(context=analysis_context)

        assert isinstance(gen_result, GenerationResult)
        assert gen_result.xml_content is not None

    @pytest.mark.asyncio
    async def test_generate_raises_on_failure(self, agent, analysis_context):
        """Test that generate() falls back to basic template on failure."""
        # Agent without LLM should fall back to basic template
        gen_result = await agent.generate(context=analysis_context)
        assert gen_result is not None

    def test_basic_template_with_string_param(self, agent, analysis_context):
        """Test basic template generates String for char*."""
        xml = agent._generate_basic_template(analysis_context)
        assert '<String name="buffer"' in xml

    def test_basic_template_with_number_param(self, agent, analysis_context):
        """Test basic template generates Number for int."""
        xml = agent._generate_basic_template(analysis_context)
        assert '<Number name="length"' in xml

    def test_basic_template_with_pointer_param(self):
        """Test basic template generates Blob for other pointers."""
        agent = ModelGeneratorAgent()

        function_info = FunctionInfo(
            name="test_func",
            file_path="test.c",
            line_number=1,
            return_type="void",
            parameters=[
                ParameterInfo(
                    name="data",
                    type="void*",
                    direction=ParameterDirection.IN,
                ),
            ],
        )

        context = AnalysisContext(
            function_info=function_info,
            data_flows=[],
            control_flow=ControlFlowInfo(),
            call_graph=CallGraphInfo(),
        )

        xml = agent._generate_basic_template(context)
        assert '<Blob name="data"' in xml

    def test_extract_datamodel_info(self, agent):
        """Test extracting DataModel info from XML."""
        xml = """
        <Secray>
            <DataModel name="Model1">...</DataModel>
            <DataModel name="Model2">...</DataModel>
        </Secray>
        """

        models = agent._extract_datamodel_info(xml)

        assert len(models) == 2
        assert models[0]["name"] == "Model1"
        assert models[1]["name"] == "Model2"

    @pytest.mark.asyncio
    async def test_llm_response_without_code_block(
        self, agent_with_llm, mock_llm_client, analysis_context
    ):
        """Test handling LLM response without code block."""
        # LLM returns XML directly without code block
        llm_response = MagicMock()
        llm_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='<?xml version="1.0"?><Secray><DataModel name="Test"/></Secray>'
                )
            )
        ]
        mock_llm_client.chat.completions.create.return_value = llm_response

        result = await agent_with_llm.run(context=analysis_context)

        assert result.success is True
        assert "DataModel" in result.data.xml_content

    @pytest.mark.asyncio
    async def test_generation_result_structure(self, agent, analysis_context):
        """Test GenerationResult structure."""
        result = await agent.run(context=analysis_context)

        assert result.success is True
        gen_result = result.data

        assert gen_result.success is True
        assert gen_result.xml_content is not None
        assert isinstance(gen_result.data_models, list)
