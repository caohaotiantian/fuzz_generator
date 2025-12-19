"""Tests for base agent classes."""

import pytest

from fuzz_generator.agents.base import (
    AgentConfig,
    AgentResult,
    BaseAgent,
    PromptTemplate,
)


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AgentConfig(name="TestAgent")
        assert config.name == "TestAgent"
        assert config.system_prompt == ""
        assert config.max_iterations == 10
        assert config.tools == []
        assert config.llm_config == {}
        assert config.verbose is True

    def test_custom_values(self):
        """Test custom configuration values."""

        def tool_func():
            pass

        config = AgentConfig(
            name="CustomAgent",
            system_prompt="Custom prompt",
            max_iterations=20,
            tools=[tool_func],
            llm_config={"model": "gpt-4"},
            verbose=False,
        )
        assert config.name == "CustomAgent"
        assert config.system_prompt == "Custom prompt"
        assert config.max_iterations == 20
        assert len(config.tools) == 1
        assert config.llm_config == {"model": "gpt-4"}
        assert config.verbose is False


class TestAgentResult:
    """Tests for AgentResult."""

    def test_success_result(self):
        """Test successful result."""
        result = AgentResult(
            success=True,
            data={"key": "value"},
            messages=[{"role": "user", "content": "test"}],
            iterations=3,
        )
        assert result.success is True
        assert result.data == {"key": "value"}
        assert len(result.messages) == 1
        assert result.iterations == 3
        assert result.error is None

    def test_error_result(self):
        """Test error result."""
        result = AgentResult(
            success=False,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_to_dict(self):
        """Test result serialization."""
        result = AgentResult(
            success=True,
            data="test",
            iterations=1,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["data"] == "test"
        assert d["iterations"] == 1


class TestPromptTemplate:
    """Tests for PromptTemplate."""

    def test_empty_template(self):
        """Test empty template."""
        template = PromptTemplate()
        assert template.name == "UnknownAgent"
        assert template.version == "1.0"
        assert template.system_prompt == ""

    def test_from_string(self):
        """Test creating template from string."""
        template = PromptTemplate.from_string("Test prompt content")
        assert template.system_prompt == "Test prompt content"

    def test_from_dict(self):
        """Test creating template from dictionary."""
        data = {
            "name": "TestAgent",
            "version": "2.0",
            "system_prompt": "Test system prompt",
            "task_template": "Task: {{ task }}",
        }
        template = PromptTemplate.from_dict(data)
        assert template.name == "TestAgent"
        assert template.version == "2.0"
        assert template.system_prompt == "Test system prompt"

    def test_render_system_prompt(self):
        """Test rendering system prompt."""
        template = PromptTemplate.from_dict(
            {
                "system_prompt": "Hello {{ name }}!",
            }
        )
        rendered = template.render_system_prompt(name="World")
        assert "Hello World!" in rendered

    def test_render_system_prompt_with_knowledge(self):
        """Test rendering with custom knowledge."""
        template = PromptTemplate.from_dict(
            {
                "system_prompt": "Base prompt",
                "custom_knowledge_section": "{% if custom_knowledge %}Knowledge: {{ custom_knowledge }}{% endif %}",
            }
        )
        rendered = template.render_system_prompt(custom_knowledge="Custom info")
        assert "Base prompt" in rendered
        assert "Knowledge: Custom info" in rendered

    def test_render_task(self):
        """Test rendering task template."""
        template = PromptTemplate.from_dict(
            {
                "task_template": "Analyze function: {{ function_name }}",
            }
        )
        rendered = template.render_task(function_name="main")
        assert "Analyze function: main" in rendered

    def test_load_from_file(self, tmp_path):
        """Test loading template from file."""
        template_file = tmp_path / "test_prompt.yaml"
        template_file.write_text("""
name: FileAgent
version: "1.0"
system_prompt: "Loaded from file"
        """)

        template = PromptTemplate(template_file)
        assert template.name == "FileAgent"
        assert template.system_prompt == "Loaded from file"

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from nonexistent file."""
        template = PromptTemplate(tmp_path / "nonexistent.yaml")
        assert template.name == "UnknownAgent"


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing."""

    async def run(self, **kwargs):
        return self._create_result(success=True, data="test")


class TestBaseAgent:
    """Tests for BaseAgent."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AgentConfig(
            name="TestAgent",
            system_prompt="Test prompt",
            max_iterations=5,
        )

    def test_agent_creation(self, config):
        """Test agent creation."""
        agent = ConcreteAgent(config)
        assert agent.name == "TestAgent"
        assert agent.system_prompt == "Test prompt"

    def test_tool_registration(self, config):
        """Test tool registration."""

        def my_tool(x: int) -> int:
            return x * 2

        agent = ConcreteAgent(config)
        agent.register_tool(my_tool)

        assert "my_tool" in agent.tools
        assert agent.get_tool("my_tool") == my_tool

    def test_tool_unregistration(self, config):
        """Test tool unregistration."""

        def my_tool():
            pass

        agent = ConcreteAgent(config)
        agent.register_tool(my_tool)
        agent.unregister_tool("my_tool")

        assert "my_tool" not in agent.tools

    def test_get_nonexistent_tool(self, config):
        """Test getting nonexistent tool."""
        agent = ConcreteAgent(config)
        assert agent.get_tool("nonexistent") is None

    def test_set_prompt_template(self, config):
        """Test setting prompt template."""
        agent = ConcreteAgent(config)
        template = PromptTemplate.from_string("Template prompt")
        agent.set_prompt_template(template)

        rendered = agent.render_system_prompt()
        assert "Template prompt" in rendered

    def test_render_system_prompt_without_template(self, config):
        """Test rendering without template."""
        agent = ConcreteAgent(config)
        rendered = agent.render_system_prompt()
        assert rendered == "Test prompt"

    @pytest.mark.asyncio
    async def test_run(self, config):
        """Test running agent."""
        agent = ConcreteAgent(config)
        result = await agent.run()

        assert result.success is True
        assert result.data == "test"

    def test_parse_json_response(self, config):
        """Test JSON parsing from response."""
        agent = ConcreteAgent(config)

        # Direct JSON
        parsed = agent._parse_json_response('{"key": "value"}')
        assert parsed == {"key": "value"}

        # JSON in code block
        parsed = agent._parse_json_response('```json\n{"key": "value"}\n```')
        assert parsed == {"key": "value"}

        # Invalid JSON
        parsed = agent._parse_json_response("not json")
        assert parsed is None

    def test_extract_xml_content(self, config):
        """Test XML extraction from response."""
        agent = ConcreteAgent(config)

        # XML in code block
        xml = agent._extract_xml_content("```xml\n<root>content</root>\n```")
        assert xml == "<root>content</root>"

        # Direct XML
        xml = agent._extract_xml_content("<?xml version='1.0'?><root/>")
        assert xml == "<?xml version='1.0'?><root/>"

        # No XML
        xml = agent._extract_xml_content("plain text")
        assert xml is None
