"""Base agent class and common utilities.

This module provides the foundation for all agents in the system:
- AgentConfig: Configuration for agent initialization
- AgentResult: Standardized result structure
- PromptTemplate: Prompt template loading and rendering
- BaseAgent: Abstract base class defining the agent interface
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment

from fuzz_generator.config import Settings
from fuzz_generator.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """Configuration for agent initialization.

    Attributes:
        name: Agent name
        system_prompt: System prompt content
        max_iterations: Maximum iterations for multi-turn conversations
        tools: List of tool functions available to the agent
        llm_config: LLM configuration dict
        verbose: Whether to show detailed output
    """

    name: str
    system_prompt: str = ""
    max_iterations: int = 10
    tools: list[Callable[..., Any]] = field(default_factory=list)
    llm_config: dict[str, Any] = field(default_factory=dict)
    verbose: bool = True


@dataclass
class AgentResult:
    """Standardized result from agent execution.

    Attributes:
        success: Whether the execution was successful
        data: Result data (type depends on agent)
        messages: Conversation messages
        iterations: Number of iterations used
        error: Error message if any
    """

    success: bool
    data: Any = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "messages": self.messages,
            "iterations": self.iterations,
            "error": self.error,
        }


class PromptTemplate:
    """Prompt template loader and renderer.

    Supports YAML format prompt files with Jinja2 templating.
    """

    def __init__(self, template_path: str | Path | None = None):
        """Initialize prompt template.

        Args:
            template_path: Path to YAML template file
        """
        self.template_path = Path(template_path) if template_path else None
        self._template_data: dict[str, Any] = {}
        self._jinja_env = Environment()

        if self.template_path and self.template_path.exists():
            self._load_template()

    def _load_template(self) -> None:
        """Load template from YAML file."""
        try:
            with open(self.template_path, encoding="utf-8") as f:
                self._template_data = yaml.safe_load(f) or {}
            logger.debug(f"Loaded prompt template from: {self.template_path}")
        except Exception as e:
            logger.warning(f"Failed to load prompt template: {e}")
            self._template_data = {}

    @property
    def name(self) -> str:
        """Get template name."""
        return self._template_data.get("name", "UnknownAgent")

    @property
    def version(self) -> str:
        """Get template version."""
        return self._template_data.get("version", "1.0")

    @property
    def system_prompt(self) -> str:
        """Get raw system prompt."""
        return self._template_data.get("system_prompt", "")

    @property
    def task_template(self) -> str:
        """Get raw task template."""
        return self._template_data.get("task_template", "")

    @property
    def custom_knowledge_section(self) -> str:
        """Get custom knowledge section template."""
        return self._template_data.get("custom_knowledge_section", "")

    @property
    def examples(self) -> list[dict[str, Any]]:
        """Get example conversations."""
        return self._template_data.get("examples", [])

    def render_system_prompt(
        self,
        custom_knowledge: str = "",
        **kwargs: Any,
    ) -> str:
        """Render system prompt with variables.

        Args:
            custom_knowledge: Custom knowledge content to inject
            **kwargs: Additional template variables

        Returns:
            Rendered system prompt
        """
        prompt_parts = []

        # Main system prompt
        if self.system_prompt:
            template = self._jinja_env.from_string(self.system_prompt)
            prompt_parts.append(template.render(**kwargs))

        # Custom knowledge section
        if custom_knowledge and self.custom_knowledge_section:
            template = self._jinja_env.from_string(self.custom_knowledge_section)
            rendered = template.render(custom_knowledge=custom_knowledge, **kwargs)
            if rendered.strip():
                prompt_parts.append(rendered)

        return "\n\n".join(prompt_parts)

    def render_task(self, **kwargs: Any) -> str:
        """Render task template with variables.

        Args:
            **kwargs: Template variables

        Returns:
            Rendered task message
        """
        if not self.task_template:
            return ""

        template = self._jinja_env.from_string(self.task_template)
        return template.render(**kwargs)

    @classmethod
    def from_string(cls, system_prompt: str) -> "PromptTemplate":
        """Create template from string content.

        Args:
            system_prompt: System prompt content

        Returns:
            PromptTemplate instance
        """
        template = cls()
        template._template_data = {"system_prompt": system_prompt}
        return template

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        """Create template from dictionary.

        Args:
            data: Template data dictionary

        Returns:
            PromptTemplate instance
        """
        template = cls()
        template._template_data = data
        return template


class BaseAgent(ABC):
    """Abstract base class for all agents.

    This class defines the interface and common functionality for agents
    in the multi-agent system.
    """

    def __init__(
        self,
        config: AgentConfig,
        settings: Settings | None = None,
    ):
        """Initialize base agent.

        Args:
            config: Agent configuration
            settings: Application settings
        """
        self.config = config
        self.settings = settings
        self._tools: dict[str, Callable[..., Any]] = {}
        self._prompt_template: PromptTemplate | None = None

        # Register tools
        for tool in config.tools:
            self.register_tool(tool)

        logger.info(f"Initialized agent: {config.name}")

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.config.name

    @property
    def system_prompt(self) -> str:
        """Get system prompt."""
        return self.config.system_prompt

    @property
    def tools(self) -> dict[str, Callable[..., Any]]:
        """Get registered tools."""
        return self._tools

    def register_tool(self, tool: Callable[..., Any]) -> None:
        """Register a tool function.

        Args:
            tool: Tool function to register
        """
        tool_name = tool.__name__
        self._tools[tool_name] = tool
        logger.debug(f"Registered tool: {tool_name}")

    def unregister_tool(self, tool_name: str) -> None:
        """Unregister a tool function.

        Args:
            tool_name: Name of tool to unregister
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.debug(f"Unregistered tool: {tool_name}")

    def get_tool(self, tool_name: str) -> Callable[..., Any] | None:
        """Get a registered tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool function or None if not found
        """
        return self._tools.get(tool_name)

    def set_prompt_template(self, template: PromptTemplate) -> None:
        """Set the prompt template.

        Args:
            template: PromptTemplate instance
        """
        self._prompt_template = template
        logger.debug(f"Set prompt template: {template.name}")

    def load_prompt_template(self, template_path: str | Path) -> None:
        """Load prompt template from file.

        Args:
            template_path: Path to template file
        """
        self._prompt_template = PromptTemplate(template_path)
        logger.debug(f"Loaded prompt template from: {template_path}")

    def render_system_prompt(
        self,
        custom_knowledge: str = "",
        **kwargs: Any,
    ) -> str:
        """Render system prompt with template.

        Args:
            custom_knowledge: Custom knowledge to inject
            **kwargs: Additional template variables

        Returns:
            Rendered system prompt
        """
        if self._prompt_template:
            return self._prompt_template.render_system_prompt(
                custom_knowledge=custom_knowledge,
                **kwargs,
            )
        return self.config.system_prompt

    def render_task(self, **kwargs: Any) -> str:
        """Render task message with template.

        Args:
            **kwargs: Template variables

        Returns:
            Rendered task message
        """
        if self._prompt_template:
            return self._prompt_template.render_task(**kwargs)
        return ""

    @abstractmethod
    async def run(self, **kwargs: Any) -> AgentResult:
        """Execute the agent's main task.

        Args:
            **kwargs: Task-specific arguments

        Returns:
            AgentResult with execution results
        """
        pass

    def _create_result(
        self,
        success: bool,
        data: Any = None,
        messages: list[dict[str, Any]] | None = None,
        iterations: int = 0,
        error: str | None = None,
    ) -> AgentResult:
        """Create standardized result.

        Args:
            success: Whether execution was successful
            data: Result data
            messages: Conversation messages
            iterations: Number of iterations
            error: Error message if any

        Returns:
            AgentResult instance
        """
        return AgentResult(
            success=success,
            data=data,
            messages=messages or [],
            iterations=iterations,
            error=error,
        )

    def _parse_json_response(self, content: str) -> dict[str, Any] | None:
        """Parse JSON from response content.

        Handles cases where JSON is embedded in markdown code blocks.

        Args:
            content: Response content

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting from code blocks
        import re

        json_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        matches = re.findall(json_pattern, content, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        return None

    def _extract_xml_content(self, content: str) -> str | None:
        """Extract XML content from response.

        Args:
            content: Response content

        Returns:
            Extracted XML or None
        """
        import re

        # Try extracting from code blocks
        xml_pattern = r"```(?:xml)?\s*\n?(.*?)\n?```"
        matches = re.findall(xml_pattern, content, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Check if content itself is XML
        if content.strip().startswith("<?xml") or content.strip().startswith("<"):
            return content.strip()

        return None
