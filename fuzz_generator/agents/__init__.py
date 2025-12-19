"""Agent module for multi-agent collaboration.

This module provides AutoGen-based agents for code analysis workflow:
- CodeAnalyzer: Analyzes function code structure
- ContextBuilder: Builds data flow and control flow context
- ModelGenerator: Generates Secray XML DataModel
- Orchestrator: Coordinates the analysis workflow
"""

from fuzz_generator.agents.autogen_agents import (
    AnalysisWorkflowRunner,
    AutoGenAgentFactory,
    create_mcp_tools,
    run_single_agent_analysis,
)
from fuzz_generator.agents.base import AgentConfig, AgentResult, BaseAgent, PromptTemplate

# Legacy agents (for backwards compatibility)
from fuzz_generator.agents.code_analyzer import CodeAnalyzerAgent
from fuzz_generator.agents.context_builder import ContextBuilderAgent
from fuzz_generator.agents.model_generator import ModelGeneratorAgent
from fuzz_generator.agents.orchestrator import OrchestratorAgent

__all__ = [
    # AutoGen-based (new)
    "AutoGenAgentFactory",
    "AnalysisWorkflowRunner",
    "create_mcp_tools",
    "run_single_agent_analysis",
    # Base classes
    "BaseAgent",
    "AgentConfig",
    "AgentResult",
    "PromptTemplate",
    # Legacy agents
    "CodeAnalyzerAgent",
    "ContextBuilderAgent",
    "ModelGeneratorAgent",
    "OrchestratorAgent",
]
