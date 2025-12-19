"""Agents module for multi-agent collaboration.

This module provides the agent system for fuzz test data modeling:
- BaseAgent: Abstract base class for all agents
- CodeAnalyzerAgent: Analyzes function code structure and parameters
- ContextBuilderAgent: Builds context via dataflow/controlflow analysis
- ModelGeneratorAgent: Generates Secray XML DataModel
- OrchestratorAgent: Coordinates the analysis workflow
"""

from fuzz_generator.agents.base import (
    AgentConfig,
    AgentResult,
    BaseAgent,
    PromptTemplate,
)
from fuzz_generator.agents.code_analyzer import CodeAnalyzerAgent
from fuzz_generator.agents.context_builder import ContextBuilderAgent
from fuzz_generator.agents.model_generator import ModelGeneratorAgent
from fuzz_generator.agents.orchestrator import OrchestratorAgent

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentConfig",
    "AgentResult",
    "PromptTemplate",
    # Specialized agents
    "CodeAnalyzerAgent",
    "ContextBuilderAgent",
    "ModelGeneratorAgent",
    "OrchestratorAgent",
]
