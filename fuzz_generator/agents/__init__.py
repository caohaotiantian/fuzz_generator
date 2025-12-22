"""Agent module for two-phase analysis workflow.

This module provides AutoGen-based agents for code analysis:
- AnalysisAgent: Combined code analysis + context building with iterative tool calls
- ModelGenerator: Generates Secray XML DataModel based on analysis

Design Reference: docs/AGENT_OPTIMIZATION.md
"""

from fuzz_generator.agents.autogen_agents import (
    AnalysisWorkflowRunner,
    ConversationRecorder,
    OutputValidator,
    PromptLoader,
    TwoPhaseWorkflow,
    create_analysis_tools,
    run_single_agent_analysis,
)
from fuzz_generator.agents.base import AgentConfig, AgentResult, BaseAgent, PromptTemplate

# Legacy agents (for backwards compatibility, may be deprecated)
from fuzz_generator.agents.code_analyzer import CodeAnalyzerAgent
from fuzz_generator.agents.context_builder import ContextBuilderAgent
from fuzz_generator.agents.model_generator import ModelGeneratorAgent
from fuzz_generator.agents.orchestrator import OrchestratorAgent

__all__ = [
    # Two-phase workflow (new design)
    "TwoPhaseWorkflow",
    "AnalysisWorkflowRunner",  # Alias for backward compatibility
    "create_analysis_tools",
    "run_single_agent_analysis",
    "PromptLoader",
    "OutputValidator",
    "ConversationRecorder",
    # Base classes
    "BaseAgent",
    "AgentConfig",
    "AgentResult",
    "PromptTemplate",
    # Legacy agents (deprecated)
    "CodeAnalyzerAgent",
    "ContextBuilderAgent",
    "ModelGeneratorAgent",
    "OrchestratorAgent",
]
