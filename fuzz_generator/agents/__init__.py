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

__all__ = [
    # Two-phase workflow
    "TwoPhaseWorkflow",
    "AnalysisWorkflowRunner",
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
]
