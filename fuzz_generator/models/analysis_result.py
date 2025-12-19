"""Analysis result models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from fuzz_generator.models.function_info import FunctionInfo


class DataFlowNode(BaseModel):
    """Node in a data flow path."""

    code: str = Field(description="Code snippet")
    file: str = Field(description="Source file")
    line: int = Field(description="Line number")
    variable: str | None = Field(
        default=None,
        description="Variable name if applicable",
    )
    node_type: str = Field(
        default="unknown",
        description="Node type (source, sink, intermediate)",
    )


class DataFlowPath(BaseModel):
    """Data flow path from source to sink."""

    source: DataFlowNode = Field(description="Source node")
    sink: DataFlowNode = Field(description="Sink node")
    path_length: int = Field(
        default=0,
        description="Path length (number of steps)",
    )
    path_details: list[DataFlowNode] = Field(
        default_factory=list,
        description="Intermediate nodes in path",
    )
    taint_type: str | None = Field(
        default=None,
        description="Type of taint (e.g., user_input)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score",
    )


class ControlFlowInfo(BaseModel):
    """Control flow analysis information."""

    has_loops: bool = Field(
        default=False,
        description="Whether function contains loops",
    )
    has_conditions: bool = Field(
        default=False,
        description="Whether function contains conditionals",
    )
    has_recursion: bool = Field(
        default=False,
        description="Whether function is recursive",
    )
    branches: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Branch information",
    )
    loops: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Loop information",
    )
    complexity: int = Field(
        default=0,
        description="Cyclomatic complexity",
    )
    max_depth: int = Field(
        default=0,
        description="Maximum nesting depth",
    )


class CallerInfo(BaseModel):
    """Information about a function caller."""

    name: str = Field(description="Caller function name")
    file: str = Field(description="Source file")
    line: int = Field(description="Line number of call")
    context: str | None = Field(
        default=None,
        description="Call context/code snippet",
    )


class CalleeInfo(BaseModel):
    """Information about a called function."""

    name: str = Field(description="Called function name")
    file: str | None = Field(
        default=None,
        description="Source file if known",
    )
    is_external: bool = Field(
        default=False,
        description="Whether function is external/library",
    )
    call_count: int = Field(
        default=1,
        description="Number of times called",
    )


class CallGraphInfo(BaseModel):
    """Call graph analysis information."""

    callers: list[CallerInfo] = Field(
        default_factory=list,
        description="Functions that call this function",
    )
    callees: list[CalleeInfo] = Field(
        default_factory=list,
        description="Functions called by this function",
    )
    call_depth: int = Field(
        default=0,
        description="Depth in call hierarchy",
    )
    is_entry_point: bool = Field(
        default=False,
        description="Whether function is an entry point",
    )
    is_leaf: bool = Field(
        default=False,
        description="Whether function is a leaf (no callees)",
    )


class AnalysisContext(BaseModel):
    """Complete analysis context for model generation."""

    function_info: FunctionInfo = Field(description="Analyzed function information")
    data_flows: list[DataFlowPath] = Field(
        default_factory=list,
        description="Data flow paths",
    )
    control_flow: ControlFlowInfo = Field(
        default_factory=ControlFlowInfo,
        description="Control flow information",
    )
    call_graph: CallGraphInfo = Field(
        default_factory=CallGraphInfo,
        description="Call graph information",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="External dependencies",
    )
    related_functions: list[FunctionInfo] = Field(
        default_factory=list,
        description="Related function information",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Analysis timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @property
    def has_data_flows(self) -> bool:
        """Check if there are data flows."""
        return len(self.data_flows) > 0

    @property
    def has_tainted_flows(self) -> bool:
        """Check if there are tainted data flows."""
        return any(f.taint_type is not None for f in self.data_flows)


class GenerationResult(BaseModel):
    """Result of DataModel generation."""

    success: bool = Field(description="Whether generation succeeded")
    xml_content: str | None = Field(
        default=None,
        description="Generated XML content",
    )
    data_models: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Generated DataModel structures",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warning messages",
    )
    generation_time_seconds: float = Field(
        default=0.0,
        description="Generation time in seconds",
    )
    model_count: int = Field(
        default=0,
        description="Number of DataModels generated",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
