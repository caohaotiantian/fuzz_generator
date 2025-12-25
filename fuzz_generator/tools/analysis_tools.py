"""Analysis tools for Joern MCP Server.

This module provides wrapper functions for code analysis MCP tools:
- track_dataflow: Track data flow between source and sink
- get_callers: Get functions that call a specific function
- get_callees: Get functions called by a specific function
- get_control_flow_graph: Get CFG for a function
- analyze_variable_flow: Analyze variable data flow
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fuzz_generator.exceptions import MCPToolError
from fuzz_generator.utils.logger import get_logger

if TYPE_CHECKING:
    from fuzz_generator.tools.mcp_client import MCPHttpClient

logger = get_logger(__name__)


def _ensure_dict_response(data: Any, tool_name: str) -> dict | None:
    """Ensure MCP tool response is a dictionary.

    Args:
        data: Response data from MCP tool
        tool_name: Name of the tool (for error logging)

    Returns:
        Parsed dictionary or None if parsing fails
    """
    logger.debug(f"[{tool_name}] 原始 data 类型: {type(data)}")
    logger.debug(f"[{tool_name}] 原始 data 内容: {str(data)}")

    # Handle case where data might be a string (needs JSON parsing)
    if isinstance(data, str):
        try:
            data = json.loads(data)
            logger.debug(f"[{tool_name}] JSON 解析成功，解析后类型: {type(data)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {tool_name} response as JSON: {e}")
            logger.error(f"Raw data: {data}")
            return None

    # Ensure data is a dictionary
    if not isinstance(data, dict):
        logger.error(f"Unexpected {tool_name} response type: {type(data)}")
        logger.error(f"Data content: {data}")
        return None

    logger.debug(f"[{tool_name}] 最终 data keys: {list(data.keys())}")
    return data


@dataclass
class FlowInfo:
    """Information about a data flow path.

    Attributes:
        source: Source node information
        sink: Sink node information
        path_length: Number of nodes in the path
        path_details: Detailed path information
    """

    source: dict[str, Any]
    sink: dict[str, Any]
    path_length: int = 0
    path_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DataFlowResult:
    """Result of data flow tracking.

    Attributes:
        success: Whether the operation was successful
        flows: List of data flow paths
        count: Total number of flows found
        error: Error message if any
    """

    success: bool
    flows: list[FlowInfo] = field(default_factory=list)
    count: int = 0
    error: str | None = None


@dataclass
class CallerInfo:
    """Information about a function caller.

    Attributes:
        name: Caller function name
        filename: Source file containing the caller
        line_number: Line number of the call site
        code: Code snippet of the call
    """

    name: str
    filename: str
    line_number: int = 0
    code: str = ""


@dataclass
class CallersResult:
    """Result of get_callers operation.

    Attributes:
        success: Whether the operation was successful
        function: Target function name
        callers: List of caller information
        count: Total number of callers
        error: Error message if any
    """

    success: bool
    function: str = ""
    callers: list[CallerInfo] = field(default_factory=list)
    count: int = 0
    error: str | None = None


@dataclass
class CalleeInfo:
    """Information about a function callee.

    Attributes:
        name: Callee function name
        filename: Source file containing the callee
        line_number: Line number in the caller where call occurs
    """

    name: str
    filename: str = ""
    line_number: int = 0


@dataclass
class CalleesResult:
    """Result of get_callees operation.

    Attributes:
        success: Whether the operation was successful
        function: Source function name
        callees: List of callee information
        count: Total number of callees
        error: Error message if any
    """

    success: bool
    function: str = ""
    callees: list[CalleeInfo] = field(default_factory=list)
    count: int = 0
    error: str | None = None


@dataclass
class ControlFlowNode:
    """Node in a control flow graph.

    Attributes:
        id: Node identifier
        code: Code at this node
        node_type: Type of control flow node
        line_number: Line number in source
    """

    id: str
    code: str = ""
    node_type: str = ""
    line_number: int = 0


@dataclass
class ControlFlowEdge:
    """Edge in a control flow graph.

    Attributes:
        from_node: Source node ID
        to_node: Target node ID
        edge_type: Type of edge (e.g., "true", "false", "default")
    """

    from_node: str
    to_node: str
    edge_type: str = ""


@dataclass
class ControlFlowResult:
    """Result of control flow graph query.

    Attributes:
        success: Whether the operation was successful
        function: Function name
        nodes: List of CFG nodes
        edges: List of CFG edges
        has_loops: Whether the function contains loops
        has_conditions: Whether the function contains conditionals
        error: Error message if any
    """

    success: bool
    function: str = ""
    nodes: list[ControlFlowNode] = field(default_factory=list)
    edges: list[ControlFlowEdge] = field(default_factory=list)
    has_loops: bool = False
    has_conditions: bool = False
    error: str | None = None


@dataclass
class VariableFlowResult:
    """Result of variable flow analysis.

    Attributes:
        success: Whether the operation was successful
        variable: Variable name
        definitions: Where the variable is defined
        usages: Where the variable is used
        error: Error message if any
    """

    success: bool
    variable: str = ""
    definitions: list[dict[str, Any]] = field(default_factory=list)
    usages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


async def track_dataflow(
    client: "MCPHttpClient",
    project_name: str,
    source_method: str | None = None,
    sink_method: str | None = None,
) -> DataFlowResult:
    """Track data flow between source and sink.

    Args:
        client: MCP HTTP client instance
        project_name: Name of the project
        source_method: Source method name (e.g., "gets", "scanf")
        sink_method: Sink method name (e.g., "system", "strcpy")
        source_pattern: Source code pattern (alternative to source_method)
        sink_pattern: Sink code pattern (alternative to sink_method)

    Returns:
        DataFlowResult with flow paths

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Tracking dataflow in project: {project_name}")
    logger.debug(f"Source: {source_method}")
    logger.debug(f"Sink: {sink_method}")

    arguments: dict[str, Any] = {"project_name": project_name}

    if source_method:
        arguments["source_method"] = source_method
    if sink_method:
        arguments["sink_method"] = sink_method

    try:
        result = await client.call_tool("track_dataflow", arguments)

        if not result.success:
            return DataFlowResult(
                success=False,
                error=result.error or "Failed to track dataflow",
            )

        data = _ensure_dict_response(result.data, "track_dataflow")
        if data is None:
            return DataFlowResult(
                success=False,
                error="Invalid response format from MCP server",
            )

        flows_data = data.get("flows", [])

        flows = []
        for flow in flows_data:
            flows.append(
                FlowInfo(
                    source=flow.get("source", {}),
                    sink=flow.get("sink", {}),
                    path_length=flow.get("pathLength", 0),
                    path_details=flow.get("pathDetails", []),
                )
            )

        return DataFlowResult(
            success=data.get("success", True),
            flows=flows,
            count=data.get("count", len(flows)),
            error=data.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to track dataflow: {e}")
        raise MCPToolError(
            f"Failed to track dataflow: {e}",
            details={"project_name": project_name},
        ) from e


async def get_callers(
    client: "MCPHttpClient",
    project_name: str,
    function_name: str,
) -> CallersResult:
    """Get functions that call a specific function.

    Args:
        client: MCP HTTP client instance
        project_name: Name of the project
        function_name: Name of the target function

    Returns:
        CallersResult with caller information

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Getting callers of function: {function_name}")

    try:
        result = await client.call_tool(
            "get_callers",
            {
                "project_name": project_name,
                "function_name": function_name,
            },
        )

        if not result.success:
            return CallersResult(
                success=False,
                function=function_name,
                error=result.error or "Failed to get callers",
            )

        data = _ensure_dict_response(result.data, "get_callers")
        if data is None:
            return CallersResult(
                success=False,
                function=function_name,
                error="Invalid response format from MCP server",
            )

        callers_data = data.get("callers", [])

        callers = []
        for caller in callers_data:
            callers.append(
                CallerInfo(
                    name=caller.get("name", ""),
                    filename=caller.get("filename", ""),
                    line_number=caller.get("lineNumber", 0),
                    code=caller.get("code", ""),
                )
            )

        return CallersResult(
            success=data.get("success", True),
            function=data.get("function", function_name),
            callers=callers,
            count=data.get("count", len(callers)),
            error=data.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to get callers: {e}")
        raise MCPToolError(
            f"Failed to get callers: {e}",
            details={"function_name": function_name},
        ) from e


async def get_callees(
    client: "MCPHttpClient",
    project_name: str,
    function_name: str,
) -> CalleesResult:
    """Get functions called by a specific function.

    Args:
        client: MCP HTTP client instance
        project_name: Name of the project
        function_name: Name of the source function

    Returns:
        CalleesResult with callee information

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Getting callees of function: {function_name}")

    try:
        result = await client.call_tool(
            "get_callees",
            {
                "project_name": project_name,
                "function_name": function_name,
            },
        )

        if not result.success:
            return CalleesResult(
                success=False,
                function=function_name,
                error=result.error or "Failed to get callees",
            )

        data = _ensure_dict_response(result.data, "get_callees")
        if data is None:
            return CalleesResult(
                success=False,
                function=function_name,
                error="Invalid response format from MCP server",
            )

        callees_data = data.get("callees", [])
        logger.debug(
            f"[get_callees] callees_data 类型: {type(callees_data)}, 长度: {len(callees_data) if isinstance(callees_data, list) else 'N/A'}"
        )

        callees = []
        for i, callee in enumerate(callees_data):
            logger.debug(f"[get_callees] callee[{i}] 类型: {type(callee)}, 内容: {callee}")
            if not isinstance(callee, dict):
                logger.error(f"[get_callees] callee[{i}] 不是字典: {type(callee)}")
                continue
            callees.append(
                CalleeInfo(
                    name=callee.get("name", ""),
                    filename=callee.get("filename", ""),
                    line_number=callee.get("lineNumber", 0),
                )
            )

        return CalleesResult(
            success=data.get("success", True),
            function=data.get("function", function_name),
            callees=callees,
            count=data.get("count", len(callees)),
            error=data.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to get callees: {e}")
        raise MCPToolError(
            f"Failed to get callees: {e}",
            details={"function_name": function_name},
        ) from e


async def get_control_flow_graph(
    client: "MCPHttpClient",
    project_name: str,
    function_name: str,
) -> ControlFlowResult:
    """Get control flow graph for a function.

    Args:
        client: MCP HTTP client instance
        project_name: Name of the project
        function_name: Name of the function

    Returns:
        ControlFlowResult with CFG information

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Getting CFG for function: {function_name}")

    try:
        result = await client.call_tool(
            "get_control_flow_graph",
            {
                "project_name": project_name,
                "function_name": function_name,
            },
        )

        if not result.success:
            return ControlFlowResult(
                success=False,
                function=function_name,
                error=result.error or "Failed to get CFG",
            )

        data = _ensure_dict_response(result.data, "get_control_flow_graph")
        if data is None:
            return ControlFlowResult(
                success=False,
                function=function_name,
                error="Invalid response format from MCP server",
            )

        # Parse nodes
        nodes_data = data.get("nodes", [])
        nodes = []
        for node in nodes_data:
            nodes.append(
                ControlFlowNode(
                    id=node.get("id", ""),
                    code=node.get("code", ""),
                    node_type=node.get("type", ""),
                    line_number=node.get("lineNumber", 0),
                )
            )

        # Parse edges
        edges_data = data.get("edges", [])
        edges = []
        for edge in edges_data:
            edges.append(
                ControlFlowEdge(
                    from_node=edge.get("from", ""),
                    to_node=edge.get("to", ""),
                    edge_type=edge.get("type", ""),
                )
            )

        return ControlFlowResult(
            success=data.get("success", True),
            function=data.get("function", function_name),
            nodes=nodes,
            edges=edges,
            has_loops=data.get("hasLoops", False),
            has_conditions=data.get("hasConditions", False),
            error=data.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to get CFG: {e}")
        raise MCPToolError(
            f"Failed to get control flow graph: {e}",
            details={"function_name": function_name},
        ) from e


async def analyze_variable_flow(
    client: "MCPHttpClient",
    project_name: str,
    function_name: str,
    variable_name: str,
) -> VariableFlowResult:
    """Analyze data flow for a specific variable.

    Args:
        client: MCP HTTP client instance
        project_name: Name of the project
        function_name: Name of the function containing the variable
        variable_name: Name of the variable to analyze

    Returns:
        VariableFlowResult with variable flow information

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Analyzing variable flow: {variable_name} in {function_name}")

    try:
        result = await client.call_tool(
            "analyze_variable_flow",
            {
                "project_name": project_name,
                "sink_method": function_name,
                "variable_name": variable_name,
            },
        )

        if not result.success:
            return VariableFlowResult(
                success=False,
                variable=variable_name,
                error=result.error or "Failed to analyze variable flow",
            )

        data = _ensure_dict_response(result.data, "analyze_variable_flow")
        if data is None:
            return VariableFlowResult(
                success=False,
                variable=variable_name,
                error="Invalid response format from MCP server",
            )

        return VariableFlowResult(
            success=data.get("success", True),
            variable=data.get("variable", variable_name),
            definitions=data.get("definitions", []),
            usages=data.get("usages", []),
            error=data.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to analyze variable flow: {e}")
        raise MCPToolError(
            f"Failed to analyze variable flow: {e}",
            details={
                "function_name": function_name,
                "variable_name": variable_name,
            },
        ) from e
