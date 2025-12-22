"""Tools module for MCP client and tool wrappers.

This module provides:
- MCPHttpClient: Async HTTP client for Joern MCP Server communication
- Project tools: parse_project, list_projects, switch_project
- Analysis tools: track_dataflow, get_callers, get_callees, etc.
- Query tools: get_function_code, list_functions, search_code, etc.
"""

from fuzz_generator.tools.analysis_tools import (
    CalleeInfo,
    CalleesResult,
    CallerInfo,
    CallersResult,
    ControlFlowResult,
    DataFlowResult,
    FlowInfo,
    analyze_variable_flow,
    get_callees,
    get_callers,
    get_control_flow_graph,
    track_dataflow,
)
from fuzz_generator.tools.mcp_client import (
    MCPClientConfig,
    MCPHttpClient,
    MCPToolResult,
)
from fuzz_generator.tools.project_tools import (
    ParseProjectResult,
    ProjectInfo,
    ProjectListResult,
    list_projects,
    parse_project,
    switch_project,
)
from fuzz_generator.tools.query_tools import (
    CodeMatch,
    FunctionBasicInfo,
    FunctionCodeResult,
    FunctionListResult,
    QueryResult,
    SearchCodeResult,
    execute_query,
    get_function_code,
    list_functions,
    search_code,
)

__all__ = [
    # MCP Client
    "MCPClientConfig",
    "MCPHttpClient",
    "MCPToolResult",
    # Project Tools
    "ParseProjectResult",
    "ProjectInfo",
    "ProjectListResult",
    "parse_project",
    "list_projects",
    "switch_project",
    # Analysis Tools
    "DataFlowResult",
    "FlowInfo",
    "CallerInfo",
    "CallersResult",
    "CalleeInfo",
    "CalleesResult",
    "ControlFlowResult",
    "track_dataflow",
    "get_callers",
    "get_callees",
    "get_control_flow_graph",
    "analyze_variable_flow",
    # Query Tools
    "FunctionCodeResult",
    "FunctionListResult",
    "FunctionBasicInfo",
    "SearchCodeResult",
    "CodeMatch",
    "QueryResult",
    "get_function_code",
    "list_functions",
    "search_code",
    "execute_query",
]
