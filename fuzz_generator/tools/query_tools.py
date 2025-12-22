"""Query tools for Joern MCP Server.

This module provides wrapper functions for code query MCP tools:
- get_function_code: Get source code of a function
- list_functions: List functions in a project
- search_code: Search for code patterns
- execute_query: Execute custom Joern query
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fuzz_generator.exceptions import MCPToolError
from fuzz_generator.utils.logger import get_logger

if TYPE_CHECKING:
    from fuzz_generator.tools.mcp_client import MCPHttpClient

logger = get_logger(__name__)


@dataclass
class FunctionCodeResult:
    """Result of get_function_code operation.

    Attributes:
        success: Whether the operation was successful
        function_name: Name of the function
        code: Source code of the function
        file: File containing the function
        line_number: Starting line number
        end_line: Ending line number
        signature: Function signature
        error: Error message if any
    """

    success: bool
    function_name: str = ""
    code: str = ""
    file: str = ""
    line_number: int = 0
    end_line: int = 0
    signature: str = ""
    error: str | None = None


@dataclass
class FunctionBasicInfo:
    """Basic information about a function.

    Attributes:
        name: Function name
        file: Source file
        line_number: Line number
        signature: Function signature
        return_type: Return type
    """

    name: str
    file: str = ""
    line_number: int = 0
    signature: str = ""
    return_type: str = ""


@dataclass
class FunctionListResult:
    """Result of list_functions operation.

    Attributes:
        success: Whether the operation was successful
        functions: List of function information
        count: Total number of functions
        error: Error message if any
    """

    success: bool
    functions: list[FunctionBasicInfo] = field(default_factory=list)
    count: int = 0
    error: str | None = None


@dataclass
class CodeMatch:
    """A code search match.

    Attributes:
        code: Matched code snippet
        file: Source file
        line_number: Line number
        context: Surrounding context
    """

    code: str
    file: str = ""
    line_number: int = 0
    context: str = ""


@dataclass
class SearchCodeResult:
    """Result of search_code operation.

    Attributes:
        success: Whether the operation was successful
        matches: List of code matches
        count: Total number of matches
        error: Error message if any
    """

    success: bool
    matches: list[CodeMatch] = field(default_factory=list)
    count: int = 0
    error: str | None = None


@dataclass
class QueryResult:
    """Result of custom query execution.

    Attributes:
        success: Whether the operation was successful
        data: Query result data
        raw_output: Raw query output
        error: Error message if any
    """

    success: bool
    data: Any = None
    raw_output: str = ""
    error: str | None = None


async def get_function_code(
    client: "MCPHttpClient",
    function_name: str,
    project_name: str,
    file_name: str | None = None,
) -> FunctionCodeResult:
    """Get source code of a function.

    Args:
        client: MCP HTTP client instance
        function_name: Name of the function
        project_name: Name of the project
        file_name: Optional file name to narrow search

    Returns:
        FunctionCodeResult with function code

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Getting code for function: {function_name}")

    arguments: dict[str, Any] = {
        "function_name": function_name,
        "project_name": project_name,
    }
    if file_name:
        arguments["file_name"] = file_name

    try:
        result = await client.call_tool("get_function_code", arguments)

        if not result.success:
            return FunctionCodeResult(
                success=False,
                function_name=function_name,
                error=result.error or "Failed to get function code",
            )

        data = result.data

        # Handle the joern_mcp response format:
        # {"success": true, "project": "...", "functions": [{"code": "..."}], "count": 1}
        functions = data.get("functions", [])
        if functions:
            func_data = functions[0]
            code_str = func_data.get("code", "")

            # Handle double-encoded JSON string
            if code_str.startswith('""') and code_str.endswith('""'):
                code_str = code_str[2:-2]  # Remove outer quotes

            # Try to parse as JSON if it looks like an array
            if code_str.startswith("["):
                try:
                    parsed = json.loads(code_str)
                    if parsed and isinstance(parsed, list) and len(parsed) > 0:
                        func_info = parsed[0]
                        return FunctionCodeResult(
                            success=True,
                            function_name=func_info.get("name", function_name),
                            code=func_info.get("code", ""),
                            file=func_info.get("filename", ""),
                            line_number=func_info.get("lineNumber", 0),
                            end_line=func_info.get("lineNumberEnd", 0),
                            signature=func_info.get("signature", ""),
                        )
                except json.JSONDecodeError:
                    pass

            # Fallback: use as-is
            return FunctionCodeResult(
                success=True,
                function_name=function_name,
                code=code_str,
            )

        # Original format fallback
        return FunctionCodeResult(
            success=data.get("success", True),
            function_name=data.get("function_name", function_name),
            code=data.get("code", ""),
            file=data.get("file", ""),
            line_number=data.get("line_number", 0),
            end_line=data.get("end_line", 0),
            signature=data.get("signature", ""),
            error=data.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to get function code: {e}")
        raise MCPToolError(
            f"Failed to get function code: {e}",
            details={"function_name": function_name},
        ) from e


async def list_functions(
    client: "MCPHttpClient",
    project_name: str,
    file_name: str | None = None,
    pattern: str | None = None,
) -> FunctionListResult:
    """List functions in a project.

    Args:
        client: MCP HTTP client instance
        project_name: Name of the project
        file_name: Optional file to filter by
        pattern: Optional function name pattern

    Returns:
        FunctionListResult with function list

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Listing functions in project: {project_name}")

    arguments: dict[str, Any] = {"project_name": project_name}
    if file_name:
        arguments["file_name"] = file_name
    if pattern:
        arguments["pattern"] = pattern

    try:
        result = await client.call_tool("list_functions", arguments)

        if not result.success:
            return FunctionListResult(
                success=False,
                error=result.error or "Failed to list functions",
            )

        data = result.data
        functions_data = data.get("functions", [])

        functions = []
        for func in functions_data:
            functions.append(
                FunctionBasicInfo(
                    name=func.get("name", ""),
                    file=func.get("file", ""),
                    line_number=func.get("lineNumber", 0),
                    signature=func.get("signature", ""),
                    return_type=func.get("returnType", ""),
                )
            )

        return FunctionListResult(
            success=data.get("success", True),
            functions=functions,
            count=data.get("count", len(functions)),
            error=data.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to list functions: {e}")
        raise MCPToolError(
            f"Failed to list functions: {e}",
            details={"project_name": project_name},
        ) from e


async def search_code(
    client: "MCPHttpClient",
    project_name: str,
    pattern: str,
    file_name: str | None = None,
    case_sensitive: bool = True,
) -> SearchCodeResult:
    """Search for code patterns.

    Args:
        client: MCP HTTP client instance
        project_name: Name of the project
        pattern: Search pattern (regex or literal)
        file_name: Optional file to search in
        case_sensitive: Whether search is case sensitive

    Returns:
        SearchCodeResult with matches

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Searching code for pattern: {pattern}")

    arguments: dict[str, Any] = {
        "project_name": project_name,
        "pattern": pattern,
        "case_sensitive": case_sensitive,
    }
    if file_name:
        arguments["file_name"] = file_name

    try:
        result = await client.call_tool("search_code", arguments)

        if not result.success:
            return SearchCodeResult(
                success=False,
                error=result.error or "Failed to search code",
            )

        data = result.data
        matches_data = data.get("matches", [])

        matches = []
        for match in matches_data:
            matches.append(
                CodeMatch(
                    code=match.get("code", ""),
                    file=match.get("file", ""),
                    line_number=match.get("lineNumber", 0),
                    context=match.get("context", ""),
                )
            )

        return SearchCodeResult(
            success=data.get("success", True),
            matches=matches,
            count=data.get("count", len(matches)),
            error=data.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to search code: {e}")
        raise MCPToolError(
            f"Failed to search code: {e}",
            details={"pattern": pattern},
        ) from e


async def execute_query(
    client: "MCPHttpClient",
    project_name: str,
    query: str,
) -> QueryResult:
    """Execute a custom Joern query.

    Args:
        client: MCP HTTP client instance
        project_name: Name of the project
        query: Joern query string

    Returns:
        QueryResult with query results

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Executing query in project: {project_name}")
    logger.debug(f"Query: {query}")

    try:
        result = await client.call_tool(
            "execute_query",
            {
                "project_name": project_name,
                "query": query,
            },
        )

        if not result.success:
            return QueryResult(
                success=False,
                error=result.error or "Failed to execute query",
            )

        data = result.data
        return QueryResult(
            success=data.get("success", True),
            data=data.get("result"),
            raw_output=result.raw_content,
            error=data.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        raise MCPToolError(
            f"Failed to execute query: {e}",
            details={"query": query},
        ) from e
