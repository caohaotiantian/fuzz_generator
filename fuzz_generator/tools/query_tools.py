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


def _ensure_dict_response(data: Any, tool_name: str) -> dict | None:
    """Ensure MCP tool response is a dictionary.

    Args:
        data: Response data from MCP tool
        tool_name: Name of the tool (for error logging)

    Returns:
        Parsed dictionary or None if parsing fails
    """
    # Handle case where data might be a string (needs JSON parsing)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse {tool_name} response as JSON: {data}")
            return None

    # Ensure data is a dictionary
    if not isinstance(data, dict):
        logger.error(f"Unexpected {tool_name} response type: {type(data)}, data: {data}")
        return None

    return data


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
    project_path: Any = None,
) -> FunctionCodeResult:
    """Get source code of a function.

    Args:
        client: MCP HTTP client instance
        function_name: Name of the function
        project_name: Name of the project
        file_name: Optional file name to narrow search (regex pattern)
        project_path: Optional project root path for reading complete source files

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
        # MCP Server expects 'file_filter' parameter
        arguments["file_filter"] = file_name

    try:
        result = await client.call_tool("get_function_code", arguments)

        if not result.success:
            return FunctionCodeResult(
                success=False,
                function_name=function_name,
                error=result.error or "Failed to get function code",
            )

        data = _ensure_dict_response(result.data, "get_function_code")
        if data is None:
            return FunctionCodeResult(
                success=False,
                function_name=function_name,
                error="Invalid response format from MCP server",
            )

        logger.info(f"[DEBUG] MCP 返回的原始数据长度: {len(str(data))} 字符")
        logger.info(f"[DEBUG] MCP 返回的 data keys: {list(data.keys())}")

        # Handle the joern_mcp response format:
        # {"success": true, "project": "...", "functions": [{"code": "..."}], "count": 1}
        functions = data.get("functions", [])
        logger.info(f"[DEBUG] functions 数量: {len(functions)}")
        if functions:
            func_data = functions[0]
            logger.info(f"[DEBUG] func_data keys: {list(func_data.keys())}")

            # Extract metadata from Joern
            filename = func_data.get("filename", "")
            line_number = func_data.get("lineNumber", 0)
            end_line = func_data.get("lineNumberEnd", 0)
            signature = func_data.get("signature", "")
            func_name = func_data.get("name", function_name)

            logger.info(
                f"[DEBUG] 元数据 - filename: {filename}, line_number: {line_number}, end_line: {end_line}"
            )
            logger.info(f"[DEBUG] project_path: {project_path}")

            # Try to read complete code from source file if project_path is provided
            complete_code = None
            if project_path and filename and line_number > 0 and end_line > 0:
                try:
                    from pathlib import Path

                    # Convert project_path to Path object
                    if isinstance(project_path, str):
                        project_path = Path(project_path)

                    source_file = project_path / filename
                    logger.info(f"尝试从源文件读取完整代码: {source_file}")

                    if source_file.exists():
                        with open(source_file, encoding="utf-8") as f:
                            lines = f.readlines()
                            # Line numbers are 1-based, array indices are 0-based
                            complete_code = "".join(lines[line_number - 1 : end_line])
                            logger.info(
                                f"✅ 从源文件读取到完整代码: {len(complete_code)} 字符, {len(complete_code.splitlines())} 行"
                            )
                    else:
                        logger.warning(f"源文件不存在: {source_file}")
                except Exception as e:
                    logger.warning(f"从源文件读取代码失败: {e}")

            # Use complete code from source file, or fallback to Joern's code
            if complete_code:
                final_code = complete_code
                logger.info("✅ 使用从源文件读取的完整代码")
            else:
                # Fallback to Joern's code (may be truncated)
                code_str = func_data.get("code", "")
                logger.info(f"⚠️ 使用 Joern 返回的代码（可能被截断）: {len(code_str)} 字符")

                # Handle multi-level JSON encoding
                # Sometimes MCP returns: {\"code\": \"\"[{...}]\"\"}
                while code_str.startswith('""') and code_str.endswith('""') and len(code_str) > 4:
                    code_str = code_str[2:-2]  # Remove outer double quotes

                # Try to parse as JSON array
                if code_str.startswith("["):
                    try:
                        parsed = json.loads(code_str)
                        if parsed and isinstance(parsed, list) and len(parsed) > 0:
                            func_info = parsed[0]
                            # Update metadata if available
                            if isinstance(func_info, dict):
                                filename = func_info.get("filename", filename)
                                line_number = func_info.get("lineNumber", line_number)
                                end_line = func_info.get("lineNumberEnd", end_line)
                                signature = func_info.get("signature", signature)
                                func_name = func_info.get("name", func_name)
                                code_str = func_info.get("code", "")
                                logger.info(
                                    f"📋 从嵌套 JSON 解析到元数据 - filename: {filename}, line_number: {line_number}, end_line: {end_line}"
                                )

                                # Now try to read from source file with parsed metadata
                                if project_path and filename and line_number > 0 and end_line > 0:
                                    try:
                                        from pathlib import Path

                                        if isinstance(project_path, str):
                                            project_path = Path(project_path)
                                        source_file = project_path / filename
                                        logger.info(f"尝试从源文件读取完整代码: {source_file}")
                                        if source_file.exists():
                                            with open(source_file, encoding="utf-8") as f:
                                                lines = f.readlines()
                                                complete_code = "".join(
                                                    lines[line_number - 1 : end_line]
                                                )
                                                logger.info(
                                                    f"✅ 从源文件读取到完整代码: {len(complete_code)} 字符, {len(complete_code.splitlines())} 行"
                                                )
                                                code_str = complete_code
                                        else:
                                            logger.warning(f"源文件不存在: {source_file}")
                                    except Exception as e:
                                        logger.warning(f"从源文件读取代码失败: {e}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON 解析失败: {e}")

                final_code = code_str

            return FunctionCodeResult(
                success=True,
                function_name=func_name,
                code=final_code,
                file=filename,
                line_number=line_number,
                end_line=end_line,
                signature=signature,
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
    limit: int = 100,
) -> FunctionListResult:
    """List functions in a project.

    Args:
        client: MCP HTTP client instance
        project_name: Name of the project
        file_name: Optional file to filter by (not supported by MCP server, ignored)
        pattern: Optional function name pattern (regex)
        limit: Maximum number of functions to return

    Returns:
        FunctionListResult with function list

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Listing functions in project: {project_name}")

    arguments: dict[str, Any] = {
        "project_name": project_name,
        "limit": limit,
    }
    # MCP Server expects 'name_filter' parameter
    if pattern:
        arguments["name_filter"] = pattern
    # Note: file_name parameter is not supported by MCP server, ignored

    try:
        result = await client.call_tool("list_functions", arguments)

        if not result.success:
            return FunctionListResult(
                success=False,
                error=result.error or "Failed to list functions",
            )

        data = _ensure_dict_response(result.data, "list_functions")
        if data is None:
            return FunctionListResult(
                success=False,
                error="Invalid response format from MCP server",
            )

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
    scope: str = "all",
) -> SearchCodeResult:
    """Search for code patterns.

    Args:
        client: MCP HTTP client instance
        project_name: Name of the project
        pattern: Search pattern (regex)
        file_name: Optional file to search in (not supported by MCP server, ignored)
        case_sensitive: Whether search is case sensitive (not supported by MCP server, ignored)
        scope: Search scope - "all" (calls and identifiers), "calls", "identifiers"

    Returns:
        SearchCodeResult with matches

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Searching code for pattern: {pattern}")

    arguments: dict[str, Any] = {
        "project_name": project_name,
        "pattern": pattern,
        "scope": scope,
    }
    # Note: file_name and case_sensitive are not supported by MCP server, ignored

    try:
        result = await client.call_tool("search_code", arguments)

        if not result.success:
            return SearchCodeResult(
                success=False,
                error=result.error or "Failed to search code",
            )

        data = _ensure_dict_response(result.data, "search_code")
        if data is None:
            return SearchCodeResult(
                success=False,
                error="Invalid response format from MCP server",
            )

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

        data = _ensure_dict_response(result.data, "execute_query")
        if data is None:
            return QueryResult(
                success=False,
                error="Invalid response format from MCP server",
            )

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
