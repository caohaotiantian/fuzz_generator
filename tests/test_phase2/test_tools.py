"""Tests for MCP tool wrappers."""

from unittest.mock import AsyncMock

import pytest

from fuzz_generator.exceptions import MCPToolError
from fuzz_generator.tools import (
    CalleesResult,
    CallersResult,
    ControlFlowResult,
    DataFlowResult,
    FunctionCodeResult,
    FunctionListResult,
    ParseProjectResult,
    ProjectListResult,
    QueryResult,
    SearchCodeResult,
    analyze_variable_flow,
    execute_query,
    get_callees,
    get_callers,
    get_control_flow_graph,
    # Query tools
    get_function_code,
    list_functions,
    list_projects,
    # Project tools
    parse_project,
    search_code,
    switch_project,
    # Analysis tools
    track_dataflow,
)
from fuzz_generator.tools.mcp_client import MCPToolResult


class TestProjectTools:
    """Tests for project management tools."""

    @pytest.fixture
    def mock_client(self):
        """Create mock MCP client."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_parse_project_success(self, mock_client):
        """Test successful project parsing."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "project_name": "test_project",
                "message": "Project parsed successfully",
            },
        )

        result = await parse_project(
            mock_client,
            source_path="/path/to/source",
            project_name="test_project",
        )

        assert isinstance(result, ParseProjectResult)
        assert result.success is True
        assert result.project_name == "test_project"
        assert result.message == "Project parsed successfully"

        mock_client.call_tool.assert_called_once_with(
            "parse_project",
            {
                "source_path": "/path/to/source",
                "project_name": "test_project",
                "language": "auto",
            },
        )

    @pytest.mark.asyncio
    async def test_parse_project_failure(self, mock_client):
        """Test failed project parsing."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=False,
            error="Failed to parse project",
        )

        result = await parse_project(mock_client, source_path="/invalid/path")

        assert result.success is False
        assert result.error == "Failed to parse project"

    @pytest.mark.asyncio
    async def test_parse_project_exception(self, mock_client):
        """Test exception during project parsing."""
        mock_client.call_tool.side_effect = Exception("Connection error")

        with pytest.raises(MCPToolError) as exc_info:
            await parse_project(mock_client, source_path="/path")

        assert "Failed to parse project" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_projects_success(self, mock_client):
        """Test listing projects."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "projects": [
                    {"name": "proj1", "inputPath": "/path1", "isActive": True},
                    {"name": "proj2", "inputPath": "/path2", "isActive": False},
                ],
                "count": 2,
            },
        )

        result = await list_projects(mock_client)

        assert isinstance(result, ProjectListResult)
        assert result.success is True
        assert len(result.projects) == 2
        assert result.projects[0].name == "proj1"
        assert result.projects[0].is_active is True
        assert result.count == 2

    @pytest.mark.asyncio
    async def test_list_projects_empty(self, mock_client):
        """Test listing projects when none exist."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={"success": True, "projects": [], "count": 0},
        )

        result = await list_projects(mock_client)

        assert result.success is True
        assert len(result.projects) == 0

    @pytest.mark.asyncio
    async def test_switch_project_success(self, mock_client):
        """Test switching project."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "project_name": "new_project",
                "message": "Switched to project",
            },
        )

        result = await switch_project(mock_client, project_name="new_project")

        assert result.success is True
        assert result.project_name == "new_project"


class TestAnalysisTools:
    """Tests for analysis tools."""

    @pytest.fixture
    def mock_client(self):
        """Create mock MCP client."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_track_dataflow_success(self, mock_client):
        """Test data flow tracking."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "flows": [
                    {
                        "source": {"code": "gets(buf)", "file": "main.c", "line": 10},
                        "sink": {"code": "system(buf)", "file": "main.c", "line": 20},
                        "pathLength": 3,
                        "pathDetails": [],
                    }
                ],
                "count": 1,
            },
        )

        result = await track_dataflow(
            mock_client,
            project_name="test_project",
            source_method="gets",
            sink_method="system",
        )

        assert isinstance(result, DataFlowResult)
        assert result.success is True
        assert len(result.flows) == 1
        assert result.flows[0].source["code"] == "gets(buf)"
        assert result.flows[0].path_length == 3

    @pytest.mark.asyncio
    async def test_track_dataflow_no_flows(self, mock_client):
        """Test data flow tracking with no results."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={"success": True, "flows": [], "count": 0},
        )

        result = await track_dataflow(
            mock_client,
            project_name="test_project",
            source_method="safe_input",
            sink_method="safe_output",
        )

        assert result.success is True
        assert len(result.flows) == 0

    @pytest.mark.asyncio
    async def test_get_callers_success(self, mock_client):
        """Test getting function callers."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "function": "process",
                "callers": [
                    {"name": "main", "filename": "main.c", "lineNumber": 10, "code": "process()"},
                    {"name": "handler", "filename": "handler.c", "lineNumber": 25, "code": ""},
                ],
                "count": 2,
            },
        )

        result = await get_callers(
            mock_client,
            project_name="test_project",
            function_name="process",
        )

        assert isinstance(result, CallersResult)
        assert result.success is True
        assert len(result.callers) == 2
        assert result.callers[0].name == "main"
        assert result.callers[0].line_number == 10

    @pytest.mark.asyncio
    async def test_get_callees_success(self, mock_client):
        """Test getting function callees."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "function": "main",
                "callees": [
                    {"name": "init", "filename": "init.c", "lineNumber": 5},
                    {"name": "process", "filename": "process.c", "lineNumber": 15},
                ],
                "count": 2,
            },
        )

        result = await get_callees(
            mock_client,
            project_name="test_project",
            function_name="main",
        )

        assert isinstance(result, CalleesResult)
        assert result.success is True
        assert len(result.callees) == 2
        assert result.callees[0].name == "init"

    @pytest.mark.asyncio
    async def test_get_control_flow_graph_success(self, mock_client):
        """Test getting control flow graph."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "function": "process",
                "nodes": [
                    {"id": "1", "code": "entry", "type": "ENTRY", "lineNumber": 1},
                    {"id": "2", "code": "if (x > 0)", "type": "CONDITION", "lineNumber": 2},
                ],
                "edges": [
                    {"from": "1", "to": "2", "type": "default"},
                ],
                "hasLoops": False,
                "hasConditions": True,
            },
        )

        result = await get_control_flow_graph(
            mock_client,
            project_name="test_project",
            function_name="process",
        )

        assert isinstance(result, ControlFlowResult)
        assert result.success is True
        assert len(result.nodes) == 2
        assert len(result.edges) == 1
        assert result.has_conditions is True
        assert result.has_loops is False

    @pytest.mark.asyncio
    async def test_analyze_variable_flow_success(self, mock_client):
        """Test variable flow analysis."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "variable": "buf",
                "definitions": [{"line": 5, "code": "char buf[256]"}],
                "usages": [{"line": 10, "code": "strcpy(buf, input)"}],
            },
        )

        result = await analyze_variable_flow(
            mock_client,
            project_name="test_project",
            function_name="process",
            variable_name="buf",
        )

        assert result.success is True
        assert result.variable == "buf"
        assert len(result.definitions) == 1
        assert len(result.usages) == 1


class TestQueryTools:
    """Tests for query tools."""

    @pytest.fixture
    def mock_client(self):
        """Create mock MCP client."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_get_function_code_success(self, mock_client):
        """Test getting function code."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "function_name": "main",
                "code": "int main() { return 0; }",
                "file": "main.c",
                "line_number": 1,
                "end_line": 3,
                "signature": "int main()",
            },
        )

        result = await get_function_code(
            mock_client,
            function_name="main",
            project_name="test_project",
        )

        assert isinstance(result, FunctionCodeResult)
        assert result.success is True
        assert result.function_name == "main"
        assert "int main()" in result.code
        assert result.file == "main.c"

        mock_client.call_tool.assert_called_once_with(
            "get_function_code",
            {
                "function_name": "main",
                "project_name": "test_project",
            },
        )

    @pytest.mark.asyncio
    async def test_get_function_code_with_file(self, mock_client):
        """Test getting function code with file name filter."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "function_name": "process",
                "code": "void process() {}",
                "file": "handler.c",
                "line_number": 10,
            },
        )

        result = await get_function_code(
            mock_client,
            function_name="process",
            project_name="test_project",
            file_name="handler.c",
        )

        assert result.success is True

        # Verify file_name was passed
        call_args = mock_client.call_tool.call_args
        assert call_args[0][1]["file_name"] == "handler.c"

    @pytest.mark.asyncio
    async def test_get_function_code_not_found(self, mock_client):
        """Test getting non-existent function."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": False,
                "error": "Function not found",
            },
        )

        result = await get_function_code(
            mock_client,
            function_name="nonexistent",
            project_name="test_project",
        )

        assert result.success is False

    @pytest.mark.asyncio
    async def test_list_functions_success(self, mock_client):
        """Test listing functions."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "functions": [
                    {
                        "name": "main",
                        "file": "main.c",
                        "lineNumber": 1,
                        "signature": "int main()",
                        "returnType": "int",
                    },
                    {
                        "name": "process",
                        "file": "handler.c",
                        "lineNumber": 10,
                        "signature": "void process()",
                        "returnType": "void",
                    },
                ],
                "count": 2,
            },
        )

        result = await list_functions(mock_client, project_name="test_project")

        assert isinstance(result, FunctionListResult)
        assert result.success is True
        assert len(result.functions) == 2
        assert result.functions[0].name == "main"
        assert result.functions[0].return_type == "int"

    @pytest.mark.asyncio
    async def test_list_functions_with_filter(self, mock_client):
        """Test listing functions with file filter."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "functions": [{"name": "process", "file": "handler.c", "lineNumber": 10}],
                "count": 1,
            },
        )

        result = await list_functions(
            mock_client,
            project_name="test_project",
            file_name="handler.c",
        )

        assert result.success is True
        assert len(result.functions) == 1

    @pytest.mark.asyncio
    async def test_search_code_success(self, mock_client):
        """Test code search."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "matches": [
                    {
                        "code": "strcpy(buf, input)",
                        "file": "main.c",
                        "lineNumber": 15,
                        "context": "void func() { strcpy(buf, input); }",
                    }
                ],
                "count": 1,
            },
        )

        result = await search_code(
            mock_client,
            project_name="test_project",
            pattern="strcpy",
        )

        assert isinstance(result, SearchCodeResult)
        assert result.success is True
        assert len(result.matches) == 1
        assert "strcpy" in result.matches[0].code

    @pytest.mark.asyncio
    async def test_search_code_no_matches(self, mock_client):
        """Test code search with no matches."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={"success": True, "matches": [], "count": 0},
        )

        result = await search_code(
            mock_client,
            project_name="test_project",
            pattern="nonexistent_pattern",
        )

        assert result.success is True
        assert len(result.matches) == 0

    @pytest.mark.asyncio
    async def test_execute_query_success(self, mock_client):
        """Test custom query execution."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=True,
            data={
                "success": True,
                "result": [{"name": "main"}, {"name": "process"}],
            },
            raw_content='[{"name": "main"}, {"name": "process"}]',
        )

        result = await execute_query(
            mock_client,
            project_name="test_project",
            query="cpg.method.name.l",
        )

        assert isinstance(result, QueryResult)
        assert result.success is True
        assert len(result.data) == 2

    @pytest.mark.asyncio
    async def test_execute_query_error(self, mock_client):
        """Test custom query with error."""
        mock_client.call_tool.return_value = MCPToolResult(
            success=False,
            error="Query syntax error",
        )

        result = await execute_query(
            mock_client,
            project_name="test_project",
            query="invalid query",
        )

        assert result.success is False
        assert result.error == "Query syntax error"


class TestToolExceptions:
    """Tests for tool exception handling."""

    @pytest.fixture
    def mock_client(self):
        """Create mock MCP client."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_project_tool_exception(self, mock_client):
        """Test exception handling in project tools."""
        mock_client.call_tool.side_effect = Exception("Network error")

        with pytest.raises(MCPToolError) as exc_info:
            await list_projects(mock_client)

        assert "Failed to list projects" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analysis_tool_exception(self, mock_client):
        """Test exception handling in analysis tools."""
        mock_client.call_tool.side_effect = Exception("Timeout")

        with pytest.raises(MCPToolError) as exc_info:
            await get_callers(
                mock_client,
                project_name="test",
                function_name="func",
            )

        assert "Failed to get callers" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_tool_exception(self, mock_client):
        """Test exception handling in query tools."""
        mock_client.call_tool.side_effect = Exception("Server error")

        with pytest.raises(MCPToolError) as exc_info:
            await search_code(
                mock_client,
                project_name="test",
                pattern="test",
            )

        assert "Failed to search code" in str(exc_info.value)
