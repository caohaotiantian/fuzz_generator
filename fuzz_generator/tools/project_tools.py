"""Project management tools for Joern MCP Server.

This module provides wrapper functions for project-related MCP tools:
- parse_project: Parse source code and create CPG
- list_projects: List all parsed projects
- switch_project: Switch active project
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fuzz_generator.exceptions import MCPToolError
from fuzz_generator.utils.logger import get_logger

if TYPE_CHECKING:
    from fuzz_generator.tools.mcp_client import MCPHttpClient

logger = get_logger(__name__)


@dataclass
class ProjectInfo:
    """Information about a parsed project.

    Attributes:
        name: Project name
        input_path: Source code path
        is_active: Whether this is the active project
    """

    name: str
    input_path: str
    is_active: bool = False


@dataclass
class ParseProjectResult:
    """Result of parse_project operation.

    Attributes:
        success: Whether parsing was successful
        project_name: Name of the parsed project
        message: Status message
        error: Error message if any
    """

    success: bool
    project_name: str = ""
    message: str = ""
    error: str | None = None


@dataclass
class ProjectListResult:
    """Result of list_projects operation.

    Attributes:
        success: Whether the operation was successful
        projects: List of project information
        count: Total number of projects
        error: Error message if any
    """

    success: bool
    projects: list[ProjectInfo] = field(default_factory=list)
    count: int = 0
    error: str | None = None


@dataclass
class SwitchProjectResult:
    """Result of switch_project operation.

    Attributes:
        success: Whether switching was successful
        project_name: Name of the active project
        message: Status message
        error: Error message if any
    """

    success: bool
    project_name: str = ""
    message: str = ""
    error: str | None = None


async def parse_project(
    client: "MCPHttpClient",
    source_path: str,
    project_name: str | None = None,
    language: str = "auto",
) -> ParseProjectResult:
    """Parse source code project and create Code Property Graph.

    Args:
        client: MCP HTTP client instance
        source_path: Path to source code directory
        project_name: Optional project name (defaults to directory name)
        language: Programming language (default: "auto" for auto-detection)

    Returns:
        ParseProjectResult with operation status

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Parsing project: {source_path}")

    arguments = {
        "source_path": source_path,
        "language": language,
    }
    if project_name:
        arguments["project_name"] = project_name

    try:
        result = await client.call_tool("parse_project", arguments)

        if not result.success:
            return ParseProjectResult(
                success=False,
                error=result.error or "Failed to parse project",
            )

        data = result.data
        return ParseProjectResult(
            success=data.get("success", True),
            project_name=data.get("project_name", ""),
            message=data.get("message", ""),
            error=data.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to parse project: {e}")
        raise MCPToolError(
            f"Failed to parse project: {e}",
            details={"source_path": source_path, "project_name": project_name},
        ) from e


async def list_projects(client: "MCPHttpClient") -> ProjectListResult:
    """List all parsed projects.

    Args:
        client: MCP HTTP client instance

    Returns:
        ProjectListResult with list of projects

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info("Listing projects")

    try:
        result = await client.call_tool("list_projects", {})

        if not result.success:
            return ProjectListResult(
                success=False,
                error=result.error or "Failed to list projects",
            )

        data = result.data
        projects_data = data.get("projects", [])

        projects = []
        for proj in projects_data:
            projects.append(
                ProjectInfo(
                    name=proj.get("name", ""),
                    input_path=proj.get("inputPath", ""),
                    is_active=proj.get("isActive", False),
                )
            )

        return ProjectListResult(
            success=data.get("success", True),
            projects=projects,
            count=data.get("count", len(projects)),
            error=data.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise MCPToolError(f"Failed to list projects: {e}") from e


async def switch_project(
    client: "MCPHttpClient",
    project_name: str,
) -> SwitchProjectResult:
    """Switch to a different active project.

    Args:
        client: MCP HTTP client instance
        project_name: Name of the project to switch to

    Returns:
        SwitchProjectResult with operation status

    Raises:
        MCPToolError: If the tool call fails
    """
    logger.info(f"Switching to project: {project_name}")

    try:
        result = await client.call_tool(
            "switch_project",
            {"project_name": project_name},
        )

        if not result.success:
            return SwitchProjectResult(
                success=False,
                error=result.error or "Failed to switch project",
            )

        data = result.data
        return SwitchProjectResult(
            success=data.get("success", True),
            project_name=data.get("project_name", project_name),
            message=data.get("message", ""),
            error=data.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to switch project: {e}")
        raise MCPToolError(
            f"Failed to switch project: {e}",
            details={"project_name": project_name},
        ) from e
