"""MCP HTTP Client for Joern MCP Server communication.

This module provides an async HTTP client for communicating with the
Joern MCP Server using the official MCP protocol via streamable HTTP transport.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from fuzz_generator.exceptions import MCPConnectionError, MCPToolError
from fuzz_generator.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MCPClientConfig:
    """Configuration for MCP HTTP client.

    Attributes:
        url: MCP server URL (e.g., "http://localhost:8000/mcp")
        timeout: Request timeout in seconds
        retry_count: Number of retries on failure
        retry_delay: Base delay between retries in seconds
        headers: Additional HTTP headers
    """

    url: str = "http://localhost:8000/mcp"
    timeout: int = 60
    retry_count: int = 3
    retry_delay: float = 2.0
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.url:
            raise ValueError("MCP server URL is required")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.retry_count < 0:
            raise ValueError("Retry count must be non-negative")


@dataclass
class MCPToolResult:
    """Result of an MCP tool call.

    Attributes:
        success: Whether the call was successful
        data: Parsed response data
        raw_content: Raw content from the response
        error: Error message if any
    """

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    raw_content: str = ""
    error: str | None = None


class MCPHttpClient:
    """Async HTTP client for Joern MCP Server using official MCP protocol.

    This client uses the official MCP SDK's streamable HTTP transport
    for communicating with FastMCP-based servers.

    Usage:
        async with MCPHttpClient(config) as client:
            result = await client.call_tool("get_function_code", {"name": "main"})
            tools = await client.list_tools()
    """

    def __init__(self, config: MCPClientConfig) -> None:
        """Initialize MCP client.

        Args:
            config: Client configuration
        """
        self.config = config
        self._session = None
        self._read = None
        self._write = None
        self._context_stack = None

    async def __aenter__(self) -> "MCPHttpClient":
        """Enter async context manager."""
        try:
            from contextlib import AsyncExitStack

            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as e:
            raise MCPConnectionError(
                "MCP SDK not available. Please install: pip install mcp",
                details={"import_error": str(e)},
            ) from e

        logger.debug(f"Connecting to MCP server: {self.config.url}")

        try:
            # Create exit stack to manage nested context managers
            self._context_stack = AsyncExitStack()
            await self._context_stack.__aenter__()

            # Connect using streamable HTTP transport
            # New API returns 3 values: read, write, get_session_id
            (
                self._read,
                self._write,
                _get_session_id,
            ) = await self._context_stack.enter_async_context(
                streamablehttp_client(self.config.url)
            )

            # Create and initialize session
            self._session = await self._context_stack.enter_async_context(
                ClientSession(self._read, self._write)
            )

            # Initialize the MCP session
            await self._session.initialize()

            logger.info(f"Connected to MCP server: {self.config.url}")
            return self

        except Exception as e:
            # Clean up on failure
            if self._context_stack:
                await self._context_stack.__aexit__(type(e), e, e.__traceback__)
                self._context_stack = None
            raise MCPConnectionError(
                f"Failed to connect to MCP server: {e}",
                details={"url": self.config.url},
            ) from e

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        if self._context_stack:
            try:
                await self._context_stack.__aexit__(exc_type, exc_val, exc_tb)
            except BaseException as e:
                # Suppress cleanup errors from anyio task groups
                # These can happen when the connection is closed
                logger.debug(f"Suppressed cleanup error: {type(e).__name__}: {e}")
            finally:
                self._context_stack = None
                self._session = None
                self._read = None
                self._write = None
                logger.debug("MCP client disconnected")

    def _extract_tool_result(self, result: Any) -> MCPToolResult:
        """Extract tool result from MCP response.

        Args:
            result: MCP call result

        Returns:
            MCPToolResult with parsed data
        """
        if not hasattr(result, "content") or not result.content:
            return MCPToolResult(success=True, data={}, raw_content="")

        # Find text content
        raw_content = ""
        for content in result.content:
            if hasattr(content, "text"):
                raw_content = content.text
                break

        # Try to parse as JSON
        try:
            data = json.loads(raw_content) if raw_content else {}
            return MCPToolResult(
                success=True,
                data=data,
                raw_content=raw_content,
            )
        except json.JSONDecodeError:
            # Return raw content if not valid JSON
            return MCPToolResult(
                success=True,
                data={"raw": raw_content},
                raw_content=raw_content,
            )

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Call an MCP tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            MCPToolResult with the response data

        Raises:
            MCPConnectionError: If not connected
            MCPToolError: If tool call fails
        """
        if not self._session:
            raise MCPConnectionError("Client not initialized. Use 'async with' context manager.")

        logger.info(f"Calling MCP tool: {tool_name}")
        logger.debug(f"Tool arguments: {arguments}")

        last_error: Exception | None = None

        for attempt in range(self.config.retry_count + 1):
            try:
                result = await asyncio.wait_for(
                    self._session.call_tool(tool_name, arguments or {}),
                    timeout=self.config.timeout,
                )
                tool_result = self._extract_tool_result(result)

                if tool_result.success:
                    logger.info(f"Tool call successful: {tool_name}")
                    logger.debug(f"Tool result: {tool_result.data}")

                return tool_result

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(f"Timeout (attempt {attempt + 1}/{self.config.retry_count + 1})")
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Tool call error (attempt {attempt + 1}/{self.config.retry_count + 1}): {e}"
                )

            # Wait before retry
            if attempt < self.config.retry_count:
                delay = self.config.retry_delay * (2**attempt)
                logger.debug(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        # All retries exhausted
        raise MCPToolError(
            f"Failed to call tool '{tool_name}' after {self.config.retry_count + 1} attempts",
            details={"tool_name": tool_name, "last_error": str(last_error)},
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available MCP tools.

        Returns:
            List of tool definitions

        Raises:
            MCPConnectionError: If not connected
        """
        if not self._session:
            raise MCPConnectionError("Client not initialized. Use 'async with' context manager.")

        logger.info("Listing available MCP tools")

        try:
            result = await asyncio.wait_for(
                self._session.list_tools(),
                timeout=self.config.timeout,
            )

            tools = []
            if hasattr(result, "tools"):
                for tool in result.tools:
                    tools.append(
                        {
                            "name": tool.name,
                            "description": getattr(tool, "description", ""),
                            "inputSchema": getattr(tool, "inputSchema", {}),
                        }
                    )

            logger.info(f"Found {len(tools)} available tools")
            return tools

        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            raise MCPConnectionError(
                f"Failed to list tools: {e}",
                details={"url": self.config.url},
            ) from e

    async def ping(self) -> bool:
        """Check if MCP server is reachable.

        Returns:
            True if server is reachable, False otherwise
        """
        try:
            tools = await self.list_tools()
            return len(tools) > 0
        except Exception as e:
            logger.warning(f"MCP server ping failed: {e}")
            return False
