"""MCP HTTP Client for Joern MCP Server communication.

This module provides an async HTTP client for communicating with the
Joern MCP Server using the MCP protocol over HTTP.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

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


@dataclass
class MCPResponse:
    """MCP JSON-RPC response wrapper.

    Attributes:
        id: Request ID
        result: Response result (for successful calls)
        error: Error information (for failed calls)
    """

    id: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    @property
    def is_success(self) -> bool:
        """Check if response is successful."""
        return self.error is None and self.result is not None


class MCPHttpClient:
    """Async HTTP client for Joern MCP Server.

    This client implements the MCP protocol using JSON-RPC 2.0 over HTTP.
    It supports async context manager protocol for proper resource management.

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
        self._client: httpx.AsyncClient | None = None
        self._request_id_counter = 0

    async def __aenter__(self) -> "MCPHttpClient":
        """Enter async context manager."""
        default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        default_headers.update(self.config.headers)

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            headers=default_headers,
        )
        logger.debug(f"MCP client initialized with URL: {self.config.url}")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.debug("MCP client closed")

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_id_counter += 1
        return f"req-{uuid.uuid4().hex[:8]}-{self._request_id_counter}"

    def _build_jsonrpc_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build JSON-RPC 2.0 request.

        Args:
            method: RPC method name
            params: Method parameters

        Returns:
            JSON-RPC request dict
        """
        request = {
            "jsonrpc": "2.0",
            "id": self._generate_request_id(),
            "method": method,
        }
        if params:
            request["params"] = params
        return request

    def _parse_response(self, response_data: dict[str, Any]) -> MCPResponse:
        """Parse JSON-RPC response.

        Args:
            response_data: Raw response data

        Returns:
            Parsed MCPResponse
        """
        return MCPResponse(
            id=response_data.get("id", ""),
            result=response_data.get("result"),
            error=response_data.get("error"),
        )

    def _extract_tool_result(self, mcp_response: MCPResponse) -> MCPToolResult:
        """Extract tool result from MCP response.

        Args:
            mcp_response: Parsed MCP response

        Returns:
            MCPToolResult with parsed data
        """
        if mcp_response.error:
            error_msg = mcp_response.error.get("message", "Unknown error")
            error_code = mcp_response.error.get("code", -1)
            return MCPToolResult(
                success=False,
                error=f"MCP Error [{error_code}]: {error_msg}",
            )

        if not mcp_response.result:
            return MCPToolResult(success=False, error="Empty response")

        # Extract content from MCP response
        content_list = mcp_response.result.get("content", [])
        if not content_list:
            return MCPToolResult(success=True, data={}, raw_content="")

        # Find text content
        raw_content = ""
        for content in content_list:
            if content.get("type") == "text":
                raw_content = content.get("text", "")
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

    async def _send_request(
        self,
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        """Send HTTP request with retry logic.

        Args:
            request_body: JSON-RPC request body

        Returns:
            Response data

        Raises:
            MCPConnectionError: If connection fails after all retries
        """
        if not self._client:
            raise MCPConnectionError("Client not initialized. Use 'async with' context manager.")

        last_error: Exception | None = None

        for attempt in range(self.config.retry_count + 1):
            try:
                logger.debug(
                    f"Sending MCP request (attempt {attempt + 1}): {request_body.get('method')}"
                )

                response = await self._client.post(
                    self.config.url,
                    json=request_body,
                )
                response.raise_for_status()
                return response.json()

            except httpx.ConnectError as e:
                last_error = e
                logger.warning(
                    f"Connection error (attempt {attempt + 1}/{self.config.retry_count + 1}): {e}"
                )
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Timeout error (attempt {attempt + 1}/{self.config.retry_count + 1}): {e}"
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(
                    f"HTTP error (attempt {attempt + 1}/{self.config.retry_count + 1}): "
                    f"{e.response.status_code}"
                )
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")
                raise MCPConnectionError(
                    f"Unexpected error during MCP request: {e}",
                    details={"error_type": type(e).__name__},
                ) from e

            # Wait before retry (exponential backoff)
            if attempt < self.config.retry_count:
                delay = self.config.retry_delay * (2**attempt)
                logger.debug(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        # All retries exhausted
        raise MCPConnectionError(
            f"Failed to connect to MCP server after {self.config.retry_count + 1} attempts",
            details={
                "url": self.config.url,
                "last_error": str(last_error),
            },
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
            MCPConnectionError: If connection fails
            MCPToolError: If tool call fails
        """
        logger.info(f"Calling MCP tool: {tool_name}")
        logger.debug(f"Tool arguments: {arguments}")

        request = self._build_jsonrpc_request(
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments or {},
            },
        )

        try:
            response_data = await self._send_request(request)
            mcp_response = self._parse_response(response_data)
            result = self._extract_tool_result(mcp_response)

            if result.success:
                logger.info(f"Tool call successful: {tool_name}")
                logger.debug(f"Tool result: {result.data}")
            else:
                logger.warning(f"Tool call failed: {tool_name} - {result.error}")

            return result

        except MCPConnectionError:
            raise
        except Exception as e:
            raise MCPToolError(
                f"Failed to call tool '{tool_name}': {e}",
                details={"tool_name": tool_name, "arguments": arguments},
            ) from e

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available MCP tools.

        Returns:
            List of tool definitions

        Raises:
            MCPConnectionError: If connection fails
        """
        logger.info("Listing available MCP tools")

        request = self._build_jsonrpc_request(method="tools/list")

        try:
            response_data = await self._send_request(request)
            mcp_response = self._parse_response(response_data)

            if mcp_response.error:
                logger.error(f"Failed to list tools: {mcp_response.error}")
                return []

            tools = mcp_response.result.get("tools", []) if mcp_response.result else []
            logger.info(f"Found {len(tools)} available tools")
            return tools

        except MCPConnectionError:
            raise
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []

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
