"""Tests for MCP HTTP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from fuzz_generator.exceptions import MCPConnectionError
from fuzz_generator.tools.mcp_client import (
    MCPClientConfig,
    MCPHttpClient,
    MCPResponse,
    MCPToolResult,
)


class TestMCPClientConfig:
    """Tests for MCPClientConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MCPClientConfig()
        assert config.url == "http://localhost:8000/mcp"
        assert config.timeout == 60
        assert config.retry_count == 3
        assert config.retry_delay == 2.0
        assert config.headers == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MCPClientConfig(
            url="http://custom:9000/mcp",
            timeout=120,
            retry_count=5,
            retry_delay=1.0,
            headers={"X-Custom": "value"},
        )
        assert config.url == "http://custom:9000/mcp"
        assert config.timeout == 120
        assert config.retry_count == 5
        assert config.retry_delay == 1.0
        assert config.headers == {"X-Custom": "value"}

    def test_url_required(self):
        """Test that URL is required."""
        with pytest.raises(ValueError, match="URL is required"):
            MCPClientConfig(url="")

    def test_timeout_positive(self):
        """Test that timeout must be positive."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            MCPClientConfig(timeout=0)

    def test_retry_count_non_negative(self):
        """Test that retry count must be non-negative."""
        with pytest.raises(ValueError, match="Retry count must be non-negative"):
            MCPClientConfig(retry_count=-1)


class TestMCPToolResult:
    """Tests for MCPToolResult."""

    def test_success_result(self):
        """Test successful result."""
        result = MCPToolResult(
            success=True,
            data={"key": "value"},
            raw_content='{"key": "value"}',
        )
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        """Test error result."""
        result = MCPToolResult(
            success=False,
            error="Tool call failed",
        )
        assert result.success is False
        assert result.error == "Tool call failed"


class TestMCPResponse:
    """Tests for MCPResponse."""

    def test_success_response(self):
        """Test successful response."""
        response = MCPResponse(
            id="req-123",
            result={"content": [{"type": "text", "text": "data"}]},
        )
        assert response.is_success is True

    def test_error_response(self):
        """Test error response."""
        response = MCPResponse(
            id="req-123",
            error={"code": -32600, "message": "Invalid Request"},
        )
        assert response.is_success is False


class TestMCPHttpClient:
    """Tests for MCPHttpClient."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MCPClientConfig(
            url="http://localhost:8000/mcp",
            timeout=30,
            retry_count=2,
            retry_delay=0.1,  # Short delay for tests
        )

    @pytest.mark.asyncio
    async def test_context_manager_enter(self, config):
        """Test client initialization via context manager."""
        async with MCPHttpClient(config) as client:
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_context_manager_exit(self, config):
        """Test client cleanup via context manager."""
        client = MCPHttpClient(config)
        async with client:
            pass
        assert client._client is None

    @pytest.mark.asyncio
    async def test_call_tool_success(self, config):
        """Test successful tool call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "req-123",
            "result": {"content": [{"type": "text", "text": '{"success": true, "data": "value"}'}]},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            async with MCPHttpClient(config) as client:
                result = await client.call_tool("test_tool", {"arg": "value"})

                assert result.success is True
                assert result.data["success"] is True
                assert result.data["data"] == "value"

    @pytest.mark.asyncio
    async def test_call_tool_error_response(self, config):
        """Test tool call with error response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "req-123",
            "error": {"code": -32600, "message": "Invalid Request"},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            async with MCPHttpClient(config) as client:
                result = await client.call_tool("test_tool", {})

                assert result.success is False
                assert "Invalid Request" in result.error

    @pytest.mark.asyncio
    async def test_call_tool_retry_on_connect_error(self, config):
        """Test retry on connection error."""
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection failed")
            response = MagicMock()
            response.json.return_value = {
                "jsonrpc": "2.0",
                "id": "req-123",
                "result": {"content": [{"type": "text", "text": '{"success": true}'}]},
            }
            response.raise_for_status = MagicMock()
            return response

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            async with MCPHttpClient(config) as client:
                result = await client.call_tool("test_tool", {})
                assert result.success is True
                assert call_count == 3

    @pytest.mark.asyncio
    async def test_call_tool_max_retries_exceeded(self, config):
        """Test exception when max retries exceeded."""
        with patch.object(httpx.AsyncClient, "post", side_effect=httpx.ConnectError("Failed")):
            async with MCPHttpClient(config) as client:
                with pytest.raises(MCPConnectionError) as exc_info:
                    await client.call_tool("test_tool", {})

                assert "Failed to connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_tool_timeout_error(self, config):
        """Test handling of timeout errors."""
        with patch.object(
            httpx.AsyncClient,
            "post",
            side_effect=httpx.TimeoutException("Request timed out"),
        ):
            async with MCPHttpClient(config) as client:
                with pytest.raises(MCPConnectionError):
                    await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_not_initialized(self, config):
        """Test error when client not initialized."""
        client = MCPHttpClient(config)
        with pytest.raises(MCPConnectionError, match="not initialized"):
            await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_list_tools_success(self, config):
        """Test listing available tools."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "req-123",
            "result": {
                "tools": [
                    {"name": "tool1", "description": "desc1"},
                    {"name": "tool2", "description": "desc2"},
                ]
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            async with MCPHttpClient(config) as client:
                tools = await client.list_tools()

                assert len(tools) == 2
                assert tools[0]["name"] == "tool1"
                assert tools[1]["name"] == "tool2"

    @pytest.mark.asyncio
    async def test_list_tools_error(self, config):
        """Test list tools with error response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "req-123",
            "error": {"code": -32600, "message": "Error"},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            async with MCPHttpClient(config) as client:
                tools = await client.list_tools()
                assert tools == []

    @pytest.mark.asyncio
    async def test_ping_success(self, config):
        """Test successful ping."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "req-123",
            "result": {"tools": [{"name": "tool1"}]},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            async with MCPHttpClient(config) as client:
                result = await client.ping()
                assert result is True

    @pytest.mark.asyncio
    async def test_ping_failure(self, config):
        """Test failed ping."""
        with patch.object(httpx.AsyncClient, "post", side_effect=httpx.ConnectError("Failed")):
            async with MCPHttpClient(config) as client:
                result = await client.ping()
                assert result is False

    @pytest.mark.asyncio
    async def test_parse_non_json_response(self, config):
        """Test handling of non-JSON response content."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "req-123",
            "result": {"content": [{"type": "text", "text": "Plain text response"}]},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            async with MCPHttpClient(config) as client:
                result = await client.call_tool("test_tool", {})

                assert result.success is True
                assert result.data == {"raw": "Plain text response"}
                assert result.raw_content == "Plain text response"

    @pytest.mark.asyncio
    async def test_empty_response_content(self, config):
        """Test handling of empty response content."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "req-123",
            "result": {"content": []},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            async with MCPHttpClient(config) as client:
                result = await client.call_tool("test_tool", {})

                assert result.success is True
                assert result.data == {}
                assert result.raw_content == ""
