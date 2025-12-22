"""Tests for MCP HTTP client.

These tests primarily test the configuration and data structures.
Full integration tests require a running MCP server.
"""

from unittest.mock import MagicMock

import pytest

from fuzz_generator.exceptions import MCPConnectionError
from fuzz_generator.tools.mcp_client import (
    MCPClientConfig,
    MCPHttpClient,
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

    def test_default_values(self):
        """Test default values."""
        result = MCPToolResult(success=True)
        assert result.data == {}
        assert result.raw_content == ""
        assert result.error is None


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

    def test_client_creation(self, config):
        """Test client can be created."""
        client = MCPHttpClient(config)
        assert client is not None
        assert client.config == config
        assert client._session is None

    @pytest.mark.asyncio
    async def test_call_tool_not_initialized(self, config):
        """Test error when client not initialized."""
        client = MCPHttpClient(config)
        with pytest.raises(MCPConnectionError, match="not initialized"):
            await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_list_tools_not_initialized(self, config):
        """Test error when listing tools without initialization."""
        client = MCPHttpClient(config)
        with pytest.raises(MCPConnectionError, match="not initialized"):
            await client.list_tools()

    def test_client_initial_state(self, config):
        """Test client initial state before initialization."""
        client = MCPHttpClient(config)
        # Before entering context manager, session should be None
        assert client._session is None
        assert client._read is None
        assert client._write is None
        assert client._context_stack is None


class TestMCPHttpClientExtractResult:
    """Tests for result extraction."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MCPClientConfig()

    def test_extract_empty_result(self, config):
        """Test extracting empty result."""
        client = MCPHttpClient(config)

        mock_result = MagicMock()
        mock_result.content = []

        tool_result = client._extract_tool_result(mock_result)

        assert tool_result.success is True
        assert tool_result.data == {}
        assert tool_result.raw_content == ""

    def test_extract_json_result(self, config):
        """Test extracting JSON result."""
        client = MCPHttpClient(config)

        mock_content = MagicMock()
        mock_content.text = '{"success": true, "data": "value"}'

        mock_result = MagicMock()
        mock_result.content = [mock_content]

        tool_result = client._extract_tool_result(mock_result)

        assert tool_result.success is True
        assert tool_result.data == {"success": True, "data": "value"}
        assert tool_result.raw_content == '{"success": true, "data": "value"}'

    def test_extract_non_json_result(self, config):
        """Test extracting non-JSON result."""
        client = MCPHttpClient(config)

        mock_content = MagicMock()
        mock_content.text = "Plain text response"

        mock_result = MagicMock()
        mock_result.content = [mock_content]

        tool_result = client._extract_tool_result(mock_result)

        assert tool_result.success is True
        assert tool_result.data == {"raw": "Plain text response"}
        assert tool_result.raw_content == "Plain text response"

    def test_extract_no_content_attribute(self, config):
        """Test extracting result with no content attribute."""
        client = MCPHttpClient(config)

        mock_result = MagicMock(spec=[])  # No content attribute

        tool_result = client._extract_tool_result(mock_result)

        assert tool_result.success is True
        assert tool_result.data == {}


class TestMCPHttpClientIntegration:
    """Integration tests that require mocking MCP session."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MCPClientConfig(
            url="http://localhost:8000/mcp",
            timeout=30,
            retry_count=1,
            retry_delay=0.01,
        )

    @pytest.mark.asyncio
    async def test_ping_when_session_fails(self, config):
        """Test ping returns False when not connected."""
        client = MCPHttpClient(config)
        # Not initialized, should return False
        result = await client.ping()
        assert result is False


class TestMCPClientConfigValidation:
    """Tests for configuration validation."""

    def test_very_long_timeout(self):
        """Test configuration with very long timeout."""
        config = MCPClientConfig(timeout=3600)
        assert config.timeout == 3600

    def test_zero_retry_count(self):
        """Test configuration with zero retries."""
        config = MCPClientConfig(retry_count=0)
        assert config.retry_count == 0

    def test_custom_headers(self):
        """Test configuration with custom headers."""
        config = MCPClientConfig(
            headers={
                "Authorization": "Bearer token",
                "X-Custom-Header": "value",
            }
        )
        assert "Authorization" in config.headers
        assert "X-Custom-Header" in config.headers
