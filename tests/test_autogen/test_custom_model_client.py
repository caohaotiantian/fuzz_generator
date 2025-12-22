"""Tests for CustomModelClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fuzz_generator.agents.custom_model_client import CustomModelClient


class TestCustomModelClient:
    """Tests for CustomModelClient class."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return CustomModelClient(
            base_url="http://localhost:1234/v1",
            model="test-model",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1024,
            timeout=30,
        )

    def test_client_creation(self, client):
        """Test client can be created."""
        assert client is not None
        assert client._model == "test-model"
        assert client._temperature == 0.7

    def test_model_info(self, client):
        """Test model info property."""
        info = client.model_info
        # ModelInfo may be a dict or object depending on autogen version
        if isinstance(info, dict):
            assert "function_calling" in info
            assert "json_output" in info
        else:
            assert hasattr(info, "function_calling")
            assert hasattr(info, "json_output")

    def test_capabilities(self, client):
        """Test capabilities property."""
        caps = client.capabilities
        assert caps["function_calling"] is True
        assert caps["json_output"] is True

    def test_total_usage(self, client):
        """Test usage tracking."""
        usage = client.total_usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test client close."""
        await client.close()
        # Should not raise

    @pytest.mark.asyncio
    async def test_create_basic(self, client):
        """Test basic create call."""
        mock_response = {
            "id": "test-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_response),
                raise_for_status=MagicMock(),
            )

            from autogen_core.models import UserMessage

            messages = [UserMessage(content="Hello", source="user")]
            result = await client.create(messages)

            assert result.content == "Hello!"
            assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_create_with_tools(self, client):
        """Test create call with tool response."""
        mock_response = {
            "id": "test-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Tokyo"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_response),
                raise_for_status=MagicMock(),
            )

            from autogen_core.models import UserMessage

            messages = [UserMessage(content="What's the weather?", source="user")]

            # Define a mock tool
            def get_weather(location: str) -> str:
                return f"Weather in {location}"

            from autogen_core.tools import FunctionTool

            tools = [FunctionTool(get_weather, description="Get weather")]

            result = await client.create(messages, tools=tools)

            # Should have function calls
            assert hasattr(result, "content")

    def test_convert_message_user(self, client):
        """Test converting user message."""
        from autogen_core.models import UserMessage

        msg = UserMessage(content="Hello", source="user")
        converted = client._convert_message(msg)

        assert converted["role"] == "user"
        assert converted["content"] == "Hello"

    def test_convert_message_assistant(self, client):
        """Test converting assistant message."""
        from autogen_core.models import AssistantMessage

        msg = AssistantMessage(content="Hi there", source="assistant")
        converted = client._convert_message(msg)

        assert converted["role"] == "assistant"
        assert converted["content"] == "Hi there"

    def test_convert_message_system(self, client):
        """Test converting system message."""
        from autogen_core.models import SystemMessage

        msg = SystemMessage(content="You are a helpful assistant")
        converted = client._convert_message(msg)

        assert converted["role"] == "system"
        assert converted["content"] == "You are a helpful assistant"


class TestCustomModelClientEdgeCases:
    """Edge case tests for CustomModelClient."""

    def test_client_with_empty_api_key(self):
        """Test client with empty API key."""
        client = CustomModelClient(
            base_url="http://localhost:1234/v1",
            model="test-model",
            api_key="",
        )
        assert client is not None

    def test_client_with_custom_timeout(self):
        """Test client with custom timeout."""
        client = CustomModelClient(
            base_url="http://localhost:1234/v1",
            model="test-model",
            timeout=120,
        )
        assert client is not None

    @pytest.mark.asyncio
    async def test_close_is_safe(self):
        """Test that close can be called safely."""
        client = CustomModelClient(
            base_url="http://localhost:1234/v1",
            model="test-model",
        )
        # Should not raise
        await client.close()

    def test_model_name(self):
        """Test model name is stored correctly."""
        client = CustomModelClient(
            base_url="http://localhost:1234/v1",
            model="custom-model-name",
        )
        assert client._model == "custom-model-name"

