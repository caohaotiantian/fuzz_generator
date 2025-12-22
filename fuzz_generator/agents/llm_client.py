"""Simple LLM client using httpx for direct API calls.

This module provides a simple LLM client that directly calls OpenAI-compatible
APIs using httpx, avoiding issues with the openai library (proxy handling, etc.).
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from fuzz_generator.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client.

    Attributes:
        base_url: API base URL (e.g., "http://localhost:1234/v1")
        api_key: API key (can be dummy for local servers)
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
    """

    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120


@dataclass
class ChatMessage:
    """A chat message.

    Attributes:
        role: Message role (system, user, assistant)
        content: Message content
        name: Optional name for the message sender
    """

    role: str
    content: str
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to API dict format."""
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ToolCall:
    """A tool/function call from the model.

    Attributes:
        id: Tool call ID
        name: Function name
        arguments: Function arguments as JSON string or dict
    """

    id: str
    name: str
    arguments: str | dict[str, Any]

    def get_arguments_dict(self) -> dict[str, Any]:
        """Get arguments as a dictionary."""
        if isinstance(self.arguments, str):
            try:
                return json.loads(self.arguments)
            except json.JSONDecodeError:
                return {}
        return self.arguments


@dataclass
class LLMResponse:
    """Response from LLM.

    Attributes:
        content: Text content of the response
        tool_calls: List of tool calls if any
        finish_reason: Why the model stopped (stop, tool_calls, length, etc.)
        raw_response: Raw API response
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw_response: dict[str, Any] = field(default_factory=dict)


class SimpleLLMClient:
    """Simple LLM client using httpx.

    This client directly calls OpenAI-compatible APIs without using
    the openai library, which can have proxy issues.

    Usage:
        config = LLMConfig(base_url="http://localhost:1234/v1", model="my-model")
        client = SimpleLLMClient(config)

        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello!"),
        ]
        response = await client.chat(messages)
        print(response.content)
    """

    def __init__(self, config: LLMConfig):
        """Initialize client.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SimpleLLMClient":
        """Enter async context."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            # Explicitly disable proxy for local services
            proxy=None,
        )
        return self

    async def __aexit__(self, *args) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

    def _build_request_body(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build API request body.

        Args:
            messages: Chat messages
            tools: Optional tool definitions

        Returns:
            Request body dict
        """
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        return body

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse API response.

        Args:
            data: Raw API response

        Returns:
            Parsed LLMResponse
        """
        if "error" in data:
            logger.error(f"LLM API error: {data['error']}")
            return LLMResponse(
                content=f"Error: {data['error']}",
                finish_reason="error",
                raw_response=data,
            )

        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(
                content="",
                finish_reason="error",
                raw_response=data,
            )

        choice = choices[0]
        message = choice.get("message", {})

        # Extract content
        content = message.get("content", "") or ""

        # Extract tool calls
        tool_calls = []
        raw_tool_calls = message.get("tool_calls", [])
        for tc in raw_tool_calls:
            if tc.get("type") == "function":
                func = tc.get("function", {})
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=func.get("name", ""),
                        arguments=func.get("arguments", "{}"),
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            raw_response=data,
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send chat completion request.

        Args:
            messages: Chat messages
            tools: Optional tool definitions

        Returns:
            LLM response

        Raises:
            Exception: If request fails
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = self._build_headers()
        body = self._build_request_body(messages, tools)

        logger.debug(f"LLM request to {url}")
        logger.debug(f"Messages: {len(messages)}, Tools: {len(tools) if tools else 0}")

        try:
            response = await self._client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()

            result = self._parse_response(data)
            logger.debug(
                f"LLM response: finish_reason={result.finish_reason}, "
                f"content_len={len(result.content)}, tool_calls={len(result.tool_calls)}"
            )
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise

    async def chat_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        tool_executor: "ToolExecutor",
        max_iterations: int = 10,
    ) -> LLMResponse:
        """Chat with automatic tool execution.

        Args:
            messages: Initial chat messages
            tools: Tool definitions
            tool_executor: Function to execute tools
            max_iterations: Maximum tool execution iterations

        Returns:
            Final LLM response
        """
        current_messages = list(messages)

        for i in range(max_iterations):
            response = await self.chat(current_messages, tools)

            # If no tool calls, return response
            if not response.tool_calls:
                return response

            # Add assistant message with tool calls
            current_messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.content or "",
                )
            )

            # Execute tools and add results
            for tool_call in response.tool_calls:
                result = await tool_executor(tool_call.name, tool_call.get_arguments_dict())
                current_messages.append(
                    ChatMessage(
                        role="tool",
                        content=str(result),
                        name=tool_call.name,
                    )
                )

            logger.debug(f"Tool iteration {i + 1}: executed {len(response.tool_calls)} tools")

        # Max iterations reached
        logger.warning(f"Max tool iterations ({max_iterations}) reached")
        return await self.chat(current_messages, tools=None)


# Type alias for tool executor function
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]
