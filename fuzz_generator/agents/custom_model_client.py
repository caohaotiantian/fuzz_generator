"""Custom Model Client using httpx for direct API calls.

This module provides a custom ChatCompletionClient implementation that
directly calls OpenAI-compatible APIs using httpx, avoiding issues with
the openai library (proxy handling, etc.).

This client is compatible with AutoGen's agent framework.
"""

from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any, Literal

import httpx
from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel

from fuzz_generator.utils.logger import get_logger

logger = get_logger(__name__)


class CustomModelClient(ChatCompletionClient):
    """Custom ChatCompletionClient using httpx for direct API calls.

    This client implements AutoGen's ChatCompletionClient interface
    but uses httpx directly instead of the openai library, which
    can have proxy issues with local servers.

    Usage:
        client = CustomModelClient(
            base_url="http://localhost:1234/v1",
            model="openai/gpt-oss-20b",
            api_key="lm-studio",
        )

        # Use with AutoGen agents
        agent = AssistantAgent(
            name="MyAgent",
            model_client=client,
            system_message="...",
        )
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "dummy",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120,
    ):
        """Initialize custom model client.

        Args:
            base_url: API base URL (e.g., "http://localhost:1234/v1")
            model: Model name
            api_key: API key (can be dummy for local servers)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

        # Usage tracking
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

        # HTTP client - use explicit no-proxy transport for local services
        # Create transport that doesn't use any proxy
        transport = httpx.AsyncHTTPTransport()

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

        # Model info
        self._model_info = ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family="custom",
            structured_output=False,
        )

    @property
    def model_info(self) -> ModelInfo:
        """Get model information."""
        return self._model_info

    @property
    def capabilities(self) -> ModelCapabilities:
        """Get model capabilities."""
        return ModelCapabilities(
            vision=False,
            function_calling=True,
            json_output=True,
        )

    def _convert_message(self, msg: LLMMessage) -> dict[str, Any]:
        """Convert LLMMessage to API format.

        Args:
            msg: AutoGen message

        Returns:
            API message dict
        """
        # Handle different message types
        if hasattr(msg, "content"):
            # Get role - map AutoGen types to OpenAI roles
            if hasattr(msg, "type"):
                msg_type = msg.type
                if msg_type == "SystemMessage":
                    role = "system"
                elif msg_type == "AssistantMessage":
                    role = "assistant"
                else:
                    role = "user"
            elif hasattr(msg, "role"):
                role = msg.role
            else:
                role = "user"

            content = msg.content

            result: dict[str, Any] = {"role": role}

            # Handle content (could be string or list)
            if isinstance(content, str):
                # Check if content is a stringified message object
                # Pattern: "content='...' source='...' type='...'"
                if content.startswith("content='") and "type='" in content:
                    # Parse the actual content from stringified message
                    import re

                    match = re.search(r"content='(.*?)'\s+(?:thought=|source=)", content, re.DOTALL)
                    if match:
                        actual_content = match.group(1)
                        # Unescape single quotes
                        actual_content = actual_content.replace("\\'", "'")
                        result["content"] = actual_content
                    else:
                        result["content"] = content
                else:
                    result["content"] = content
            elif isinstance(content, list):
                # Convert list content (could contain FunctionCall, etc.)
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif hasattr(item, "content"):
                        text_parts.append(str(item.content))
                    elif hasattr(item, "text"):
                        text_parts.append(str(item.text))
                    else:
                        text_parts.append(str(item))
                result["content"] = "\n".join(text_parts) if text_parts else ""
            else:
                result["content"] = str(content) if content else ""

            # Add name if present (for multi-agent scenarios)
            if hasattr(msg, "source") and msg.source:
                result["name"] = str(msg.source)
            elif hasattr(msg, "name") and msg.name:
                result["name"] = msg.name

            return result
        else:
            # Fallback for unknown message types
            return {"role": "user", "content": str(msg)}

    def _convert_tools(self, tools: Sequence[Tool | ToolSchema]) -> list[dict[str, Any]]:
        """Convert tools to API format.

        Args:
            tools: AutoGen tools

        Returns:
            API tools list in OpenAI format
        """
        result = []
        for tool in tools:
            # Check if it's already in the correct format
            if isinstance(tool, dict):
                if "type" in tool and "function" in tool:
                    result.append(tool)
                elif "name" in tool:
                    # Convert flat dict to nested format
                    result.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.get("name", "unknown"),
                                "description": tool.get("description", ""),
                                "parameters": tool.get("parameters", {}),
                            },
                        }
                    )
            elif hasattr(tool, "schema"):
                # Tool object with schema property
                schema = tool.schema if isinstance(tool.schema, dict) else {}
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": schema.get("name", getattr(tool, "name", "unknown")),
                            "description": schema.get("description", ""),
                            "parameters": schema.get("parameters", {}),
                        },
                    }
                )
            elif hasattr(tool, "name"):
                # ToolSchema-like object with direct attributes
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": getattr(tool, "name", "unknown"),
                            "description": getattr(tool, "description", ""),
                            "parameters": getattr(tool, "parameters", {}),
                        },
                    }
                )
        return result

    def _parse_response(self, data: dict[str, Any]) -> CreateResult:
        """Parse API response to CreateResult.

        Args:
            data: Raw API response

        Returns:
            CreateResult
        """
        if "error" in data:
            logger.error(f"API error: {data['error']}")
            return CreateResult(
                finish_reason="unknown",
                content=f"Error: {data['error']}",
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False,
            )

        choices = data.get("choices", [])
        if not choices:
            return CreateResult(
                finish_reason="unknown",
                content="",
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False,
            )

        choice = choices[0]
        message = choice.get("message", {})

        # Parse content
        content: str | list[FunctionCall] = message.get("content", "") or ""

        # Parse tool calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            function_calls = []
            for tc in tool_calls:
                if tc.get("type") == "function":
                    func = tc.get("function", {})
                    function_calls.append(
                        FunctionCall(
                            id=tc.get("id", ""),
                            name=func.get("name", ""),
                            arguments=func.get("arguments", "{}"),
                        )
                    )
            if function_calls:
                content = function_calls

        # Parse finish reason
        finish_reason_map = {
            "stop": "stop",
            "length": "length",
            "tool_calls": "function_calls",
            "function_call": "function_calls",
            "content_filter": "content_filter",
        }
        raw_finish = choice.get("finish_reason", "stop")
        finish_reason = finish_reason_map.get(raw_finish, "unknown")

        # Parse usage
        usage_data = data.get("usage", {})
        usage = RequestUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )

        # Update totals
        self._total_prompt_tokens += usage.prompt_tokens
        self._total_completion_tokens += usage.completion_tokens

        return CreateResult(
            finish_reason=finish_reason,
            content=content,
            usage=usage,
            cached=False,
            thought=message.get("reasoning"),  # Some models return reasoning
        )

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        """Create a chat completion.

        Args:
            messages: Chat messages
            tools: Available tools
            tool_choice: Tool selection mode
            json_output: Whether to request JSON output
            extra_create_args: Extra API arguments
            cancellation_token: Cancellation token

        Returns:
            CreateResult with response
        """
        # Build request body
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [self._convert_message(m) for m in messages],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        # Add tools if provided
        if tools:
            body["tools"] = self._convert_tools(tools)
            if isinstance(tool_choice, str):
                body["tool_choice"] = tool_choice
            elif hasattr(tool_choice, "name"):
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice.name},
                }

        # Add JSON mode if requested
        if json_output:
            body["response_format"] = {"type": "json_object"}

        # Merge extra args
        body.update(extra_create_args)

        # Make request
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        logger.debug(f"LLM request: {len(messages)} messages, {len(tools)} tools")
        logger.debug(f"Request body: {body}")

        try:
            response = await self._client.post(url, headers=headers, json=body)
            if response.status_code != 200:
                logger.error(f"Response body: {response.text}")
            response.raise_for_status()
            data = response.json()

            result = self._parse_response(data)
            logger.debug(
                f"LLM response: finish={result.finish_reason}, "
                f"content_type={type(result.content).__name__}"
            )
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code}")
            return CreateResult(
                finish_reason="unknown",
                content=f"HTTP Error: {e.response.status_code}",
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False,
            )
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return CreateResult(
                finish_reason="unknown",
                content=f"Error: {e}",
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False,
            )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        """Create streaming chat completion.

        For simplicity, this falls back to non-streaming.
        """
        result = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield result

    def count_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
    ) -> int:
        """Estimate token count.

        This is a rough estimate; actual count depends on the tokenizer.
        """
        total = 0
        for msg in messages:
            content = getattr(msg, "content", str(msg))
            if isinstance(content, str):
                # Rough estimate: ~4 chars per token
                total += len(content) // 4
            elif isinstance(content, list):
                for item in content:
                    total += len(str(item)) // 4
        return total

    def remaining_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
    ) -> int:
        """Get remaining tokens available."""
        used = self.count_tokens(messages, tools=tools)
        return max(0, self._max_tokens - used)

    def actual_usage(self) -> RequestUsage:
        """Get actual usage from last request."""
        return RequestUsage(
            prompt_tokens=self._total_prompt_tokens,
            completion_tokens=self._total_completion_tokens,
        )

    def total_usage(self) -> RequestUsage:
        """Get total usage across all requests."""
        return RequestUsage(
            prompt_tokens=self._total_prompt_tokens,
            completion_tokens=self._total_completion_tokens,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
