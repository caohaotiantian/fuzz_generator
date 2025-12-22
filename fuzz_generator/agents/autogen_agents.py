"""AutoGen-based Agent implementation for two-phase analysis workflow.

This module implements a simplified two-agent system:
- AnalysisAgent: Combined code analysis + context building with iterative tool calls
- ModelGenerator: DataModel generation based on analysis results

Design Reference: docs/AGENT_OPTIMIZATION.md

The workflow is sequential and deterministic:
1. AnalysisAgent iteratively collects code context (multiple tool calls)
2. ModelGenerator generates XML based on analysis results
"""

import functools
import json
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from fuzz_generator.agents.base import PromptTemplate
from fuzz_generator.agents.custom_model_client import CustomModelClient
from fuzz_generator.config import Settings
from fuzz_generator.models import AnalysisTask, TaskResult
from fuzz_generator.tools.mcp_client import MCPHttpClient
from fuzz_generator.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Prompt Loader
# ============================================================================


class PromptLoader:
    """Load agent prompts from YAML files.

    Loads prompts from config/defaults/prompts/ directory.
    """

    def __init__(self, prompts_dir: Path | None = None):
        """Initialize prompt loader.

        Args:
            prompts_dir: Directory containing prompt YAML files.
        """
        if prompts_dir is None:
            package_dir = Path(__file__).parent.parent
            prompts_dir = package_dir / "config" / "defaults" / "prompts"

        self.prompts_dir = prompts_dir
        self._cache: dict[str, PromptTemplate] = {}

    def load(self, agent_name: str) -> PromptTemplate:
        """Load prompt template for an agent."""
        if agent_name in self._cache:
            return self._cache[agent_name]

        yaml_file = self.prompts_dir / f"{agent_name}.yaml"

        if yaml_file.exists():
            template = PromptTemplate(yaml_file)
            self._cache[agent_name] = template
            logger.debug(f"Loaded prompt template from: {yaml_file}")
            return template
        else:
            logger.warning(f"Prompt file not found: {yaml_file}")
            return PromptTemplate()

    def get_system_prompt(
        self,
        agent_name: str,
        custom_knowledge: str = "",
        **kwargs: Any,
    ) -> str:
        """Get rendered system prompt for an agent."""
        template = self.load(agent_name)
        return template.render_system_prompt(
            custom_knowledge=custom_knowledge,
            **kwargs,
        )


# ============================================================================
# Default Fallback Prompts
# ============================================================================

DEFAULT_ANALYSIS_AGENT_PROMPT = """你是代码分析专家。分析目标函数，收集完整的上下文信息用于生成 Fuzz 测试数据模型。

## 可用工具

- get_function_code(function_name): 获取函数源代码
- list_functions(file_path): 列出文件中的所有函数
- search_code(pattern): 搜索代码模式
- track_dataflow(source_pattern): 追踪数据流
- get_callees(function_name): 获取被调用的函数
- get_callers(function_name): 获取调用者
- get_control_flow_graph(function_name): 获取控制流图

## 工作流程

1. 首先调用 get_function_code 获取目标函数代码
2. 从代码中识别参数列表
3. 对关键参数调用 track_dataflow 追踪数据流
4. 调用 get_callees 获取被调用函数
5. 调用 get_control_flow_graph 获取控制流

## 规则

- **必须** 先调用工具获取代码，禁止猜测
- 工具出错时记录错误继续，不要讨论
- 分析完成后输出 JSON，以 ANALYSIS_COMPLETE 结束

## 输出格式

```json
{
  "status": "success",
  "function": {"name": "函数名", "return_type": "返回类型", "source_code": "代码"},
  "parameters": [{"name": "参数名", "type": "类型", "data_flow": [], "passed_to": []}],
  "callees": [{"name": "函数名", "handles_parameters": []}],
  "control_flow": {"conditions": [], "loops": []},
  "errors": []
}
```

ANALYSIS_COMPLETE
"""

DEFAULT_MODEL_GENERATOR_PROMPT = """你是 Fuzz 测试数据模型生成专家。根据代码分析结果生成 XML DataModel。

## DataModel 元素

- <String>: 字符串 (name, value, maxLength)
- <Number>: 数值 (name, size, signed, endian)
- <Blob>: 二进制数据 (name, length)
- <Block>: 结构块 (name, ref)
- <Choice>: 选择结构

## 规则

1. 每个参数对应一个元素
2. char*/const char* → <String>
3. int/long → <Number size="32">
4. void* + length → <Blob length="length_field">

## 输出格式

直接输出 XML，以 MODEL_COMPLETE 结束：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<DataModel name="模型名">
  <!-- 元素 -->
</DataModel>
```

MODEL_COMPLETE

只输出 XML，不要解释。
"""


# ============================================================================
# MCP Tool Functions
# ============================================================================


def create_analysis_tools(
    mcp_client: MCPHttpClient,
    project_name: str,
    source_file: str | None = None,
    project_path: Any = None,
    cache_wrapper: Callable[[Callable, str], Callable] | None = None,
) -> list:
    """Create all analysis tool functions for AnalysisAgent.

    Args:
        mcp_client: MCP HTTP client instance
        project_name: Active project name in Joern
        source_file: Optional source file to narrow function search
        project_path: Optional project root path for reading complete source files
        cache_wrapper: Optional function to wrap tools with caching

    Returns:
        List of tool functions
    """

    async def get_function_code(function_name: str) -> str:
        """Get the source code of a function.

        Args:
            function_name: Name of the function to retrieve

        Returns:
            Function source code as string
        """
        from fuzz_generator.tools.query_tools import get_function_code as _get_func

        # Pass source_file to narrow search if specified, and project_path for complete code
        result = await _get_func(
            mcp_client,
            function_name,
            project_name,
            file_name=source_file,
            project_path=project_path,
        )
        if result.success:
            file_info = f" (from {result.file})" if result.file else ""
            return f"函数 {function_name} 的源代码{file_info}:\n{result.code}"
        return f"Error: {result.error}"

    async def list_functions(file_path: str | None = None) -> str:
        """List all functions in the project or a specific file.

        Args:
            file_path: Optional file path to filter functions

        Returns:
            JSON string of function list
        """
        from fuzz_generator.tools.query_tools import list_functions as _list_funcs

        result = await _list_funcs(mcp_client, project_name, file_name=file_path)
        if result.success:
            funcs = [
                {"name": f.name, "file": f.file, "line": f.line_number} for f in result.functions
            ]
            return f"找到 {len(funcs)} 个函数:\n{json.dumps(funcs, indent=2, ensure_ascii=False)}"
        return f"Error: {result.error}"

    async def search_code(pattern: str) -> str:
        """Search for code patterns in the project.

        Args:
            pattern: Search pattern (regex supported)

        Returns:
            JSON string of search results
        """
        from dataclasses import asdict

        from fuzz_generator.tools.query_tools import search_code as _search

        result = await _search(mcp_client, project_name, pattern)
        if result.success:
            matches = [asdict(m) for m in result.matches]
            return f"搜索结果 ({result.count} 个匹配):\n{json.dumps(matches, indent=2, ensure_ascii=False)}"
        return f"Error: {result.error}"

    async def track_dataflow(source_method: str, sink_method: str) -> str:
        """Track data flow from source method to sink method.

        Args:
            source_method: Source method name (e.g., function parameter, "gets", "scanf")
            sink_method: Sink method name (e.g., "strcpy", "printf", "system")

        Returns:
            JSON string of dataflow paths
        """
        from dataclasses import asdict

        from fuzz_generator.tools.analysis_tools import track_dataflow as _track

        # Note: joern_mcp requires both source_method and sink_method
        result = await _track(
            mcp_client,
            project_name,
            source_method=source_method,
            sink_method=sink_method,
        )
        if result.success:
            flows = [asdict(f) for f in result.flows]
            return f"数据流追踪结果 ({source_method} -> {sink_method}):\n{json.dumps(flows, indent=2, ensure_ascii=False)}"
        return f"Error: {result.error}"

    async def get_callers(function_name: str) -> str:
        """Get functions that call the specified function.

        Args:
            function_name: Target function name

        Returns:
            JSON string of caller information
        """
        from dataclasses import asdict

        from fuzz_generator.tools.analysis_tools import get_callers as _get_callers

        result = await _get_callers(mcp_client, project_name, function_name)
        if result.success:
            callers = [asdict(c) for c in result.callers]
            return (
                f"调用 {function_name} 的函数:\n{json.dumps(callers, indent=2, ensure_ascii=False)}"
            )
        return f"Error: {result.error}"

    async def get_callees(function_name: str) -> str:
        """Get functions called by the specified function.

        Args:
            function_name: Target function name

        Returns:
            JSON string of callee information
        """
        from dataclasses import asdict

        from fuzz_generator.tools.analysis_tools import get_callees as _get_callees

        result = await _get_callees(mcp_client, project_name, function_name)
        if result.success:
            callees = [asdict(c) for c in result.callees]
            return (
                f"{function_name} 调用的函数:\n{json.dumps(callees, indent=2, ensure_ascii=False)}"
            )
        return f"Error: {result.error}"

    async def get_control_flow_graph(function_name: str) -> str:
        """Get the control flow graph of a function.

        Args:
            function_name: Target function name

        Returns:
            JSON string of CFG information
        """
        from dataclasses import asdict

        from fuzz_generator.tools.analysis_tools import (
            get_control_flow_graph as _get_cfg,
        )

        result = await _get_cfg(mcp_client, project_name, function_name)
        if result.success:
            cfg = asdict(result.cfg)
            return f"{function_name} 的控制流图:\n{json.dumps(cfg, indent=2, ensure_ascii=False)}"
        return f"Error: {result.error}"

    # Apply cache wrapper if provided (Optimization 1)
    tools = [
        get_function_code,
        list_functions,
        search_code,
        track_dataflow,
        get_callers,
        get_callees,
        get_control_flow_graph,
    ]

    if cache_wrapper:
        # Wrap each tool with caching
        # functools.wraps in cache_wrapper preserves all metadata
        wrapped_tools = [cache_wrapper(tool, tool.__name__) for tool in tools]
        return wrapped_tools

    return tools


# ============================================================================
# Conversation Recorder
# ============================================================================


class ConversationRecorder:
    """Records agent conversations for debugging.

    Records messages with structured tool call information including:
    - Agent thinking/reasoning
    - Tool calls with arguments
    - Tool results
    - Final outputs
    """

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path
        self.messages: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []  # Structured tool call log

    def record(self, agent_name: str, content: str, role: str = "assistant") -> None:
        """Record a message."""
        self.messages.append(
            {
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name,
                "role": role,
                "content": content,  # No truncation - full content needed for debugging
            }
        )

    def record_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        call_id: str | None = None,
    ) -> None:
        """Record a tool call with structured arguments."""
        self.tool_calls.append(
            {
                "timestamp": datetime.now().isoformat(),
                "type": "tool_call",
                "agent": agent_name,
                "call_id": call_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": None,  # Will be filled by record_tool_result
            }
        )

    def record_tool_result(
        self,
        call_id: str,
        result: str,
        is_error: bool = False,
    ) -> None:
        """Record the result of a tool call."""
        # Find matching tool call and update result
        for tc in reversed(self.tool_calls):
            if tc.get("call_id") == call_id:
                tc["result"] = result  # No truncation - full result needed for LLM analysis
                tc["is_error"] = is_error
                tc["completed_at"] = datetime.now().isoformat()
                break

    def get_messages(self) -> list[dict[str, Any]]:
        return self.messages

    def get_tool_calls(self) -> list[dict[str, Any]]:
        return self.tool_calls

    async def save(self, task_id: str) -> None:
        """Save conversation and tool calls to files."""
        if self.storage_path is None:
            return

        intermediate_dir = self.storage_path / "results" / task_id / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Save conversations
        conversations_file = intermediate_dir / "agent_conversations.json"
        conversations_file.write_text(json.dumps(self.messages, indent=2, ensure_ascii=False))
        logger.debug(f"Saved conversation to {conversations_file}")

        # Save structured tool calls
        tool_calls_file = intermediate_dir / "tool_calls.json"
        tool_calls_file.write_text(json.dumps(self.tool_calls, indent=2, ensure_ascii=False))
        logger.debug(f"Saved tool calls to {tool_calls_file}")


# ============================================================================
# Output Validators
# ============================================================================


class OutputValidator:
    """Validates agent outputs."""

    @staticmethod
    def extract_json(text: str) -> dict[str, Any] | None:
        """Extract JSON from text."""
        # Try to find JSON block
        json_patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # Markdown code block
            r"```\s*([\s\S]*?)\s*```",  # Generic code block
            r"(\{[\s\S]*\})",  # Raw JSON
        ]

        for pattern in json_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        return None

    @staticmethod
    def extract_xml(text: str) -> str | None:
        """Extract XML DataModel from text."""
        # Look for DataModel block
        xml_match = re.search(r"<DataModel[\s\S]*?</DataModel>", text)
        if xml_match:
            return xml_match.group()
        return None

    @staticmethod
    def validate_analysis_result(data: dict) -> bool:
        """Validate analysis result structure (supports both old and new format)."""
        has_status = "status" in data
        has_function = "function_info" in data or "function" in data
        has_parameters = "parameters" in data
        return has_status and has_function and has_parameters


# ============================================================================
# Two-Phase Workflow
# ============================================================================


class TwoPhaseWorkflow:
    """Two-phase analysis workflow.

    Phase 1: AnalysisAgent iteratively collects code context
    Phase 2: ModelGenerator creates XML DataModel

    Design Reference: docs/AGENT_OPTIMIZATION.md Section 4
    """

    def __init__(
        self,
        settings: Settings,
        mcp_client: MCPHttpClient,
        project_name: str,
        custom_knowledge: str = "",
        storage_path: Path | None = None,
        project_path: Any = None,
    ):
        """Initialize workflow.

        Args:
            settings: Application settings
            mcp_client: MCP HTTP client
            project_name: Active project name
            custom_knowledge: Custom background knowledge
            storage_path: Path for storing intermediate results
            project_path: Project root path for reading complete source files
        """
        self.settings = settings
        self.mcp_client = mcp_client
        self.project_name = project_name
        self.custom_knowledge = custom_knowledge
        self.storage_path = storage_path
        self.project_path = project_path

        # Load prompts
        self.prompt_loader = PromptLoader()

        # Create LLM client
        self.model_client = CustomModelClient(
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=settings.llm.timeout,
        )

        # Tools will be created in _run_analysis_phase with task-specific source_file
        # self.tools = create_analysis_tools(mcp_client, project_name)

        # Tool call cache and tracking (Optimization 1)
        self.tool_call_cache: dict[str, Any] = {}  # Cache for tool results
        self.tool_call_counts: dict[str, int] = {}  # Track call counts per tool+params
        self.max_same_tool_calls = 3  # Maximum repeated calls with same params

        # Conversation recorder
        self.recorder = ConversationRecorder(storage_path)

        # Validator
        self.validator = OutputValidator()

    async def close(self) -> None:
        """Close resources (model client)."""
        if self.model_client:
            await self.model_client.close()

    async def __aenter__(self) -> "TwoPhaseWorkflow":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.close()

    def _validate_analysis_result(self, result: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate analysis result completeness and quality (Optimization 4).

        Args:
            result: Analysis result dictionary

        Returns:
            Tuple of (is_valid, warnings_list)
        """
        warnings = []

        # Check required top-level fields
        if "function_info" not in result:
            warnings.append("缺少 function_info 字段")
            return False, warnings

        if "parameters" not in result:
            warnings.append("缺少 parameters 字段")

        if "status" not in result:
            warnings.append("缺少 status 字段")

        # Check function_info completeness
        func_info = result.get("function_info", {})

        if not func_info.get("name"):
            warnings.append("function_info.name 为空")

        source_code_status = func_info.get("source_code_status", "unavailable")
        if source_code_status == "unavailable":
            warnings.append("未获取到源代码")
        elif source_code_status == "partial":
            warnings.append("源代码不完整")

        if not func_info.get("source_code"):
            warnings.append("source_code 字段为空")

        # Check parameters
        params = result.get("parameters", [])
        if not params:
            warnings.append("未识别任何函数参数")
        else:
            for i, param in enumerate(params):
                if not param.get("name"):
                    warnings.append(f"参数 {i} 缺少 name 字段")
                if not param.get("type"):
                    warnings.append(f"参数 '{param.get('name', i)}' 缺少 type 字段")
                if "data_flow" not in param:
                    warnings.append(f"参数 '{param.get('name', i)}' 缺少数据流分析")

        # Check callees
        if "callees" not in result or not result["callees"]:
            warnings.append("未分析函数调用关系")

        # Check confidence
        confidence = result.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            warnings.append("confidence 字段类型错误")
            confidence = 0.0

        if confidence < 0.3:
            warnings.append(f"分析置信度过低: {confidence}")
        elif confidence < 0.6:
            warnings.append(f"分析置信度较低: {confidence}")

        # Determine overall validity
        # Valid if:
        # 1. No critical errors (has function_info with name)
        # 2. Confidence >= 0.5 OR has parameters
        is_valid = (
            func_info.get("name")
            and (confidence >= 0.5 or params)
            and result.get("status") != "failed"
        )

        return is_valid, warnings

    def _create_cached_tool(
        self,
        tool_func: Callable[..., Awaitable[str]],
        tool_name: str,
    ) -> Callable[..., Awaitable[str]]:
        """Create a cached version of a tool function.

        Implements caching and call count limiting to prevent redundant tool calls.

        Args:
            tool_func: Original async tool function
            tool_name: Name of the tool for logging

        Returns:
            Wrapped tool function with caching and limiting
        """

        @functools.wraps(tool_func)
        async def cached_tool(*args: Any, **kwargs: Any) -> str:
            # Generate cache key from tool name and arguments
            # Convert args to dict based on function signature for consistent caching
            cache_key = f"{tool_name}:{json.dumps(kwargs, sort_keys=True)}"

            # Check cache
            if cache_key in self.tool_call_cache:
                logger.info(f"🔄 使用缓存结果: {tool_name} (参数: {kwargs})")
                return self.tool_call_cache[cache_key]

            # Track call count for this specific tool+params combination
            self.tool_call_counts[cache_key] = self.tool_call_counts.get(cache_key, 0) + 1
            call_count = self.tool_call_counts[cache_key]

            # Check if exceeded maximum calls
            if call_count > self.max_same_tool_calls:
                error_msg = (
                    f"⚠️ 工具 '{tool_name}' 已被调用 {call_count} 次（参数: {kwargs}），"
                    f"超过限制 ({self.max_same_tool_calls})。\n"
                    f"建议：\n"
                    f"  1. 如果结果看起来不完整，尝试使用其他工具组合获取信息\n"
                    f"  2. 使用 search_code 搜索相关代码片段\n"
                    f"  3. 使用 get_callees 了解函数调用关系\n"
                    f"  4. 基于已有部分信息继续分析，在结果中标注数据不完整"
                )
                logger.warning(error_msg)
                return error_msg

            # Log warning if approaching limit
            if call_count == self.max_same_tool_calls:
                logger.warning(
                    f"⚠️ 工具 '{tool_name}' 即将达到调用限制 "
                    f"({call_count}/{self.max_same_tool_calls})"
                )

            # Execute tool
            try:
                result = await tool_func(*args, **kwargs)

                # Cache successful result
                self.tool_call_cache[cache_key] = result

                logger.debug(
                    f"✅ 工具调用成功: {tool_name} "
                    f"(第 {call_count} 次，结果长度: {len(result)} 字符)"
                )

                return result

            except Exception as e:
                error_msg = f"工具调用失败: {tool_name}\n错误: {str(e)}"
                logger.error(error_msg)
                # Don't cache errors
                return error_msg

        return cached_tool

    def _get_prompt(self, agent_name: str, fallback: str) -> str:
        """Get prompt from YAML or fallback."""
        template = self.prompt_loader.load(agent_name)
        if template.system_prompt:
            prompt = template.render_system_prompt(custom_knowledge=self.custom_knowledge)
            return prompt

        # Use fallback
        if "{custom_knowledge}" in fallback and self.custom_knowledge:
            return fallback.format(custom_knowledge=f"\n## 背景知识\n{self.custom_knowledge}")
        return fallback.replace("{custom_knowledge}", "")

    async def run(self, task: AnalysisTask, verbose: bool = True) -> TaskResult:
        """Run the two-phase analysis workflow.

        Args:
            task: Analysis task to execute
            verbose: Whether to show conversation output

        Returns:
            TaskResult with analysis results
        """
        logger.info(f"Starting two-phase analysis for: {task.function_name}")

        try:
            # Phase 1: Analysis
            logger.info("Phase 1: Running AnalysisAgent...")
            analysis_result = await self._run_analysis_phase(task, verbose)

            if not analysis_result:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    xml_content=None,
                    errors=["Analysis phase failed to produce valid results"],
                    warnings=[],
                )

            # Save intermediate analysis result
            await self._save_analysis_result(task, analysis_result)

            # Phase 2: Model Generation
            logger.info("Phase 2: Running ModelGenerator...")
            xml_content = await self._run_generation_phase(task, analysis_result, verbose)

            # Save conversation
            await self.recorder.save(task.task_id)

            if xml_content:
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    xml_content=self._wrap_xml(xml_content),
                    errors=[],
                    warnings=[],
                )
            else:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    xml_content=None,
                    errors=["Model generation failed to produce valid XML"],
                    warnings=[],
                )

        except Exception as e:
            logger.exception(f"Workflow error: {e}")
            await self.recorder.save(task.task_id)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                xml_content=None,
                errors=[str(e)],
                warnings=[],
            )

    async def _run_analysis_phase(self, task: AnalysisTask, verbose: bool) -> dict[str, Any] | None:
        """Run the analysis phase with AnalysisAgent.

        The agent can make multiple tool calls iteratively.
        """
        # Get prompt
        system_prompt = self._get_prompt("analysis_agent", DEFAULT_ANALYSIS_AGENT_PROMPT)

        # Create tools with task-specific source_file for precise function lookup
        # Apply caching wrapper to prevent redundant tool calls (Optimization 1)
        # Pass project_path to enable reading complete source code directly from files
        tools = create_analysis_tools(
            self.mcp_client,
            self.project_name,
            source_file=task.source_file,
            project_path=self.project_path,
            cache_wrapper=self._create_cached_tool,
        )

        # Create AnalysisAgent with all tools
        analysis_agent = AssistantAgent(
            name="AnalysisAgent",
            model_client=self.model_client,
            system_message=system_prompt,
            tools=tools,
            description="负责分析代码结构、数据流和控制流",
        )

        # Create a simple team for single agent with tool execution
        # Using RoundRobinGroupChat with single agent allows tool calls
        termination = TextMentionTermination("ANALYSIS_COMPLETE") | MaxMessageTermination(
            self.settings.agents.code_analyzer.max_iterations
        )

        team = RoundRobinGroupChat(
            participants=[analysis_agent],
            termination_condition=termination,
        )

        # Build simple task message (multi-line messages may cause LLM issues)
        task_message = (
            f"分析函数 {task.function_name}（项目: {self.project_name}，文件: {task.source_file}）"
        )

        # Run analysis
        try:
            # Note: Console(run_stream) has output issues, using run() directly
            # and printing messages manually for verbose mode
            result = await team.run(task=task_message)

            if verbose and hasattr(result, "messages"):
                for msg in result.messages:
                    source = getattr(msg, "source", "unknown")
                    content = self._extract_content(msg)
                    msg_type = type(msg).__name__
                    logger.info(f"[{msg_type}] {source}: {content}...")

            # Record conversation and tool calls
            if hasattr(result, "messages"):
                self._record_messages_with_tool_calls(result.messages, "AnalysisAgent")

            # Extract JSON from the final messages
            return self._extract_analysis_result(result)

        except Exception as e:
            logger.error(f"Analysis phase error: {e}")
            return None

    async def _run_generation_phase(
        self,
        task: AnalysisTask,
        analysis_result: dict[str, Any],
        verbose: bool,
    ) -> str | None:
        """Run the model generation phase."""
        # Get prompt
        system_prompt = self._get_prompt("model_generator", DEFAULT_MODEL_GENERATOR_PROMPT)

        # Create ModelGenerator (no tools needed)
        model_generator = AssistantAgent(
            name="ModelGenerator",
            model_client=self.model_client,
            system_message=system_prompt,
            description="负责生成 XML DataModel",
        )

        # Single call, no iteration needed
        termination = TextMentionTermination("MODEL_COMPLETE") | MaxMessageTermination(3)

        team = RoundRobinGroupChat(
            participants=[model_generator],
            termination_condition=termination,
        )

        # Build generation prompt (keep JSON compact for better LLM handling)
        analysis_json = json.dumps(analysis_result, ensure_ascii=False)
        task_message = f"根据分析结果生成 DataModel（名称: {task.output_name or task.function_name + 'Model'}）: {analysis_json}"

        try:
            result = await team.run(task=task_message)

            if verbose and hasattr(result, "messages"):
                for msg in result.messages:
                    source = getattr(msg, "source", "unknown")
                    content = self._extract_content(msg)
                    msg_type = type(msg).__name__
                    logger.info(f"[Gen] [{msg_type}] {source}: {content}...")

            # Record conversation (no tool calls in generation phase)
            if hasattr(result, "messages"):
                self._record_messages_with_tool_calls(result.messages, "ModelGenerator")

            # Extract XML from result
            return self._extract_xml_result(result)

        except Exception as e:
            logger.error(f"Generation phase error: {e}")
            return None

    def _record_messages_with_tool_calls(self, messages: list[Any], default_agent: str) -> None:
        """Record messages and extract structured tool call information.

        Args:
            messages: List of autogen messages
            default_agent: Default agent name if not specified in message
        """
        for msg in messages:
            agent_name = getattr(msg, "source", default_agent)
            msg_type = type(msg).__name__

            # Get content for all message types
            content = getattr(msg, "content", None)

            # Debug: log message type and full content (no truncation)
            if content:
                content_str = str(content)
                logger.debug(f"Message type: {msg_type}, content length: {len(content_str)}")
                logger.debug(f"Message content: {content_str}")

            # Handle ToolCallRequestEvent - tool calls made by agent
            if msg_type == "ToolCallRequestEvent":
                if isinstance(content, list):
                    for item in content:
                        if hasattr(item, "name") and hasattr(item, "arguments"):
                            # This is a FunctionCall object
                            try:
                                args = (
                                    json.loads(item.arguments)
                                    if isinstance(item.arguments, str)
                                    else item.arguments
                                )
                            except json.JSONDecodeError:
                                args = {"raw": item.arguments}

                            call_id = getattr(item, "id", None)
                            logger.debug(f"Recording tool call: {item.name} with call_id={call_id}")
                            self.recorder.record_tool_call(
                                agent_name=agent_name,
                                tool_name=item.name,
                                arguments=args,
                                call_id=call_id,
                            )
                continue  # Don't record raw ToolCallRequestEvent to conversation

            # Handle ToolCallExecutionEvent - tool execution results
            if msg_type == "ToolCallExecutionEvent":
                if isinstance(content, list):
                    for item in content:
                        if hasattr(item, "call_id") and hasattr(item, "content"):
                            # This is a FunctionExecutionResult object
                            call_id = getattr(item, "call_id", None)
                            result_content = getattr(item, "content", "")
                            is_error = getattr(item, "is_error", False)

                            logger.debug(
                                f"Recording tool result: call_id={call_id}, content_len={len(str(result_content))}"
                            )
                            if call_id:
                                self.recorder.record_tool_result(
                                    call_id, str(result_content), is_error
                                )
                continue  # Don't record raw ToolCallExecutionEvent to conversation

            # Handle ToolCallSummaryMessage - clean summary of tool results for conversation
            if msg_type == "ToolCallSummaryMessage":
                if isinstance(content, str):
                    self.recorder.record(agent_name, content)
                continue

            # Handle other message types (TextMessage, ThoughtEvent, etc.)
            if content is None:
                continue

            if isinstance(content, str):
                # Check for tool result pattern in string format (legacy format)
                # Format: content='...' name='tool_name' call_id='xxx' is_error=False
                if "call_id=" in content and "is_error=" in content:
                    logger.debug(
                        f"Detected legacy tool result pattern in content (length={len(content)})"
                    )
                    self._parse_tool_result_string(agent_name, content)
                    # Extract and record only the actual content
                    content_match = re.search(r"content=['\"](.+?)['\"] name=", content, re.DOTALL)
                    if content_match:
                        actual_content = (
                            content_match.group(1).replace("\\n", "\n").replace("\\\\", "\\")
                        )
                        self.recorder.record(agent_name, actual_content)
                # Check for FunctionCall string pattern (legacy format)
                elif content.startswith("FunctionCall("):
                    self._parse_function_call_string(agent_name, content)
                    # Don't record the raw FunctionCall string to conversation
                else:
                    # Normal text message
                    self.recorder.record(agent_name, content)
            elif isinstance(content, list):
                # Some messages have list content that's not tool-related
                for item in content:
                    self.recorder.record(agent_name, str(item))
            else:
                # Fallback for other content types
                self.recorder.record(agent_name, str(content))

    def _parse_function_call_string(self, agent_name: str, content: str) -> None:
        """Parse FunctionCall string representation and record it."""
        # Pattern: FunctionCall(id='xxx', arguments='{"key": "value"}', name='tool_name')
        id_match = re.search(r"id=['\"]([^'\"]+)['\"]", content)
        name_match = re.search(r"name=['\"]([^'\"]+)['\"]", content)
        args_match = re.search(r"arguments=['\"](.+?)['\"](?:,|\))", content)

        if name_match:
            tool_name = name_match.group(1)
            call_id = id_match.group(1) if id_match else None

            args = {}
            if args_match:
                try:
                    # Handle escaped JSON
                    args_str = args_match.group(1).replace('\\"', '"')
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {"raw": args_match.group(1)}

            self.recorder.record_tool_call(
                agent_name=agent_name,
                tool_name=tool_name,
                arguments=args,
                call_id=call_id,
            )

    def _parse_tool_result_string(self, agent_name: str, content: str) -> None:
        """Parse tool result string and record it.

        Format: content='...' name='tool_name' call_id='xxx' is_error=False
        """
        call_id_match = re.search(r"call_id=['\"]([^'\"]+)['\"]", content)
        if not call_id_match:
            logger.debug(f"No call_id found in tool result string (length={len(content)})")
            logger.debug(f"Full content: {content}")
            return

        call_id = call_id_match.group(1)
        is_error_match = re.search(r"is_error=(True|False)", content)
        is_error = is_error_match.group(1) == "True" if is_error_match else False

        # Try to extract content between content='...' and name=
        # Use DOTALL flag to match across newlines
        content_match = re.search(r"content=['\"](.+?)['\"] name=", content, re.DOTALL)

        if content_match:
            result_content = content_match.group(1)
            # Unescape common escape sequences
            result_content = result_content.replace("\\n", "\n").replace("\\\\", "\\")
        else:
            # Fallback: use the whole content string
            result_content = content
            logger.debug("Could not extract result content from string, using full content")

        logger.debug(
            f"Parsed tool result: call_id={call_id}, is_error={is_error}, content_len={len(result_content)}"
        )
        self.recorder.record_tool_result(call_id, result_content, is_error)

    def _extract_content(self, msg: Any) -> str:
        """Extract string content from message."""
        content = getattr(msg, "content", None)
        if content is None:
            return str(msg)
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        if not isinstance(content, str):
            return str(content)
        return content

    def _extract_analysis_result(self, result: Any) -> dict[str, Any] | None:
        """Extract and validate analysis JSON from agent result (Optimization 4)."""
        if not hasattr(result, "messages"):
            return None

        # Search from the end for ANALYSIS_COMPLETE marker
        for msg in reversed(result.messages):
            content = self._extract_content(msg)
            if "ANALYSIS_COMPLETE" in content or "status" in content:
                json_data = self.validator.extract_json(content)
                if json_data and self.validator.validate_analysis_result(json_data):
                    # Validate result completeness and quality (Optimization 4)
                    is_valid, warnings = self._validate_analysis_result(json_data)

                    if warnings:
                        logger.warning(f"分析结果存在 {len(warnings)} 个质量问题:")
                        for w in warnings:
                            logger.warning(f"  - {w}")

                        # Add warnings to result
                        if "analysis_notes" not in json_data:
                            json_data["analysis_notes"] = []
                        json_data["analysis_notes"].extend([f"质量问题: {w}" for w in warnings])
                        json_data["quality_warnings"] = warnings

                    if not is_valid:
                        logger.error("分析结果验证失败，但仍然返回部分结果")
                        if "status" not in json_data:
                            json_data["status"] = "partial"

                    return json_data

        # Fallback: try to extract any JSON from any message
        for msg in reversed(result.messages):
            content = self._extract_content(msg)
            json_data = self.validator.extract_json(content)
            if json_data:
                logger.warning("从消息中提取到 JSON，但没有 ANALYSIS_COMPLETE 标记")

                # Still validate even without marker
                is_valid, warnings = self._validate_analysis_result(json_data)
                if warnings:
                    json_data["quality_warnings"] = warnings
                    if "analysis_notes" not in json_data:
                        json_data["analysis_notes"] = []
                    json_data["analysis_notes"].extend([f"质量问题: {w}" for w in warnings])

                return json_data

        return None

    def _extract_xml_result(self, result: Any) -> str | None:
        """Extract XML from agent result."""
        if not hasattr(result, "messages"):
            return None

        for msg in reversed(result.messages):
            content = self._extract_content(msg)
            xml = self.validator.extract_xml(content)
            if xml:
                return xml

        return None

    def _wrap_xml(self, xml_content: str) -> str:
        """Wrap XML content in Secray root element."""
        if xml_content.startswith("<?xml"):
            # Remove existing XML declaration
            xml_content = re.sub(r"<\?xml[^?]*\?>\s*", "", xml_content)

        return f'<?xml version="1.0" encoding="utf-8"?>\n<Secray>\n{xml_content}\n</Secray>'

    async def _save_analysis_result(
        self, task: AnalysisTask, analysis_result: dict[str, Any]
    ) -> None:
        """Save intermediate analysis result."""
        if self.storage_path is None:
            return

        task_dir = self.storage_path / "results" / task.task_id
        intermediate_dir = task_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Save task metadata
        task_meta = {
            "task_id": task.task_id,
            "source_file": task.source_file,
            "function_name": task.function_name,
            "output_name": task.output_name,
            "project_name": self.project_name,
            "created_at": datetime.now().isoformat(),
        }
        (task_dir / "task_meta.json").write_text(
            json.dumps(task_meta, indent=2, ensure_ascii=False)
        )

        # Save analysis result
        (intermediate_dir / "analysis_result.json").write_text(
            json.dumps(analysis_result, indent=2, ensure_ascii=False)
        )
        logger.debug(f"Saved analysis result to {intermediate_dir}")


# ============================================================================
# Legacy Compatibility Alias
# ============================================================================


class AnalysisWorkflowRunner(TwoPhaseWorkflow):
    """Alias for backward compatibility with runner.py."""

    async def run_analysis(self, task: AnalysisTask, verbose: bool = True) -> TaskResult:
        """Run analysis (alias for run method)."""
        return await self.run(task, verbose=verbose)


# ============================================================================
# Single Agent Analysis (Simplified)
# ============================================================================


async def run_single_agent_analysis(
    settings: Settings,
    mcp_client: MCPHttpClient,
    project_name: str,
    task: AnalysisTask,
    custom_knowledge: str = "",
    storage_path: Path | None = None,
    project_path: Any = None,
) -> TaskResult:
    """Run a simplified single-agent analysis.

    Fallback when full workflow is not needed.
    """
    async with TwoPhaseWorkflow(
        settings, mcp_client, project_name, custom_knowledge, storage_path, project_path
    ) as workflow:
        return await workflow.run(task, verbose=False)
