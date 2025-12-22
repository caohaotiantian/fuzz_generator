"""AutoGen-based Agent implementation for two-phase analysis workflow.

This module implements a simplified two-agent system:
- AnalysisAgent: Combined code analysis + context building with iterative tool calls
- ModelGenerator: DataModel generation based on analysis results

Design Reference: docs/AGENT_OPTIMIZATION.md

The workflow is sequential and deterministic:
1. AnalysisAgent iteratively collects code context (multiple tool calls)
2. ModelGenerator generates XML based on analysis results
"""

import json
import re
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


def create_analysis_tools(mcp_client: MCPHttpClient, project_name: str) -> list:
    """Create all analysis tool functions for AnalysisAgent.

    Args:
        mcp_client: MCP HTTP client instance
        project_name: Active project name in Joern

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

        # Note: query_tools.get_function_code signature is (client, function_name, project_name)
        result = await _get_func(mcp_client, function_name, project_name)
        if result.success:
            return f"函数 {function_name} 的源代码:\n{result.code}"
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
        from fuzz_generator.tools.query_tools import search_code as _search

        result = await _search(mcp_client, project_name, pattern)
        if result.success:
            return f"搜索结果:\n{json.dumps(result.results, indent=2, ensure_ascii=False)}"
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

    return [
        get_function_code,
        list_functions,
        search_code,
        track_dataflow,
        get_callers,
        get_callees,
        get_control_flow_graph,
    ]


# ============================================================================
# Conversation Recorder
# ============================================================================


class ConversationRecorder:
    """Records agent conversations for debugging."""

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path
        self.messages: list[dict[str, Any]] = []

    def record(self, agent_name: str, content: str, role: str = "assistant") -> None:
        """Record a message."""
        self.messages.append(
            {
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name,
                "role": role,
                "content": content[:2000]
                if len(content) > 2000
                else content,  # Truncate long content
            }
        )

    def get_messages(self) -> list[dict[str, Any]]:
        return self.messages

    async def save(self, task_id: str) -> None:
        """Save conversation to file."""
        if self.storage_path is None:
            return

        intermediate_dir = self.storage_path / "results" / task_id / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        conversations_file = intermediate_dir / "agent_conversations.json"
        conversations_file.write_text(json.dumps(self.messages, indent=2, ensure_ascii=False))
        logger.debug(f"Saved conversation to {conversations_file}")


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
        """Validate analysis result structure."""
        required = ["status", "function", "parameters"]
        return all(key in data for key in required)


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
    ):
        """Initialize workflow.

        Args:
            settings: Application settings
            mcp_client: MCP HTTP client
            project_name: Active project name
            custom_knowledge: Custom background knowledge
            storage_path: Path for storing intermediate results
        """
        self.settings = settings
        self.mcp_client = mcp_client
        self.project_name = project_name
        self.custom_knowledge = custom_knowledge
        self.storage_path = storage_path

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

        # Create tools
        self.tools = create_analysis_tools(mcp_client, project_name)

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

        # Create AnalysisAgent with all tools
        analysis_agent = AssistantAgent(
            name="AnalysisAgent",
            model_client=self.model_client,
            system_message=system_prompt,
            tools=self.tools,
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
                    logger.info(f"[{msg_type}] {source}: {content[:200]}...")

            # Record conversation
            if hasattr(result, "messages"):
                for msg in result.messages:
                    agent_name = getattr(msg, "source", "unknown")
                    content = self._extract_content(msg)
                    self.recorder.record(agent_name, content)

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
                    logger.info(f"[Gen] [{msg_type}] {source}: {content[:200]}...")

            # Record conversation
            if hasattr(result, "messages"):
                for msg in result.messages:
                    agent_name = getattr(msg, "source", "unknown")
                    content = self._extract_content(msg)
                    self.recorder.record(agent_name, content)

            # Extract XML from result
            return self._extract_xml_result(result)

        except Exception as e:
            logger.error(f"Generation phase error: {e}")
            return None

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
        """Extract analysis JSON from agent result."""
        if not hasattr(result, "messages"):
            return None

        # Search from the end for ANALYSIS_COMPLETE marker
        for msg in reversed(result.messages):
            content = self._extract_content(msg)
            if "ANALYSIS_COMPLETE" in content or "status" in content:
                json_data = self.validator.extract_json(content)
                if json_data and self.validator.validate_analysis_result(json_data):
                    return json_data

        # Fallback: try to extract any JSON from any message
        for msg in reversed(result.messages):
            content = self._extract_content(msg)
            json_data = self.validator.extract_json(content)
            if json_data:
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
) -> TaskResult:
    """Run a simplified single-agent analysis.

    Fallback when full workflow is not needed.
    """
    async with TwoPhaseWorkflow(
        settings, mcp_client, project_name, custom_knowledge, storage_path
    ) as workflow:
        return await workflow.run(task, verbose=False)
