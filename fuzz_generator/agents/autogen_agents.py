"""AutoGen-based Agent implementation for multi-agent collaboration.

This module implements the Agent system using AutoGen's AgentChat API,
following the design specification for GroupChat-based collaboration.

Design Reference: docs/TECHNICAL_DESIGN.md Section 4.2

Agent Roles:
- Orchestrator: Task coordination, workflow control, batch task management
- CodeAnalyzer: Function code parsing, parameter analysis
- ContextBuilder: Data flow/control flow analysis
- ModelGenerator: DataModel generation

Reference: https://microsoft.github.io/autogen/stable/reference/index.html
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from fuzz_generator.agents.base import PromptTemplate
from fuzz_generator.config import Settings
from fuzz_generator.models import (
    AnalysisTask,
    TaskResult,
)
from fuzz_generator.tools.mcp_client import MCPHttpClient
from fuzz_generator.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Prompt Loader
# ============================================================================


class PromptLoader:
    """Load agent prompts from YAML files.

    Design Reference: docs/TECHNICAL_DESIGN.md Section 4.1.2
    Loads prompts from config/defaults/prompts/ directory.
    """

    def __init__(self, prompts_dir: Path | None = None):
        """Initialize prompt loader.

        Args:
            prompts_dir: Directory containing prompt YAML files.
                        If None, uses default prompts directory.
        """
        if prompts_dir is None:
            # Default to package's prompts directory
            package_dir = Path(__file__).parent.parent
            prompts_dir = package_dir / "config" / "defaults" / "prompts"

        self.prompts_dir = prompts_dir
        self._cache: dict[str, PromptTemplate] = {}

    def load(self, agent_name: str) -> PromptTemplate:
        """Load prompt template for an agent.

        Args:
            agent_name: Agent name (e.g., 'code_analyzer', 'orchestrator')

        Returns:
            PromptTemplate instance
        """
        if agent_name in self._cache:
            return self._cache[agent_name]

        # Try to find the YAML file
        yaml_file = self.prompts_dir / f"{agent_name}.yaml"

        if yaml_file.exists():
            template = PromptTemplate(yaml_file)
            self._cache[agent_name] = template
            logger.debug(f"Loaded prompt template from: {yaml_file}")
            return template
        else:
            logger.warning(f"Prompt file not found: {yaml_file}, using default")
            # Return empty template, will use fallback prompts
            return PromptTemplate()

    def get_system_prompt(
        self,
        agent_name: str,
        custom_knowledge: str = "",
        **kwargs: Any,
    ) -> str:
        """Get rendered system prompt for an agent.

        Args:
            agent_name: Agent name
            custom_knowledge: Custom knowledge to inject
            **kwargs: Additional template variables

        Returns:
            Rendered system prompt string
        """
        template = self.load(agent_name)

        # Render system prompt with custom knowledge
        return template.render_system_prompt(
            custom_knowledge=custom_knowledge,
            **kwargs,
        )


# ============================================================================
# Default Fallback Prompts (used when YAML files are not available)
# ============================================================================

DEFAULT_CODE_ANALYZER_PROMPT = """你是一个专业的代码分析专家（CodeAnalyzer），负责分析C/C++函数的结构和参数。

## 你的任务
1. 使用 get_function_code 工具获取目标函数的源代码
2. 分析函数的输入参数及其类型
3. 识别函数的输出（返回值、输出参数）
4. 分析参数的约束条件（如：长度限制、取值范围等）

## 可用工具
- get_function_code: 获取函数源代码
- list_functions: 列出项目中的函数
- search_code: 搜索代码片段

## 输出格式
请以结构化的JSON格式输出分析结果：

```json
{
  "function_name": "函数名",
  "file_path": "文件路径",
  "line_number": 行号,
  "return_type": "返回类型",
  "description": "函数功能描述",
  "parameters": [
    {
      "name": "参数名",
      "type": "参数类型",
      "direction": "in/out/inout",
      "description": "参数描述",
      "constraints": ["约束条件"]
    }
  ]
}
```

完成分析后，请说 "CODE_ANALYSIS_COMPLETE" 并附上JSON结果。
"""

DEFAULT_CONTEXT_BUILDER_PROMPT = """你是一个代码上下文构建专家（ContextBuilder），负责分析函数的数据流和控制流。

## 你的任务
1. 使用 track_dataflow 追踪函数参数的数据流
2. 使用 get_callers/get_callees 分析调用关系
3. 使用 get_control_flow_graph 获取控制流信息
4. 综合分析得出完整的上下文信息

## 可用工具
- track_dataflow: 追踪数据流
- get_callers: 获取调用者
- get_callees: 获取被调用函数
- get_control_flow_graph: 获取控制流图

## 输出格式
请以结构化的JSON格式输出上下文信息：

```json
{
  "data_flows": [
    {
      "parameter": "参数名",
      "flows_to": ["使用位置"],
      "transformations": ["数据变换"]
    }
  ],
  "callers": ["调用者列表"],
  "callees": ["被调用函数列表"],
  "control_flow_complexity": "简单/中等/复杂",
  "key_constraints": ["关键约束"]
}
```

完成分析后，请说 "CONTEXT_BUILDING_COMPLETE" 并附上JSON结果。
"""

DEFAULT_MODEL_GENERATOR_PROMPT = """你是一个fuzz测试数据建模专家（ModelGenerator），负责生成Secray格式的XML DataModel。

## 你的任务
基于代码分析和上下文信息，为函数参数生成合适的DataModel定义。

## Secray DataModel元素类型

### 基本元素
- **String**: 字符串数据
  - name: 元素名称（必需）
  - value: 默认值
  - token: 是否为固定标记
  - mutable: 是否可变

- **Number**: 数值数据
  - name: 元素名称（必需）
  - size: 位数（8/16/32/64）
  - signed: 是否有符号
  - endian: 字节序（big/little）

- **Blob**: 二进制数据
  - name: 元素名称（必需）
  - length: 长度（固定值或引用其他字段）
  - minLength/maxLength: 长度范围

### 结构元素
- **Block**: 块结构
  - name: 元素名称
  - ref: 引用其他DataModel
  - minOccurs/maxOccurs: 出现次数

- **Choice**: 选择结构
  - name: 元素名称
  - 包含多个可选元素

## 输出格式
请输出有效的XML DataModel定义：

```xml
<DataModel name="模型名称">
    <String name="字段1" value="默认值" />
    <Number name="长度字段" size="32" signed="false" />
    <Blob name="数据" length="长度字段" />
</DataModel>
```

完成生成后，请说 "MODEL_GENERATION_COMPLETE" 并附上XML结果。
"""

DEFAULT_ORCHESTRATOR_PROMPT = """你是分析流程协调者（Orchestrator），负责协调代码分析、上下文构建和模型生成三个阶段。

## 工作流程
根据设计文档 docs/TECHNICAL_DESIGN.md Section 4.2.1 的协作流程：

1. **Phase 1 - 代码解析**: 让 CodeAnalyzer 分析目标函数的结构和参数
2. **Phase 2 - 上下文构建**: 让 ContextBuilder 构建数据流和控制流上下文
3. **Phase 3 - 模型生成**: 让 ModelGenerator 生成DataModel定义

## 你的职责
- 确保各阶段按顺序执行
- 汇总各阶段的分析结果
- 处理错误情况并提供反馈
- 决定下一个应该执行的Agent

## 调度规则
- 收到用户任务后，先调用 CodeAnalyzer
- CodeAnalyzer 完成后（看到 CODE_ANALYSIS_COMPLETE），调用 ContextBuilder
- ContextBuilder 完成后（看到 CONTEXT_BUILDING_COMPLETE），调用 ModelGenerator
- ModelGenerator 完成后（看到 MODEL_GENERATION_COMPLETE），汇总结果

## 任务完成标志
当所有阶段完成且生成了有效的XML DataModel时，请说 "ANALYSIS_WORKFLOW_COMPLETE"。

{custom_knowledge}
"""


# ============================================================================
# Tool Functions for MCP Integration
# ============================================================================


def create_mcp_tools(mcp_client: MCPHttpClient, project_name: str) -> dict[str, Any]:
    """Create tool functions that use the MCP client.

    Args:
        mcp_client: MCP HTTP client instance
        project_name: Active project name in Joern

    Returns:
        Dictionary of tool functions
    """

    async def get_function_code(function_name: str) -> str:
        """Get the source code of a function.

        Args:
            function_name: Name of the function to retrieve

        Returns:
            Function source code as string
        """
        from fuzz_generator.tools.query_tools import get_function_code as _get_func

        result = await _get_func(mcp_client, project_name, function_name)
        if result.success:
            return result.code
        return f"Error: {result.error}"

    async def list_functions(file_path: str | None = None) -> str:
        """List all functions in the project or a specific file.

        Args:
            file_path: Optional file path to filter functions

        Returns:
            JSON string of function list
        """
        from fuzz_generator.tools.query_tools import list_functions as _list_funcs

        result = await _list_funcs(mcp_client, project_name, file_filter=file_path)
        if result.success:
            return json.dumps([f.model_dump() for f in result.functions], indent=2)
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
            return json.dumps(result.results, indent=2)
        return f"Error: {result.error}"

    async def track_dataflow(source_pattern: str, sink_pattern: str | None = None) -> str:
        """Track data flow from source to sink.

        Args:
            source_pattern: Source variable or pattern
            sink_pattern: Optional sink pattern

        Returns:
            JSON string of dataflow paths
        """
        from fuzz_generator.tools.analysis_tools import track_dataflow as _track

        result = await _track(mcp_client, project_name, source_pattern, sink_pattern=sink_pattern)
        if result.success:
            return json.dumps([f.model_dump() for f in result.flows], indent=2)
        return f"Error: {result.error}"

    async def get_callers(function_name: str) -> str:
        """Get functions that call the specified function.

        Args:
            function_name: Target function name

        Returns:
            JSON string of caller information
        """
        from fuzz_generator.tools.analysis_tools import get_callers as _get_callers

        result = await _get_callers(mcp_client, project_name, function_name)
        if result.success:
            return json.dumps([c.model_dump() for c in result.callers], indent=2)
        return f"Error: {result.error}"

    async def get_callees(function_name: str) -> str:
        """Get functions called by the specified function.

        Args:
            function_name: Target function name

        Returns:
            JSON string of callee information
        """
        from fuzz_generator.tools.analysis_tools import get_callees as _get_callees

        result = await _get_callees(mcp_client, project_name, function_name)
        if result.success:
            return json.dumps([c.model_dump() for c in result.callees], indent=2)
        return f"Error: {result.error}"

    async def get_control_flow_graph(function_name: str) -> str:
        """Get the control flow graph of a function.

        Args:
            function_name: Target function name

        Returns:
            JSON string of CFG information
        """
        from fuzz_generator.tools.analysis_tools import (
            get_control_flow_graph as _get_cfg,
        )

        result = await _get_cfg(mcp_client, project_name, function_name)
        if result.success:
            return json.dumps(result.cfg.model_dump(), indent=2)
        return f"Error: {result.error}"

    return {
        "get_function_code": get_function_code,
        "list_functions": list_functions,
        "search_code": search_code,
        "track_dataflow": track_dataflow,
        "get_callers": get_callers,
        "get_callees": get_callees,
        "get_control_flow_graph": get_control_flow_graph,
    }


# ============================================================================
# AutoGen Agent Factory
# ============================================================================


class AutoGenAgentFactory:
    """Factory for creating AutoGen-based agents.

    Implements the Agent roles defined in docs/TECHNICAL_DESIGN.md Section 3.2.3:
    - Orchestrator: Task coordination, workflow control
    - CodeAnalyzer: Function code parsing, parameter analysis
    - ContextBuilder: Data flow/control flow analysis
    - ModelGenerator: DataModel generation

    Prompts are loaded from YAML files as per docs/TECHNICAL_DESIGN.md Section 4.1.2.
    Agent settings (max_iterations, etc.) are read from Settings.agents.
    """

    def __init__(
        self,
        settings: Settings,
        mcp_client: MCPHttpClient,
        project_name: str,
        custom_knowledge: str = "",
        prompts_dir: Path | None = None,
    ):
        """Initialize agent factory.

        Args:
            settings: Application settings (includes agents.* config)
            mcp_client: MCP HTTP client
            project_name: Active project name
            custom_knowledge: Custom background knowledge
            prompts_dir: Directory containing prompt YAML files
        """
        self.settings = settings
        self.mcp_client = mcp_client
        self.project_name = project_name
        self.custom_knowledge = custom_knowledge

        # Load prompts from YAML files
        self.prompt_loader = PromptLoader(prompts_dir)

        # Create LLM client based on settings
        self.model_client = OpenAIChatCompletionClient(
            model=settings.llm.model,
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
        )

        # Create tool functions
        self.tools = create_mcp_tools(mcp_client, project_name)

    def _get_system_prompt(self, agent_name: str, fallback_prompt: str) -> str:
        """Get system prompt from YAML file or use fallback.

        Args:
            agent_name: Agent name for YAML lookup
            fallback_prompt: Default prompt if YAML not found

        Returns:
            System prompt string
        """
        template = self.prompt_loader.load(agent_name)

        # Check if template has content
        if template.system_prompt:
            # Render with custom knowledge if applicable
            prompt = template.render_system_prompt(
                custom_knowledge=self.custom_knowledge,
            )
            return prompt

        # Use fallback
        logger.debug(f"Using fallback prompt for {agent_name}")
        if "{custom_knowledge}" in fallback_prompt and self.custom_knowledge:
            return fallback_prompt.format(
                custom_knowledge=f"\n## 背景知识\n{self.custom_knowledge}"
            )
        return fallback_prompt.replace("{custom_knowledge}", "")

    def create_orchestrator(self) -> AssistantAgent:
        """Create Orchestrator agent.

        Design Reference: docs/TECHNICAL_DESIGN.md Section 3.2.3
        Role: Task coordination, workflow control, batch task management
        Tools: None (coordinates other agents)

        Settings from: settings.agents.orchestrator
        """
        # Get prompt from YAML or fallback
        prompt = self._get_system_prompt("orchestrator", DEFAULT_ORCHESTRATOR_PROMPT)

        # Get max_iterations from settings
        max_iterations = self.settings.agents.orchestrator.max_iterations
        logger.debug(f"Orchestrator max_iterations: {max_iterations}")

        return AssistantAgent(
            name="Orchestrator",
            model_client=self.model_client,
            system_message=prompt,
            description="负责协调分析流程，决定下一步执行哪个Agent",
        )

    def create_code_analyzer(self) -> AssistantAgent:
        """Create CodeAnalyzer agent.

        Design Reference: docs/TECHNICAL_DESIGN.md Section 3.2.3
        Role: Function code parsing, parameter analysis
        Tools: get_function_code, list_functions, search_code

        Settings from: settings.agents.code_analyzer
        """
        # Get prompt from YAML or fallback
        prompt = self._get_system_prompt("code_analyzer", DEFAULT_CODE_ANALYZER_PROMPT)

        # Get settings
        agent_settings = self.settings.agents.code_analyzer
        logger.debug(f"CodeAnalyzer max_iterations: {agent_settings.max_iterations}")

        return AssistantAgent(
            name="CodeAnalyzer",
            model_client=self.model_client,
            system_message=prompt,
            tools=[
                self.tools["get_function_code"],
                self.tools["list_functions"],
                self.tools["search_code"],
            ],
            description="负责分析函数代码结构和参数，使用MCP工具获取源代码",
        )

    def create_context_builder(self) -> AssistantAgent:
        """Create ContextBuilder agent.

        Design Reference: docs/TECHNICAL_DESIGN.md Section 3.2.3
        Role: Data flow/control flow analysis
        Tools: track_dataflow, get_callees, get_callers, get_control_flow_graph

        Settings from: settings.agents.context_builder
        """
        # Get prompt from YAML or fallback
        prompt = self._get_system_prompt("context_builder", DEFAULT_CONTEXT_BUILDER_PROMPT)

        # Get settings
        agent_settings = self.settings.agents.context_builder
        logger.debug(f"ContextBuilder max_iterations: {agent_settings.max_iterations}")

        return AssistantAgent(
            name="ContextBuilder",
            model_client=self.model_client,
            system_message=prompt,
            tools=[
                self.tools["track_dataflow"],
                self.tools["get_callers"],
                self.tools["get_callees"],
                self.tools["get_control_flow_graph"],
            ],
            description="负责构建数据流和控制流上下文，分析调用关系",
        )

    def create_model_generator(self) -> AssistantAgent:
        """Create ModelGenerator agent.

        Design Reference: docs/TECHNICAL_DESIGN.md Section 3.2.3
        Role: DataModel generation
        Tools: None (generates based on context)

        Settings from: settings.agents.model_generator
        """
        # Get prompt from YAML or fallback
        prompt = self._get_system_prompt("model_generator", DEFAULT_MODEL_GENERATOR_PROMPT)

        # Get settings
        agent_settings = self.settings.agents.model_generator
        logger.debug(f"ModelGenerator max_iterations: {agent_settings.max_iterations}")

        return AssistantAgent(
            name="ModelGenerator",
            model_client=self.model_client,
            system_message=prompt,
            description="负责生成Secray XML DataModel定义",
        )

    def get_max_rounds(self) -> int:
        """Get max rounds from orchestrator settings.

        Returns:
            Maximum rounds for GroupChat
        """
        return self.settings.agents.orchestrator.max_iterations


# ============================================================================
# Conversation History Recorder
# ============================================================================


class ConversationRecorder:
    """Records agent conversations for debugging and traceability.

    Design Reference: docs/TECHNICAL_DESIGN.md Section 4.7.4
    Saves: agent_conversations.json
    """

    def __init__(self, storage_path: Path | None = None):
        """Initialize recorder.

        Args:
            storage_path: Path to store conversation logs
        """
        self.storage_path = storage_path
        self.messages: list[dict[str, Any]] = []

    def record(self, agent_name: str, content: str, role: str = "assistant") -> None:
        """Record a message.

        Args:
            agent_name: Name of the agent
            content: Message content
            role: Message role
        """
        self.messages.append(
            {
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name,
                "role": role,
                "content": content,
            }
        )

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all recorded messages."""
        return self.messages

    async def save(self, task_id: str) -> None:
        """Save conversation to file.

        Args:
            task_id: Task identifier
        """
        if self.storage_path is None:
            return

        # Create intermediate directory
        intermediate_dir = self.storage_path / "results" / task_id / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Save conversations
        conversations_file = intermediate_dir / "agent_conversations.json"
        conversations_file.write_text(json.dumps(self.messages, indent=2, ensure_ascii=False))
        logger.debug(f"Saved conversation to {conversations_file}")


# ============================================================================
# Analysis Workflow Runner
# ============================================================================


class AnalysisWorkflowRunner:
    """Runner for the multi-agent analysis workflow.

    Implements the analysis flow defined in docs/TECHNICAL_DESIGN.md Section 5.1:
    1. CLI receives analyze command
    2. Orchestrator initializes and coordinates
    3. CodeAnalyzer analyzes function code
    4. ContextBuilder builds data flow context
    5. ModelGenerator generates XML DataModel
    6. Results are saved and returned
    """

    def __init__(
        self,
        settings: Settings,
        mcp_client: MCPHttpClient,
        project_name: str,
        custom_knowledge: str = "",
        storage_path: Path | None = None,
    ):
        """Initialize workflow runner.

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

        # Create agent factory
        self.factory = AutoGenAgentFactory(settings, mcp_client, project_name, custom_knowledge)

        # Conversation recorder
        self.recorder = ConversationRecorder(storage_path)

    async def run_analysis(
        self,
        task: AnalysisTask,
        verbose: bool = True,
    ) -> TaskResult:
        """Run the complete analysis workflow for a task.

        Implements the workflow in docs/TECHNICAL_DESIGN.md Section 4.2.1

        Args:
            task: Analysis task to execute
            verbose: Whether to show conversation output

        Returns:
            TaskResult with analysis results
        """
        logger.info(f"Starting analysis for: {task.function_name} in {task.source_file}")

        try:
            # Create agents as per design doc Section 3.2.3
            orchestrator = self.factory.create_orchestrator()
            code_analyzer = self.factory.create_code_analyzer()
            context_builder = self.factory.create_context_builder()
            model_generator = self.factory.create_model_generator()

            # Get max_rounds from settings (settings.agents.orchestrator.max_iterations)
            max_rounds = self.factory.get_max_rounds()

            # Define termination conditions
            termination = TextMentionTermination(
                "ANALYSIS_WORKFLOW_COMPLETE"
            ) | MaxMessageTermination(max_rounds)

            # Create SelectorGroupChat for intelligent agent selection
            # Design doc Section 4.2.2: speaker_selection_method="auto"
            team = SelectorGroupChat(
                participants=[orchestrator, code_analyzer, context_builder, model_generator],
                model_client=self.factory.model_client,
                termination_condition=termination,
                selector_prompt="""你是一个智能调度器，负责选择下一个应该发言的Agent。

选择规则：
1. 任务开始时，选择 Orchestrator 进行协调
2. 需要分析代码时，选择 CodeAnalyzer
3. 需要构建上下文时，选择 ContextBuilder
4. 需要生成DataModel时，选择 ModelGenerator
5. 需要协调或汇总时，选择 Orchestrator

根据当前对话内容，选择最合适的Agent继续任务。
""",
            )

            # Build initial task message
            task_message = f"""请分析以下函数并生成DataModel：

项目: {self.project_name}
源文件: {task.source_file}
函数名: {task.function_name}
输出名称: {task.output_name or task.function_name + "Model"}

请按照设计文档的流程执行：
1. CodeAnalyzer: 分析函数代码结构
2. ContextBuilder: 构建数据流和控制流上下文
3. ModelGenerator: 生成XML DataModel

{f"背景知识: {self.custom_knowledge}" if self.custom_knowledge else ""}
"""

            # Run the team
            if verbose:
                result = await Console(team.run_stream(task=task_message))
            else:
                result = await team.run(task=task_message)

            # Record conversation
            if hasattr(result, "messages"):
                for msg in result.messages:
                    agent_name = getattr(msg, "source", "unknown")
                    content = getattr(msg, "content", str(msg))
                    self.recorder.record(agent_name, content)

            # Save conversation
            await self.recorder.save(task.task_id)

            # Save intermediate results
            await self._save_intermediate_results(task, result)

            # Extract XML from result
            xml_content = self._extract_xml_from_result(result)

            if xml_content:
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    xml_content=xml_content,
                    errors=[],
                    warnings=[],
                )
            else:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    xml_content=None,
                    errors=["Failed to extract XML from agent response"],
                    warnings=[],
                )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return TaskResult(
                task_id=task.task_id,
                success=False,
                xml_content=None,
                errors=[str(e)],
                warnings=[],
            )

    async def _save_intermediate_results(
        self,
        task: AnalysisTask,
        result: Any,
    ) -> None:
        """Save intermediate analysis results.

        Design Reference: docs/TECHNICAL_DESIGN.md Section 4.7.4
        Saves: task_meta.json, code_analysis.json, context_info.json
        """
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

        # Extract and save phase results from messages
        if hasattr(result, "messages"):
            code_analysis = self._extract_phase_result(result.messages, "CODE_ANALYSIS_COMPLETE")
            if code_analysis:
                (intermediate_dir / "code_analysis.json").write_text(
                    json.dumps(code_analysis, indent=2, ensure_ascii=False)
                )

            context_info = self._extract_phase_result(result.messages, "CONTEXT_BUILDING_COMPLETE")
            if context_info:
                (intermediate_dir / "context_info.json").write_text(
                    json.dumps(context_info, indent=2, ensure_ascii=False)
                )

    def _extract_phase_result(
        self,
        messages: list[Any],
        marker: str,
    ) -> dict[str, Any] | None:
        """Extract phase result from messages by marker.

        Args:
            messages: List of messages
            marker: Completion marker to look for

        Returns:
            Extracted JSON data or None
        """
        for msg in messages:
            content = getattr(msg, "content", str(msg))
            if isinstance(content, str) and marker in content:
                # Try to extract JSON
                try:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    if json_start != -1 and json_end > json_start:
                        return json.loads(content[json_start:json_end])
                except json.JSONDecodeError:
                    pass
        return None

    def _extract_xml_from_result(self, result: Any) -> str | None:
        """Extract XML content from agent conversation result.

        Args:
            result: Result from team.run()

        Returns:
            Extracted XML string or None
        """
        # Try to find XML in the messages
        if hasattr(result, "messages"):
            for msg in reversed(result.messages):
                content = getattr(msg, "content", str(msg))
                if isinstance(content, str):
                    # Look for XML DataModel block
                    xml_start = content.find("<DataModel")
                    if xml_start != -1:
                        # Find the closing tag
                        xml_end = content.find("</DataModel>", xml_start)
                        if xml_end != -1:
                            xml_content = content[xml_start : xml_end + len("</DataModel>")]
                            # Wrap in Secray root
                            return f'<?xml version="1.0" encoding="utf-8"?>\n<Secray>\n{xml_content}\n</Secray>'

        return None


# ============================================================================
# Simplified Single-Step Analysis (Fallback)
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

    This is a fallback when GroupChat is not needed.

    Args:
        settings: Application settings
        mcp_client: MCP HTTP client
        project_name: Active project name
        task: Analysis task
        custom_knowledge: Custom knowledge
        storage_path: Storage path for results

    Returns:
        TaskResult with analysis results
    """
    runner = AnalysisWorkflowRunner(
        settings, mcp_client, project_name, custom_knowledge, storage_path
    )
    return await runner.run_analysis(task, verbose=False)
