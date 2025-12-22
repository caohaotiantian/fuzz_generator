"""Simple workflow implementation using direct LLM API calls.

This module implements a simplified analysis workflow that:
1. Uses direct LLM API calls via SimpleLLMClient
2. Follows a sequential workflow: Analyze -> Context -> Generate
3. Avoids complex multi-agent coordination

Design Reference: docs/TECHNICAL_DESIGN.md Section 4.2
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fuzz_generator.agents.llm_client import (
    ChatMessage,
    LLMConfig,
    SimpleLLMClient,
)
from fuzz_generator.config import Settings
from fuzz_generator.models import AnalysisTask, TaskResult
from fuzz_generator.tools.mcp_client import MCPHttpClient
from fuzz_generator.utils.logger import get_logger

logger = get_logger(__name__)


class SimpleAnalysisWorkflow:
    """Simple sequential analysis workflow.

    This implements a straightforward workflow:
    1. Get function code via MCP
    2. Analyze code structure with LLM
    3. Get data flow and call graph via MCP
    4. Build context with LLM
    5. Generate DataModel XML with LLM

    This avoids the complexity of multi-agent coordination while
    achieving the same result.
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
            project_name: Active project name in MCP server
            custom_knowledge: Custom background knowledge for prompts
            storage_path: Path for storing intermediate results
        """
        self.settings = settings
        self.mcp_client = mcp_client
        self.project_name = project_name
        self.custom_knowledge = custom_knowledge
        self.storage_path = storage_path

        # LLM config
        self.llm_config = LLMConfig(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            model=settings.llm.model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=settings.llm.timeout,
        )

        # Conversation history for recording
        self.conversation_history: list[dict[str, Any]] = []

    async def run(self, task: AnalysisTask, verbose: bool = False) -> TaskResult:
        """Run the analysis workflow.

        Args:
            task: Analysis task to execute
            verbose: Whether to show progress output

        Returns:
            TaskResult with analysis results
        """
        logger.info(f"Starting analysis for: {task.function_name} in {task.source_file}")

        try:
            async with SimpleLLMClient(self.llm_config) as llm:
                # Step 1: Get function code
                if verbose:
                    print("  [1/4] Getting function code...")

                function_code = await self._get_function_code(task.function_name)
                if not function_code:
                    return self._error_result(
                        task, f"Failed to get code for function: {task.function_name}"
                    )

                self._record("system", f"Retrieved function code for {task.function_name}")

                # Step 2: Analyze code structure
                if verbose:
                    print("  [2/4] Analyzing code structure...")

                code_analysis = await self._analyze_code(llm, task, function_code)
                self._record("CodeAnalyzer", code_analysis)

                # Step 3: Build context (data flow, call graph)
                if verbose:
                    print("  [3/4] Building context...")

                context_info = await self._build_context(llm, task, function_code, code_analysis)
                self._record("ContextBuilder", context_info)

                # Step 4: Generate DataModel XML
                if verbose:
                    print("  [4/4] Generating DataModel...")

                xml_content = await self._generate_datamodel(llm, task, code_analysis, context_info)
                self._record("ModelGenerator", xml_content)

                # Save intermediate results
                await self._save_results(task)

                if xml_content and "<DataModel" in xml_content:
                    return TaskResult(
                        task_id=task.task_id,
                        success=True,
                        xml_content=xml_content,
                        errors=[],
                        warnings=[],
                    )
                else:
                    return self._error_result(task, "Failed to generate valid XML DataModel")

        except Exception as e:
            logger.exception(f"Analysis workflow failed: {e}")
            return self._error_result(task, str(e))

    async def _get_function_code(self, function_name: str) -> str | None:
        """Get function code via MCP.

        Args:
            function_name: Name of the function

        Returns:
            Function code string or None
        """
        try:
            result = await self.mcp_client.call_tool(
                "get_function_code",
                {
                    "function_name": function_name,
                    "project_name": self.project_name,
                },
            )

            if result.success and result.data.get("success"):
                return result.data.get("code", "")
            else:
                error_msg = result.data.get("error") or result.error or result.raw_content
                logger.error(f"Failed to get function code: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"MCP call failed: {e}")
            return None

    async def _get_dataflow(self, function_name: str) -> dict[str, Any]:
        """Get data flow information via MCP.

        Args:
            function_name: Name of the function

        Returns:
            Data flow information dict
        """
        try:
            # Get callers
            callers_result = await self.mcp_client.call_tool(
                "get_callers",
                {
                    "function_name": function_name,
                    "project_name": self.project_name,
                },
            )

            # Get callees
            callees_result = await self.mcp_client.call_tool(
                "get_callees",
                {
                    "function_name": function_name,
                    "project_name": self.project_name,
                },
            )

            return {
                "callers": callers_result.data.get("callers", []) if callers_result.success else [],
                "callees": callees_result.data.get("callees", []) if callees_result.success else [],
            }

        except Exception as e:
            logger.warning(f"Failed to get dataflow: {e}")
            return {"callers": [], "callees": []}

    async def _analyze_code(
        self,
        llm: SimpleLLMClient,
        task: AnalysisTask,
        function_code: str,
    ) -> str:
        """Analyze function code structure.

        Args:
            llm: LLM client
            task: Analysis task
            function_code: Function source code

        Returns:
            Code analysis result as JSON string
        """
        system_prompt = f"""你是一个代码分析专家。请分析给定的C函数代码，提取以下信息：

1. 函数签名（返回类型、函数名、参数列表）
2. 输入参数的类型和含义
3. 返回值的类型和含义
4. 局部变量
5. 函数调用的外部函数
6. 关键的控制流结构（if/else, while, for等）

{f"背景知识:{chr(10)}{self.custom_knowledge}" if self.custom_knowledge else ""}

请以JSON格式返回分析结果。"""

        user_prompt = f"""请分析以下函数代码：

函数名: {task.function_name}
源文件: {task.source_file}

代码:
```c
{function_code}
```

请提取函数的结构信息并以JSON格式返回。"""

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = await llm.chat(messages)
        return response.content

    async def _build_context(
        self,
        llm: SimpleLLMClient,
        task: AnalysisTask,
        function_code: str,
        code_analysis: str,
    ) -> str:
        """Build context information for DataModel generation.

        Args:
            llm: LLM client
            task: Analysis task
            function_code: Function source code
            code_analysis: Previous code analysis result

        Returns:
            Context information as string
        """
        # Get data flow info from MCP
        dataflow = await self._get_dataflow(task.function_name)

        system_prompt = f"""你是一个上下文构建专家。基于代码分析结果和数据流信息，
构建用于生成Fuzz测试DataModel的上下文信息。

需要识别：
1. 输入数据的来源和约束
2. 数据在函数中的流向
3. 边界条件和有效值范围
4. 潜在的安全敏感操作

{f"背景知识:{chr(10)}{self.custom_knowledge}" if self.custom_knowledge else ""}

请总结关键信息，为DataModel生成做准备。"""

        user_prompt = f"""基于以下信息构建上下文：

## 函数代码
```c
{function_code}
```

## 代码分析结果
{code_analysis}

## 数据流信息
调用者: {json.dumps(dataflow.get("callers", []), indent=2, ensure_ascii=False)}
被调用函数: {json.dumps(dataflow.get("callees", []), indent=2, ensure_ascii=False)}

请总结用于生成DataModel的关键上下文信息。"""

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = await llm.chat(messages)
        return response.content

    async def _generate_datamodel(
        self,
        llm: SimpleLLMClient,
        task: AnalysisTask,
        code_analysis: str,
        context_info: str,
    ) -> str:
        """Generate DataModel XML.

        Args:
            llm: LLM client
            task: Analysis task
            code_analysis: Code analysis result
            context_info: Context information

        Returns:
            Generated XML string
        """
        output_name = task.output_name or f"{task.function_name}Model"

        system_prompt = f"""你是一个Fuzz测试数据模型生成专家。
基于代码分析和上下文信息，生成Peach Fuzzer格式的DataModel XML。

DataModel应包含：
1. 函数输入参数的数据定义
2. 参数的类型约束（如长度限制、取值范围）
3. 参数之间的关系
4. 适当的变异提示

{f"背景知识:{chr(10)}{self.custom_knowledge}" if self.custom_knowledge else ""}

请直接输出完整的XML内容，不要添加任何解释。"""

        user_prompt = f"""请基于以下信息生成DataModel XML：

## 输出名称
{output_name}

## 代码分析
{code_analysis}

## 上下文信息
{context_info}

请生成完整的Peach Fuzzer DataModel XML。格式示例：
```xml
<?xml version="1.0" encoding="utf-8"?>
<Peach>
    <DataModel name="{output_name}">
        <!-- 数据元素定义 -->
    </DataModel>
</Peach>
```"""

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = await llm.chat(messages)

        # Extract XML from response
        content = response.content

        # Try to find XML block
        if "```xml" in content:
            start = content.find("```xml") + 6
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()

        return content

    def _record(self, agent: str, content: str) -> None:
        """Record conversation message.

        Args:
            agent: Agent name
            content: Message content
        """
        self.conversation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "agent": agent,
                "content": content[:1000] if len(content) > 1000 else content,
            }
        )

    async def _save_results(self, task: AnalysisTask) -> None:
        """Save intermediate results.

        Args:
            task: Analysis task
        """
        if not self.storage_path:
            return

        try:
            # Create directories
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

            # Save conversation history
            (intermediate_dir / "conversation_history.json").write_text(
                json.dumps(self.conversation_history, indent=2, ensure_ascii=False)
            )

            logger.debug(f"Saved results to {task_dir}")

        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")

    def _error_result(self, task: AnalysisTask, error: str) -> TaskResult:
        """Create error result.

        Args:
            task: Analysis task
            error: Error message

        Returns:
            TaskResult with error
        """
        return TaskResult(
            task_id=task.task_id,
            success=False,
            xml_content=None,
            errors=[error],
            warnings=[],
        )
