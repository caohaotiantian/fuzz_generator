"""Context Builder Agent for dataflow and controlflow analysis.

This agent is responsible for:
- Tracking dataflow for function parameters
- Analyzing control flow structure
- Getting call relationships (callers/callees)
- Building complete analysis context
"""

from typing import TYPE_CHECKING, Any

from fuzz_generator.agents.base import AgentConfig, AgentResult, BaseAgent
from fuzz_generator.exceptions import AnalysisError
from fuzz_generator.models import (
    AnalysisContext,
    FunctionInfo,
)
from fuzz_generator.models.analysis_result import (
    CalleeInfo as ModelCalleeInfo,
)
from fuzz_generator.models.analysis_result import (
    CallerInfo as ModelCallerInfo,
)
from fuzz_generator.models.analysis_result import (
    CallGraphInfo,
    ControlFlowInfo,
    DataFlowNode,
    DataFlowPath,
)
from fuzz_generator.tools.analysis_tools import (
    get_callees,
    get_callers,
    get_control_flow_graph,
    track_dataflow,
)
from fuzz_generator.utils.logger import get_logger

if TYPE_CHECKING:
    from fuzz_generator.config import Settings
    from fuzz_generator.tools.mcp_client import MCPHttpClient

logger = get_logger(__name__)

# Default system prompt for ContextBuilder
DEFAULT_SYSTEM_PROMPT = """你是一个代码分析专家，负责分析函数的数据流和控制流信息。

## 你的任务
1. 追踪函数参数的数据流路径
2. 分析函数的控制流结构
3. 获取函数的调用关系（调用者和被调用者）
4. 汇总所有信息构建完整的分析上下文

## 可用工具
- track_dataflow: 追踪数据流
- get_callers: 获取调用该函数的函数
- get_callees: 获取该函数调用的函数
- get_control_flow_graph: 获取控制流图
- analyze_variable_flow: 分析变量流

## 输出格式
请汇总分析结果，输出完整的上下文信息。
"""


class ContextBuilderAgent(BaseAgent):
    """Agent for building analysis context.

    This agent gathers dataflow, controlflow, and call graph
    information to build a complete analysis context.
    """

    def __init__(
        self,
        mcp_client: "MCPHttpClient",
        settings: "Settings | None" = None,
        config: AgentConfig | None = None,
    ):
        """Initialize ContextBuilder agent.

        Args:
            mcp_client: MCP HTTP client for tool calls
            settings: Application settings
            config: Agent configuration
        """
        if config is None:
            config = AgentConfig(
                name="ContextBuilder",
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                max_iterations=15,
            )

        super().__init__(config, settings)

        self.mcp_client = mcp_client
        self._project_name: str = ""

    async def run(
        self,
        project_name: str,
        function_info: FunctionInfo,
        **kwargs: Any,
    ) -> AgentResult:
        """Build analysis context for a function.

        Args:
            project_name: Name of the project in Joern
            function_info: FunctionInfo from code analysis
            **kwargs: Additional arguments

        Returns:
            AgentResult with AnalysisContext in data field
        """
        self._project_name = project_name
        logger.info(f"Building context for function: {function_info.name}")

        try:
            # Gather all context information
            data_flows = await self._get_dataflows(function_info)
            control_flow = await self._get_control_flow(function_info.name)
            call_graph = await self._get_call_graph(function_info.name)

            # Build analysis context
            context = AnalysisContext(
                function_info=function_info,
                data_flows=data_flows,
                control_flow=control_flow,
                call_graph=call_graph,
            )

            logger.info(f"Context built for function: {function_info.name}")

            return self._create_result(
                success=True,
                data=context,
                iterations=1,
            )

        except Exception as e:
            logger.error(f"Context building failed: {e}")
            return self._create_result(
                success=False,
                error=str(e),
            )

    async def _get_dataflows(
        self,
        function_info: FunctionInfo,
    ) -> list[DataFlowPath]:
        """Get dataflow information for function parameters.

        Args:
            function_info: Function information

        Returns:
            List of dataflow paths
        """
        data_flows: list[DataFlowPath] = []

        try:
            # Track dataflow for each input parameter
            for param in function_info.input_parameters:
                result = await track_dataflow(
                    self.mcp_client,
                    project_name=self._project_name,
                    source_pattern=param.name,
                )

                if result.success:
                    for flow in result.flows:
                        # Convert source/sink dicts to DataFlowNode objects
                        source_node = (
                            DataFlowNode(
                                code=flow.source.get("code", ""),
                                file=flow.source.get("file", ""),
                                line=flow.source.get("line", 0),
                                node_type=flow.source.get("node_type", "source"),
                            )
                            if isinstance(flow.source, dict)
                            else DataFlowNode(
                                code=str(flow.source),
                                file="",
                                line=0,
                            )
                        )
                        sink_node = (
                            DataFlowNode(
                                code=flow.sink.get("code", ""),
                                file=flow.sink.get("file", ""),
                                line=flow.sink.get("line", 0),
                                node_type=flow.sink.get("node_type", "sink"),
                            )
                            if isinstance(flow.sink, dict)
                            else DataFlowNode(
                                code=str(flow.sink),
                                file="",
                                line=0,
                            )
                        )
                        data_flows.append(
                            DataFlowPath(
                                source=source_node,
                                sink=sink_node,
                                path_length=flow.path_length,
                            )
                        )

        except Exception as e:
            logger.warning(f"Failed to get dataflows: {e}")

        return data_flows

    async def _get_control_flow(
        self,
        function_name: str,
    ) -> ControlFlowInfo:
        """Get control flow information.

        Args:
            function_name: Function name

        Returns:
            ControlFlowInfo instance
        """
        try:
            result = await get_control_flow_graph(
                self.mcp_client,
                project_name=self._project_name,
                function_name=function_name,
            )

            if result.success:
                # Extract branch information from nodes
                branches = []
                for node in result.nodes:
                    if node.node_type in ("IF", "CONDITION", "SWITCH"):
                        branches.append(
                            {
                                "id": node.id,
                                "code": node.code,
                                "type": node.node_type,
                                "line": node.line_number,
                            }
                        )

                return ControlFlowInfo(
                    has_loops=result.has_loops,
                    has_conditions=result.has_conditions,
                    branches=branches,
                    complexity=len(result.nodes),
                )

        except Exception as e:
            logger.warning(f"Failed to get control flow: {e}")

        return ControlFlowInfo(
            has_loops=False,
            has_conditions=False,
            branches=[],
            complexity=0,
        )

    async def _get_call_graph(
        self,
        function_name: str,
    ) -> CallGraphInfo:
        """Get call graph information.

        Args:
            function_name: Function name

        Returns:
            CallGraphInfo instance
        """
        callers: list[ModelCallerInfo] = []
        callees: list[ModelCalleeInfo] = []

        try:
            # Get callers
            callers_result = await get_callers(
                self.mcp_client,
                project_name=self._project_name,
                function_name=function_name,
            )

            if callers_result.success:
                callers = [
                    ModelCallerInfo(
                        name=c.name,
                        file=c.filename,
                        line=c.line_number,
                    )
                    for c in callers_result.callers
                ]

        except Exception as e:
            logger.warning(f"Failed to get callers: {e}")

        try:
            # Get callees
            callees_result = await get_callees(
                self.mcp_client,
                project_name=self._project_name,
                function_name=function_name,
            )

            if callees_result.success:
                callees = [
                    ModelCalleeInfo(
                        name=c.name,
                        file=c.filename,
                    )
                    for c in callees_result.callees
                ]

        except Exception as e:
            logger.warning(f"Failed to get callees: {e}")

        return CallGraphInfo(
            callers=callers,
            callees=callees,
            call_depth=1,  # Direct calls only for now
        )

    async def build_context(
        self,
        project_name: str,
        function_info: FunctionInfo,
        **kwargs: Any,
    ) -> AnalysisContext:
        """Convenience method to build context.

        Args:
            project_name: Project name
            function_info: Function information
            **kwargs: Additional arguments

        Returns:
            AnalysisContext with all gathered information

        Raises:
            AnalysisError: If context building fails
        """
        result = await self.run(
            project_name=project_name,
            function_info=function_info,
            **kwargs,
        )

        if not result.success:
            raise AnalysisError(
                f"Failed to build context for '{function_info.name}': {result.error}"
            )

        return result.data
