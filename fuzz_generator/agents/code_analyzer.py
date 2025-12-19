"""Code Analyzer Agent for function code analysis.

This agent is responsible for:
- Retrieving function source code via MCP tools
- Analyzing function structure and parameters
- Identifying parameter types, directions, and constraints
- Outputting structured function information
"""

from typing import TYPE_CHECKING, Any

from fuzz_generator.agents.base import AgentConfig, AgentResult, BaseAgent
from fuzz_generator.exceptions import AnalysisError
from fuzz_generator.models import FunctionInfo, ParameterDirection, ParameterInfo
from fuzz_generator.tools.query_tools import get_function_code
from fuzz_generator.utils.logger import get_logger

if TYPE_CHECKING:
    from fuzz_generator.config import Settings
    from fuzz_generator.tools.mcp_client import MCPHttpClient

logger = get_logger(__name__)

# Default system prompt for CodeAnalyzer
DEFAULT_SYSTEM_PROMPT = """你是一个专业的代码分析专家，负责分析C/C++函数的结构和参数。

## 你的任务
1. 分析目标函数的代码结构
2. 识别函数的输入参数及其类型
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
"""


class CodeAnalyzerAgent(BaseAgent):
    """Agent for analyzing function code structure.

    This agent uses MCP tools to retrieve function code and LLM
    to analyze parameter information and constraints.
    """

    def __init__(
        self,
        mcp_client: "MCPHttpClient",
        llm_client: Any = None,
        settings: "Settings | None" = None,
        config: AgentConfig | None = None,
    ):
        """Initialize CodeAnalyzer agent.

        Args:
            mcp_client: MCP HTTP client for tool calls
            llm_client: LLM client for analysis (OpenAI-compatible)
            settings: Application settings
            config: Agent configuration
        """
        if config is None:
            config = AgentConfig(
                name="CodeAnalyzer",
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                max_iterations=10,
            )

        super().__init__(config, settings)

        self.mcp_client = mcp_client
        self.llm_client = llm_client
        self._project_name: str = ""

    async def run(
        self,
        project_name: str,
        function_name: str,
        source_file: str | None = None,
        custom_knowledge: str = "",
        **kwargs: Any,
    ) -> AgentResult:
        """Analyze a function and extract its structure.

        Args:
            project_name: Name of the project in Joern
            function_name: Name of the function to analyze
            source_file: Optional source file path
            custom_knowledge: Custom knowledge to inject into prompt
            **kwargs: Additional arguments

        Returns:
            AgentResult with FunctionInfo in data field
        """
        self._project_name = project_name
        logger.info(f"Analyzing function: {function_name}")

        try:
            # Step 1: Get function code
            function_code = await self._get_function_code(function_name, source_file)
            if not function_code:
                return self._create_result(
                    success=False,
                    error=f"Function '{function_name}' not found",
                )

            # Step 2: Analyze with LLM if available
            if self.llm_client:
                function_info = await self._analyze_with_llm(
                    function_name=function_name,
                    function_code=function_code,
                    custom_knowledge=custom_knowledge,
                )
            else:
                # Basic analysis without LLM
                function_info = self._basic_analysis(
                    function_name=function_name,
                    function_code=function_code,
                )

            logger.info(f"Analysis complete for function: {function_name}")

            return self._create_result(
                success=True,
                data=function_info,
                iterations=1,
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self._create_result(
                success=False,
                error=str(e),
            )

    async def _get_function_code(
        self,
        function_name: str,
        source_file: str | None = None,
    ) -> dict[str, Any] | None:
        """Get function code via MCP tool.

        Args:
            function_name: Function name
            source_file: Optional source file

        Returns:
            Function code data or None
        """
        try:
            result = await get_function_code(
                self.mcp_client,
                function_name=function_name,
                project_name=self._project_name,
                file_name=source_file,
            )

            if result.success:
                return {
                    "function_name": result.function_name,
                    "code": result.code,
                    "file": result.file,
                    "line_number": result.line_number,
                    "signature": result.signature,
                }
            return None

        except Exception as e:
            logger.warning(f"Failed to get function code: {e}")
            return None

    async def _analyze_with_llm(
        self,
        function_name: str,
        function_code: dict[str, Any],
        custom_knowledge: str = "",
    ) -> FunctionInfo:
        """Analyze function using LLM.

        Args:
            function_name: Function name
            function_code: Function code data
            custom_knowledge: Custom knowledge to inject

        Returns:
            FunctionInfo with analysis results
        """
        # Render system prompt
        system_prompt = self.render_system_prompt(custom_knowledge=custom_knowledge)

        # Build user message
        user_message = f"""请分析以下函数：

**函数名**: {function_name}
**文件**: {function_code.get("file", "unknown")}
**行号**: {function_code.get("line_number", 0)}

**源代码**:
```c
{function_code.get("code", "")}
```

请输出结构化的JSON格式分析结果。"""

        # Call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = await self._call_llm(messages)
            parsed = self._parse_json_response(response)

            if parsed:
                return self._build_function_info(parsed, function_code)
            else:
                # Fallback to basic analysis
                logger.warning("LLM response not parseable, using basic analysis")
                return self._basic_analysis(function_name, function_code)

        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, using basic analysis")
            return self._basic_analysis(function_name, function_code)

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call LLM and get response.

        Args:
            messages: Chat messages

        Returns:
            Response content
        """
        if hasattr(self.llm_client, "chat"):
            # OpenAI-style client
            response = await self.llm_client.chat.completions.create(
                messages=messages,
                model=self.settings.llm.model if self.settings else "gpt-4",
                temperature=self.settings.llm.temperature if self.settings else 0.7,
            )
            return response.choices[0].message.content
        elif hasattr(self.llm_client, "create"):
            # Alternative interface
            response = self.llm_client.create(messages=messages)
            return response.choices[0].message.content
        else:
            raise AnalysisError("Unsupported LLM client interface")

    def _basic_analysis(
        self,
        function_name: str,
        function_code: dict[str, Any],
    ) -> FunctionInfo:
        """Perform basic analysis without LLM.

        Args:
            function_name: Function name
            function_code: Function code data

        Returns:
            FunctionInfo with basic information
        """
        code = function_code.get("code", "")
        signature = function_code.get("signature", "")

        # Extract basic info from signature/code
        return_type = "unknown"
        parameters: list[ParameterInfo] = []

        # Simple parsing (can be enhanced)
        if signature:
            # Try to extract return type
            parts = signature.split(function_name)
            if len(parts) > 0:
                return_type = parts[0].strip() or "unknown"

        return FunctionInfo(
            name=function_name,
            file_path=function_code.get("file", ""),
            line_number=function_code.get("line_number", 0),
            return_type=return_type,
            parameters=parameters,
            source_code=code,
            description=f"Function {function_name}",
        )

    def _build_function_info(
        self,
        parsed: dict[str, Any],
        function_code: dict[str, Any],
    ) -> FunctionInfo:
        """Build FunctionInfo from parsed LLM response.

        Args:
            parsed: Parsed JSON from LLM
            function_code: Original function code data

        Returns:
            FunctionInfo instance
        """
        # Parse parameters
        parameters: list[ParameterInfo] = []
        for param in parsed.get("parameters", []):
            direction_str = param.get("direction", "in").lower()
            direction = ParameterDirection.IN
            if direction_str == "out":
                direction = ParameterDirection.OUT
            elif direction_str == "inout":
                direction = ParameterDirection.INOUT

            parameters.append(
                ParameterInfo(
                    name=param.get("name", ""),
                    type=param.get("type", "unknown"),
                    direction=direction,
                    description=param.get("description", ""),
                    constraints=param.get("constraints", []),
                )
            )

        return FunctionInfo(
            name=parsed.get("function_name", function_code.get("function_name", "")),
            file_path=parsed.get("file_path", function_code.get("file", "")),
            line_number=parsed.get("line_number", function_code.get("line_number", 0)),
            return_type=parsed.get("return_type", "unknown"),
            parameters=parameters,
            source_code=function_code.get("code", ""),
            description=parsed.get("description", ""),
        )

    async def analyze(
        self,
        project_name: str,
        function_name: str,
        **kwargs: Any,
    ) -> FunctionInfo:
        """Convenience method to analyze a function.

        Args:
            project_name: Project name
            function_name: Function name
            **kwargs: Additional arguments

        Returns:
            FunctionInfo with analysis results

        Raises:
            AnalysisError: If analysis fails
        """
        result = await self.run(
            project_name=project_name,
            function_name=function_name,
            **kwargs,
        )

        if not result.success:
            raise AnalysisError(f"Failed to analyze function '{function_name}': {result.error}")

        return result.data
