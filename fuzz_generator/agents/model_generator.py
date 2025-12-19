"""Model Generator Agent for DataModel generation.

This agent is responsible for:
- Taking analysis context as input
- Generating Secray format XML DataModel
- Applying custom knowledge for domain-specific modeling
- Validating generated DataModel structure
"""

from typing import TYPE_CHECKING, Any

from fuzz_generator.agents.base import AgentConfig, AgentResult, BaseAgent
from fuzz_generator.exceptions import AnalysisError
from fuzz_generator.models import AnalysisContext, GenerationResult
from fuzz_generator.utils.logger import get_logger

if TYPE_CHECKING:
    from fuzz_generator.config import Settings

logger = get_logger(__name__)

# Default system prompt for ModelGenerator
DEFAULT_SYSTEM_PROMPT = """你是一个专业的Fuzz测试数据建模专家，负责根据代码分析结果生成测试数据模型。

## 你的职责
1. 根据函数参数信息设计DataModel结构
2. 为每个参数字段选择合适的数据类型元素
3. 设置适当的约束条件和属性
4. 生成符合Secray格式的XML定义

## Secray DataModel 基础元素

### 1. String 元素
用于表示字符串类型数据：
```xml
<String name="字段名" value="默认值" token="true/false" mutable="true/false" />
```

### 2. Block 元素
用于组合和引用其他DataModel：
```xml
<Block name="块名" ref="引用的DataModel名" minOccurs="0" maxOccurs="1" />
```

### 3. Choice 元素
用于表示多选一的情况：
```xml
<Choice name="选择名">
    <String name="选项1" value="value1" token="true" />
    <String name="选项2" value="value2" token="true" />
</Choice>
```

### 4. Number 元素
用于表示数值类型：
```xml
<Number name="数值名" size="32" signed="true" endian="big" />
```

### 5. Blob 元素
用于表示二进制数据：
```xml
<Blob name="数据名" length="100" />
```

## 输出格式
请直接输出完整的XML内容：

```xml
<?xml version="1.0" encoding="utf-8"?>
<Secray>
    <DataModel name="模型名">
        <!-- 字段定义 -->
    </DataModel>
</Secray>
```
"""


class ModelGeneratorAgent(BaseAgent):
    """Agent for generating DataModel XML.

    This agent uses LLM to generate Secray format DataModel
    based on the analysis context.
    """

    def __init__(
        self,
        llm_client: Any = None,
        settings: "Settings | None" = None,
        config: AgentConfig | None = None,
    ):
        """Initialize ModelGenerator agent.

        Args:
            llm_client: LLM client for generation (OpenAI-compatible)
            settings: Application settings
            config: Agent configuration
        """
        if config is None:
            config = AgentConfig(
                name="ModelGenerator",
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                max_iterations=5,
            )

        super().__init__(config, settings)

        self.llm_client = llm_client

    async def run(
        self,
        context: AnalysisContext,
        custom_knowledge: str = "",
        output_name: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Generate DataModel from analysis context.

        Args:
            context: Analysis context with function and flow info
            custom_knowledge: Custom knowledge to inject
            output_name: Optional name for the generated DataModel
            **kwargs: Additional arguments

        Returns:
            AgentResult with GenerationResult in data field
        """
        function_name = context.function_info.name
        logger.info(f"Generating DataModel for function: {function_name}")

        try:
            # Generate with LLM if available
            if self.llm_client:
                xml_content = await self._generate_with_llm(
                    context=context,
                    custom_knowledge=custom_knowledge,
                    output_name=output_name,
                )
            else:
                # Generate basic template without LLM
                xml_content = self._generate_basic_template(
                    context=context,
                    output_name=output_name,
                )

            # Create generation result
            generation_result = GenerationResult(
                success=True,
                xml_content=xml_content,
                data_models=self._extract_datamodel_info(xml_content),
            )

            logger.info(f"DataModel generated for function: {function_name}")

            return self._create_result(
                success=True,
                data=generation_result,
                iterations=1,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._create_result(
                success=False,
                error=str(e),
            )

    async def _generate_with_llm(
        self,
        context: AnalysisContext,
        custom_knowledge: str = "",
        output_name: str | None = None,
    ) -> str:
        """Generate DataModel using LLM.

        Args:
            context: Analysis context
            custom_knowledge: Custom knowledge to inject
            output_name: Optional DataModel name

        Returns:
            Generated XML content
        """
        # Render system prompt
        system_prompt = self.render_system_prompt(custom_knowledge=custom_knowledge)

        # Build context description
        func_info = context.function_info
        model_name = output_name or f"{func_info.name.title().replace('_', '')}Model"

        # Build user message
        user_message = f"""请根据以下分析结果生成Secray格式的DataModel定义。

## 函数信息
- **函数名**: {func_info.name}
- **文件**: {func_info.file_path}
- **返回类型**: {func_info.return_type}
- **描述**: {func_info.description}

### 参数列表
"""
        for param in func_info.parameters:
            user_message += f"""
- **{param.name}** ({param.type})
  - 方向: {param.direction.value}
  - 描述: {param.description}
  - 约束: {", ".join(param.constraints) if param.constraints else "无"}
"""

        # Add dataflow info
        if context.data_flows:
            user_message += "\n## 数据流信息\n"
            for i, flow in enumerate(context.data_flows[:5], 1):  # Limit to 5
                user_message += (
                    f"- 路径{i}: {flow.source} -> {flow.sink} (长度: {flow.path_length})\n"
                )

        # Add control flow info
        user_message += f"""
## 控制流信息
- 包含循环: {"是" if context.control_flow.has_loops else "否"}
- 包含条件: {"是" if context.control_flow.has_conditions else "否"}
- 复杂度: {context.control_flow.complexity}
"""

        user_message += f"""
## 要求
1. 生成的DataModel名称为: {model_name}
2. 为每个输入参数创建对应的字段
3. 正确设置token和mutable属性
4. 输出完整的XML内容
"""

        # Call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = await self._call_llm(messages)
        xml_content = self._extract_xml_content(response)

        if not xml_content:
            # Use response as-is if no XML block found
            xml_content = response

        return xml_content

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

    def _generate_basic_template(
        self,
        context: AnalysisContext,
        output_name: str | None = None,
    ) -> str:
        """Generate basic DataModel template without LLM.

        Args:
            context: Analysis context
            output_name: Optional DataModel name

        Returns:
            Basic XML template
        """
        func_info = context.function_info
        model_name = output_name or f"{func_info.name.title().replace('_', '')}Model"

        # Build basic structure
        elements = []
        for param in func_info.parameters:
            if "char*" in param.type or "string" in param.type.lower():
                elements.append(f'        <String name="{param.name}" />')
            elif "int" in param.type or "long" in param.type:
                elements.append(f'        <Number name="{param.name}" size="32" signed="true" />')
            elif "*" in param.type:
                elements.append(f'        <Blob name="{param.name}" />')
            else:
                elements.append(f'        <String name="{param.name}" />')

        elements_str = "\n".join(elements) if elements else '        <String name="data" />'

        return f'''<?xml version="1.0" encoding="utf-8"?>
<Secray>
    <DataModel name="{model_name}">
{elements_str}
    </DataModel>
</Secray>'''

    def _extract_datamodel_info(self, xml_content: str) -> list[dict[str, Any]]:
        """Extract DataModel information from XML.

        Args:
            xml_content: XML content

        Returns:
            List of DataModel info dicts
        """
        import re

        data_models = []

        # Simple regex extraction
        pattern = r'<DataModel\s+name="([^"]+)"'
        matches = re.findall(pattern, xml_content)

        for name in matches:
            data_models.append({"name": name})

        return data_models

    async def generate(
        self,
        context: AnalysisContext,
        **kwargs: Any,
    ) -> GenerationResult:
        """Convenience method to generate DataModel.

        Args:
            context: Analysis context
            **kwargs: Additional arguments

        Returns:
            GenerationResult with XML content

        Raises:
            AnalysisError: If generation fails
        """
        result = await self.run(context=context, **kwargs)

        if not result.success:
            raise AnalysisError(f"Failed to generate DataModel: {result.error}")

        return result.data
