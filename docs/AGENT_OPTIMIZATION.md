# Agent 交互优化方案（v2）

> 版本更新：合并 CodeAnalyzer 和 ContextBuilder，支持多轮工具调用

## 1. 当前问题分析

### 1.1 从日志中观察到的问题

#### 问题 1：Agent 对话陷入无效循环
```
Orchestrator: "请把代码粘贴过来..."
ContextBuilder: "请把代码粘贴过来..."
... (重复 30+ 次)
```

**原因**：Agent 没有意识到可以使用工具获取代码

#### 问题 2：工具调用失败后 LLM 讨论错误本身
```
ContextBuilder 调用 get_callees → 返回错误
↓
LLM 开始长篇讨论 Pydantic 版本问题
```

**原因**：Prompt 没有告诉 Agent 如何处理工具错误

#### 问题 3：Agent 职责重叠
- CodeAnalyzer 和 ContextBuilder 都在"收集代码信息"
- 信息在 Agent 之间传递时会丢失上下文
- 增加了不必要的 LLM 调用

#### 问题 4：缺少迭代分析能力
- 当前设计是"单次调用"模式
- 但真实的上下文构建需要根据中间结果决定下一步
- 例如：追踪参数 → 发现调用了其他函数 → 需要继续追踪那个函数的数据流

### 1.2 根本原因

1. **过度拆分职责**：CodeAnalyzer 和 ContextBuilder 的边界不清晰
2. **缺乏迭代能力**：没有支持 Agent 的"探索式"分析
3. **Prompt 设计问题**：没有明确行为边界和工具使用场景

---

## 2. 优化方案（v2）

### 2.1 核心思想：两阶段工作流 + 迭代分析

```
┌─────────────────────────────────────────────────────────────────┐
│                   Two-Phase Workflow                             │
│                                                                 │
│   Phase 1: 分析阶段（可迭代）          Phase 2: 生成阶段         │
│  ┌────────────────────────────┐      ┌──────────────────┐      │
│  │     AnalysisAgent          │      │  ModelGenerator  │      │
│  │                            │      │                  │      │
│  │  ┌─────┐   ┌─────────┐    │      │  [生成 XML]      │      │
│  │  │工具1│ → │分析结果 │────┼──►   │                  │      │
│  │  └─────┘   └────┬────┘    │      │                  │      │
│  │       ↑         │         │      └──────────────────┘      │
│  │       │    需要更多信息?   │                                 │
│  │       │         │         │                                 │
│  │       ↓    Yes  ↓  No     │                                 │
│  │  ┌─────┐   ┌─────────┐    │                                 │
│  │  │工具2│ ← │继续分析 │    │                                 │
│  │  └─────┘   └─────────┘    │                                 │
│  └────────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent 设计

#### 合并为两个 Agent

| Agent | 职责 | 工具 | 特点 |
|-------|------|------|------|
| **AnalysisAgent** | 代码分析 + 上下文构建 | 所有分析工具 | 支持多轮工具调用，自主决策 |
| **ModelGenerator** | 生成 XML DataModel | 无 | 单次调用，基于分析结果 |

#### 为什么合并 CodeAnalyzer 和 ContextBuilder？

1. **减少信息损失**：两者传递信息时，中间结果的细节可能丢失
2. **支持迭代分析**：合并后 Agent 可以根据数据流结果继续深入分析
3. **减少 Token 消耗**：少一次 LLM 调用
4. **简化工作流**：两阶段比三阶段更清晰

### 2.3 AnalysisAgent 的迭代分析流程

```
输入: 函数名 (process_request)
│
├─► Step 1: 获取函数代码
│   └─► 调用 get_function_code("process_request")
│       └─► 结果: 函数源代码、参数列表
│
├─► Step 2: 分析每个参数的数据流
│   └─► 调用 track_dataflow("request_id")
│   └─► 调用 track_dataflow("payload")
│       └─► 结果: 参数在函数内的使用路径
│
├─► Step 3: 获取调用关系
│   └─► 调用 get_callees("process_request")
│       └─► 结果: [handle_service, log_error, send_response]
│
├─► Step 4: 深入分析关键被调函数（迭代）
│   └─► 发现 handle_service 处理核心逻辑
│   └─► 调用 get_function_code("handle_service")
│   └─► 调用 track_dataflow("payload") in handle_service
│       └─► 结果: payload 的完整处理路径
│
├─► Step 5: 获取控制流
│   └─► 调用 get_control_flow_graph("process_request")
│       └─► 结果: 条件分支、循环结构
│
└─► 输出: 完整的分析结果 (JSON)
```

### 2.4 迭代终止条件

Agent 在以下情况停止迭代：

1. **达到最大深度**：追踪的调用链超过 N 层（可配置，默认 3）
2. **信息充足**：所有参数的数据流都已追踪
3. **无新发现**：连续 2 次工具调用没有获得新信息
4. **达到 Token 限制**：接近上下文窗口限制

---

## 3. Prompt 优化（v2）

### 3.1 设计原则

1. **工具驱动**：明确告诉 Agent 必须使用工具获取信息
2. **迭代思维**：指导 Agent 根据中间结果决定下一步
3. **格式约束**：要求固定格式输出，避免闲聊
4. **终止条件**：明确什么时候应该停止分析

### 3.2 AnalysisAgent Prompt

```yaml
# config/defaults/prompts/analysis_agent.yaml
system_prompt: |
  你是代码分析专家。你的任务是分析目标函数，收集完整的上下文信息用于生成 Fuzz 测试数据模型。

  ## 你的目标
  
  收集以下信息：
  1. 函数签名（返回类型、参数列表）
  2. 每个参数的数据流（如何被使用、传递到哪里）
  3. 函数调用关系（调用了哪些函数、被谁调用）
  4. 控制流结构（条件分支、循环）

  ## 可用工具
  
  - `get_function_code(function_name)`: 获取函数源代码
  - `list_functions(file_path)`: 列出文件中的所有函数
  - `track_dataflow(source_pattern)`: 追踪数据流
  - `get_callees(function_name)`: 获取被调用的函数
  - `get_callers(function_name)`: 获取调用者
  - `get_control_flow_graph(function_name)`: 获取控制流图

  ## 分析策略（迭代式）
  
  1. **首先** 调用 `get_function_code` 获取目标函数代码
  2. 从代码中识别参数列表
  3. 对每个参数调用 `track_dataflow` 追踪数据流
  4. 调用 `get_callees` 获取被调用函数
  5. **如果** 发现关键的被调用函数（处理参数的函数），**则**：
     - 获取该函数的代码
     - 继续追踪参数在该函数中的数据流
  6. 调用 `get_control_flow_graph` 获取控制流
  7. 汇总所有信息

  ## 迭代终止条件
  
  - 已追踪所有参数的数据流
  - 调用链深度超过 3 层
  - 连续 2 次工具调用没有新发现

  ## 工具调用规则
  
  - **必须** 在开始时调用工具获取代码，**禁止** 凭空猜测
  - 工具返回错误时，记录错误并尝试其他工具，**不要** 讨论错误本身
  - 每次工具调用后，评估是否需要继续深入分析

  ## 输出格式（JSON）
  
  分析完成后，输出以下格式的 JSON：
  
  ```json
  {
    "status": "success" | "partial" | "error",
    "function": {
      "name": "函数名",
      "return_type": "返回类型",
      "source_code": "源代码"
    },
    "parameters": [
      {
        "name": "参数名",
        "type": "类型",
        "data_flow": ["使用位置1", "使用位置2"],
        "passed_to": ["被传递到的函数"]
      }
    ],
    "callees": [
      {
        "name": "被调用函数",
        "relevance": "high|medium|low",
        "handles_parameters": ["处理的参数"]
      }
    ],
    "callers": ["调用者列表"],
    "control_flow": {
      "conditions": ["条件分支"],
      "loops": ["循环结构"]
    },
    "analysis_depth": 2,
    "errors": ["工具调用错误（如有）"]
  }
  ```

  ## 禁止行为
  
  - **不要** 与用户对话或解释你的思考过程
  - **不要** 询问用户提供代码
  - **不要** 讨论工具调用的错误细节
  - **不要** 在没有调用工具的情况下输出分析结果

  {custom_knowledge}
```

### 3.3 ModelGenerator Prompt

```yaml
# config/defaults/prompts/model_generator.yaml
system_prompt: |
  你是 Fuzz 测试数据模型生成专家。根据代码分析结果生成 XML DataModel。

  ## 输入
  
  你会收到 AnalysisAgent 的分析结果，包含：
  - 函数签名和参数
  - 参数的数据流信息
  - 调用关系和控制流

  ## DataModel 元素类型
  
  | C 类型 | DataModel 元素 | 属性 |
  |--------|---------------|------|
  | `char*`, `const char*` | `<String>` | name, value, maxLength |
  | `int`, `short`, `long` | `<Number>` | name, size, signed, endian |
  | `unsigned int`, `size_t` | `<Number>` | name, size, signed=false |
  | `struct X*` | `<Block>` | name, ref (引用其他 DataModel) |
  | `char[]`, `uint8_t[]` | `<Blob>` | name, length, minLength, maxLength |

  ## 生成规则
  
  1. 每个函数参数 → 一个 DataModel 元素
  2. 利用数据流信息添加约束：
     - 如果参数传递给 `strlen()` → 是字符串
     - 如果参数传递给 `memcpy(dst, src, len)` → Blob，length 关联其他字段
     - 如果参数用于数组索引 → 添加范围约束
  3. 利用控制流信息：
     - 条件检查 `if (len > MAX)` → 添加 maxLength
     - 循环次数 → minOccurs/maxOccurs

  ## 输出格式
  
  直接输出 XML，不要添加任何解释：
  
  ```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <DataModel name="模型名称">
    <!-- 元素定义 -->
  </DataModel>
  ```

  ## 禁止行为
  
  - 只输出 XML，不要解释
  - 不要询问确认
  - 不要添加与 DataModel 无关的内容

  {custom_knowledge}
```

---

## 4. 工作流实现

### 4.1 TwoPhaseWorkflow

```python
class TwoPhaseWorkflow:
    """两阶段分析工作流"""
    
    def __init__(
        self,
        analysis_agent: AnalysisAgent,
        model_generator: ModelGenerator,
        max_analysis_iterations: int = 10,
    ):
        self.analysis_agent = analysis_agent
        self.model_generator = model_generator
        self.max_iterations = max_analysis_iterations
    
    async def run(self, task: AnalysisTask) -> TaskResult:
        # Phase 1: 迭代分析
        analysis_result = await self._run_analysis_phase(task)
        if not analysis_result.success:
            return TaskResult(success=False, error=analysis_result.error)
        
        # Phase 2: 模型生成
        model_result = await self._run_generation_phase(task, analysis_result)
        
        return model_result
    
    async def _run_analysis_phase(self, task: AnalysisTask) -> AnalysisResult:
        """运行分析阶段，支持多轮工具调用"""
        
        # 初始提示
        prompt = f"""
        请分析以下函数：
        - 项目: {task.project_name}
        - 文件: {task.source_file}
        - 函数: {task.function_name}
        
        按照分析策略，使用工具收集完整的上下文信息。
        """
        
        # 使用 AutoGen 的 Agent 运行，允许多轮工具调用
        result = await self.analysis_agent.run(
            prompt,
            max_iterations=self.max_iterations,
        )
        
        return self._parse_analysis_result(result)
    
    async def _run_generation_phase(
        self, task: AnalysisTask, analysis: AnalysisResult
    ) -> TaskResult:
        """运行生成阶段"""
        
        prompt = f"""
        根据以下分析结果生成 DataModel：
        
        {analysis.to_json()}
        
        输出名称: {task.output_name}
        """
        
        result = await self.model_generator.run(prompt)
        
        return self._extract_xml_result(result)
```

### 4.2 AnalysisAgent 实现

```python
class AnalysisAgent:
    """分析 Agent，支持迭代工具调用"""
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        tools: list[Callable],
        system_prompt: str,
    ):
        self.agent = AssistantAgent(
            name="AnalysisAgent",
            model_client=model_client,
            tools=tools,
            system_message=system_prompt,
        )
    
    async def run(self, prompt: str, max_iterations: int = 10) -> str:
        """运行 Agent，支持多轮工具调用
        
        AutoGen 会自动处理：
        1. Agent 决定调用哪个工具
        2. 执行工具调用
        3. 将结果返回给 Agent
        4. Agent 决定是否继续
        """
        # 使用 AutoGen 的 run_stream 或 run 方法
        # 它会自动处理多轮工具调用
        result = await self.agent.run(
            task=prompt,
            cancellation_token=None,
        )
        
        return result.messages[-1].content
```

---

## 5. 错误处理策略

### 5.1 工具错误处理

在 Prompt 中已经指导 Agent：
- 遇到错误时记录并继续
- 不讨论错误细节
- 尝试其他工具

### 5.2 输出验证

```python
class OutputValidator:
    """验证 Agent 输出"""
    
    def validate_analysis(self, output: str) -> tuple[bool, dict | str]:
        """验证分析结果"""
        try:
            # 提取 JSON 部分
            json_match = re.search(r'\{[\s\S]*\}', output)
            if not json_match:
                return False, "No JSON found in output"
            
            data = json.loads(json_match.group())
            
            # 检查必要字段
            if "function" not in data or "parameters" not in data:
                return False, "Missing required fields"
            
            return True, data
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
    
    def validate_xml(self, output: str) -> tuple[bool, str]:
        """验证 XML 输出"""
        xml_match = re.search(r'<DataModel[\s\S]*?</DataModel>', output)
        if not xml_match:
            return False, "No DataModel found"
        
        try:
            ET.fromstring(xml_match.group())
            return True, xml_match.group()
        except ET.ParseError as e:
            return False, f"Invalid XML: {e}"
```

---

## 6. 实现计划

### Phase 1: Agent 合并（1 小时）
- [ ] 创建 `AnalysisAgent` 类
- [ ] 合并 CodeAnalyzer 和 ContextBuilder 的工具
- [ ] 移除 Orchestrator

### Phase 2: 工作流重构（1 小时）
- [ ] 创建 `TwoPhaseWorkflow` 类
- [ ] 实现迭代分析逻辑
- [ ] 集成 AutoGen 的多轮工具调用

### Phase 3: Prompt 优化（0.5 小时）
- [ ] 创建 `analysis_agent.yaml`
- [ ] 更新 `model_generator.yaml`
- [ ] 移除旧的 Prompt 文件

### Phase 4: 测试（1 小时）
- [ ] 单工具调用测试
- [ ] 多轮迭代测试
- [ ] 完整流程测试

---

## 7. 对比分析

| 维度 | 当前设计 | v1 优化 | v2 优化（本方案） |
|------|---------|---------|------------------|
| Agent 数量 | 4 | 3 | **2** |
| 分析模式 | 动态选择 | 顺序执行 | **迭代分析** |
| 工具调用 | 单次 | 单次 | **多轮** |
| 上下文完整性 | 低 | 中 | **高** |
| Token 消耗 | 高 | 低 | 中 |
| 灵活性 | 高（失控） | 低 | **中（可控）** |

---

## 8. 示例场景

### 场景：分析 `process_request` 函数

```
[AnalysisAgent 开始]

Iteration 1:
  → 调用 get_function_code("process_request")
  ← 获取源代码，识别参数: request_id, payload

Iteration 2:
  → 调用 track_dataflow("request_id")
  ← request_id 用于验证

Iteration 3:
  → 调用 track_dataflow("payload")
  ← payload 传递给 handle_service()

Iteration 4:
  → 调用 get_callees("process_request")
  ← 被调用: [handle_service, log_error, send_response]

Iteration 5: (迭代深入)
  → 决定继续分析 handle_service（因为它处理 payload）
  → 调用 get_function_code("handle_service")
  ← 获取 handle_service 代码

Iteration 6:
  → 调用 track_dataflow("payload") in handle_service
  ← payload.data 被解析，payload.length 用于分配内存

Iteration 7:
  → 调用 get_control_flow_graph("process_request")
  ← 获取控制流

[分析完成，输出 JSON]

[ModelGenerator 开始]
  → 输入: 分析结果 JSON
  ← 输出: XML DataModel
```

---

## 9. 注意事项

1. **迭代深度控制**：防止无限递归，设置最大深度
2. **Token 消耗监控**：多轮调用可能消耗较多 Token
3. **中间结果缓存**：避免重复调用相同工具
4. **超时处理**：设置整体超时时间

