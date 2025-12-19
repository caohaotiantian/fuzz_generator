# Fuzz Generator 用户指南

## 目录

1. [简介](#简介)
2. [快速开始](#快速开始)
3. [安装](#安装)
4. [命令行使用](#命令行使用)
5. [任务配置](#任务配置)
6. [批量分析](#批量分析)
7. [自定义配置](#自定义配置)
8. [自定义 Prompt](#自定义-prompt)
9. [输出格式](#输出格式)
10. [故障排除](#故障排除)
11. [FAQ](#faq)

---

## 简介

Fuzz Generator 是一个 AI 驱动的 fuzz 测试数据建模工具。它通过分析 C/C++ 源代码中的函数，自动生成适用于 fuzz 测试的数据模型（Secray XML DataModel 格式）。

### 主要特性

- 🤖 **AI 驱动分析**：使用 LLM 理解代码语义和参数约束
- 🔍 **深度代码分析**：通过 Joern 进行数据流和控制流分析
- 📦 **批量处理**：支持一次分析多个函数
- 💾 **断点续传**：任务中断后可恢复执行
- 🔧 **高度可配置**：支持自定义 Prompt 和领域知识

---

## 快速开始

### 1. 基本流程

```bash
# 1. 创建任务文件
cat > tasks.yaml << EOF
project_path: "./my_project"
tasks:
  - source_file: "handler.c"
    function_name: "process_input"
EOF

# 2. 运行分析
fuzz-generator analyze --task-file tasks.yaml --output output/

# 3. 查看结果
cat output/process_input.xml
```

### 2. 分析单个函数

```bash
fuzz-generator analyze \
  --project ./my_project \
  --source-file handler.c \
  --function process_input \
  --output output/
```

---

## 安装

### 环境要求

- Python 3.10+
- Joern MCP 服务器（用于代码分析）
- 本地 LLM 服务（如 Ollama）或兼容 OpenAI API 的服务

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/fuzz_generator.git
cd fuzz_generator

# 2. 安装依赖
pip install -e .

# 3. 验证安装
fuzz-generator --version
```

### 配置 MCP 服务器

确保 Joern MCP 服务器正在运行：

```bash
# 启动 MCP 服务器（参见 joern_mcp 文档）
cd joern_mcp
python -m joern_mcp --port 8000
```

### 配置 LLM

创建配置文件或设置环境变量：

```bash
# 使用环境变量
export FUZZ_GENERATOR_LLM_BASE_URL="http://localhost:11434/v1"
export FUZZ_GENERATOR_LLM_MODEL="qwen2.5:32b"

# 或使用配置文件
fuzz-generator analyze --config config.yaml ...
```

---

## 命令行使用

### 主命令

```bash
fuzz-generator [OPTIONS] COMMAND [ARGS]...
```

**全局选项**：

| 选项 | 说明 |
|------|------|
| `--config FILE` | 配置文件路径 |
| `--verbose, -v` | 详细输出模式 |
| `--quiet, -q` | 静默模式 |
| `--version` | 显示版本 |
| `--help` | 显示帮助 |

### analyze - 分析命令

分析函数并生成 DataModel：

```bash
# 使用任务文件
fuzz-generator analyze --task-file tasks.yaml --output output/

# 分析单个函数
fuzz-generator analyze \
  --project ./src \
  --source-file main.c \
  --function handle_request \
  --output output/

# 带自定义知识
fuzz-generator analyze \
  --task-file tasks.yaml \
  --knowledge knowledge.md \
  --output output/
```

**选项**：

| 选项 | 说明 |
|------|------|
| `--task-file FILE` | 任务配置文件（YAML/JSON） |
| `--project DIR` | 项目路径 |
| `--source-file FILE` | 源文件（相对于项目） |
| `--function NAME` | 函数名称 |
| `--output DIR` | 输出目录 |
| `--knowledge FILE` | 自定义知识文件 |
| `--resume` | 从上次中断处继续 |

### parse - 解析命令

验证任务文件格式：

```bash
# 解析并验证任务文件
fuzz-generator parse tasks.yaml

# 详细输出
fuzz-generator parse tasks.yaml --verbose
```

### status - 状态命令

查看当前任务状态：

```bash
# 查看所有任务状态
fuzz-generator status

# 查看特定批次
fuzz-generator status --batch-id batch_xxx
```

### results - 结果命令

查看分析结果：

```bash
# 列出所有结果
fuzz-generator results

# 查看特定任务
fuzz-generator results --task-id task_xxx

# 导出结果
fuzz-generator results --export output/
```

### clean - 清理命令

清理缓存和临时文件：

```bash
# 清理缓存
fuzz-generator clean

# 清理所有内容（包括结果）
fuzz-generator clean --all

# 强制清理（无确认）
fuzz-generator clean --force
```

### tools - 工具命令

测试 MCP 工具连接：

```bash
# 列出可用工具
fuzz-generator tools list

# 测试工具调用
fuzz-generator tools test
```

---

## 任务配置

### 任务文件格式

支持 YAML 和 JSON 格式：

```yaml
# tasks.yaml
project_path: "./my_project"
description: "My analysis batch"

tasks:
  - source_file: "handler.c"
    function_name: "process_request"
    output_name: "ProcessRequestModel"  # 可选
    priority: 10                         # 可选，默认 0

  - source_file: "parser.c"
    function_name: "parse_data"
    depends_on:
      - "0"  # 依赖第一个任务

# 配置覆盖（可选）
config_overrides:
  llm:
    temperature: 0.5
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `project_path` | string | 是 | 项目根目录 |
| `description` | string | 否 | 批次描述 |
| `tasks` | list | 是 | 任务列表 |
| `config_overrides` | object | 否 | 配置覆盖 |

**任务字段**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `source_file` | string | 是 | 源文件路径（相对于项目） |
| `function_name` | string | 是 | 函数名称 |
| `output_name` | string | 否 | 输出模型名称 |
| `priority` | int | 否 | 优先级（数字越大优先级越高） |
| `depends_on` | list | 否 | 依赖的任务索引或 ID |

---

## 批量分析

### 基本使用

```bash
# 执行批量分析
fuzz-generator analyze --task-file tasks.yaml --output output/
```

### 并发控制

在配置文件中设置并发数：

```yaml
# config.yaml
batch:
  max_concurrent: 4    # 最大并发任务数
  fail_fast: false     # 单个失败是否终止整个批次
  task_timeout: 600    # 单任务超时（秒）
```

### 断点续传

```bash
# 首次运行
fuzz-generator analyze --task-file tasks.yaml --output output/

# 中断后恢复
fuzz-generator analyze --task-file tasks.yaml --output output/ --resume
```

### 监控进度

```bash
# 实时查看状态
watch -n 5 fuzz-generator status
```

---

## 自定义配置

### 配置文件

创建 `config.yaml`：

```yaml
version: "1.0"

# LLM 配置
llm:
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
  model: "qwen2.5:32b"
  temperature: 0.7
  max_tokens: 4096

# MCP 服务器配置
mcp_server:
  url: "http://localhost:8000/mcp"
  timeout: 60
  retry_count: 3

# Agent 配置
agents:
  code_analyzer:
    max_iterations: 10
  context_builder:
    max_iterations: 15
  model_generator:
    max_iterations: 5

# 批量处理配置
batch:
  max_concurrent: 2
  fail_fast: false

# 存储配置
storage:
  work_dir: ".fuzz_generator"
  enable_cache: true

# 日志配置
logging:
  level: "INFO"
  file: "logs/fuzz_generator.log"

# 输出配置
output:
  format: "xml"
  encoding: "utf-8"
  indent: 4
  include_comments: true
```

### 环境变量

所有配置项都可以通过环境变量覆盖：

```bash
# 格式：FUZZ_GENERATOR_{SECTION}_{KEY}
export FUZZ_GENERATOR_LLM_MODEL="llama3:8b"
export FUZZ_GENERATOR_LLM_TEMPERATURE="0.5"
export FUZZ_GENERATOR_BATCH_MAX_CONCURRENT="4"
```

---

## 自定义 Prompt

### 创建自定义 Prompt

```yaml
# prompts/my_analyzer.yaml
name: "MyCodeAnalyzer"
version: "1.0"

system_prompt: |
  你是一个专业的代码分析专家。
  
  ## 分析目标
  识别函数的输入参数及其约束条件。
  
  ## 输出要求
  以 JSON 格式输出参数信息。

custom_knowledge: |
  {{ custom_knowledge | default('') }}

task_template: |
  分析函数：{{ function_name }}
  文件：{{ source_file }}
```

### 使用自定义 Prompt

```yaml
# config.yaml
agents:
  code_analyzer:
    system_prompt_file: "prompts/my_analyzer.yaml"
```

### 注入领域知识

创建知识文件：

```markdown
# knowledge.md

## 协议规范
- 输入数据遵循 XYZ 协议格式
- 长度字段为大端序

## 安全约束
- buffer 最大长度为 4096 字节
- 需要特别关注边界条件
```

使用知识文件：

```bash
fuzz-generator analyze \
  --task-file tasks.yaml \
  --knowledge knowledge.md \
  --output output/
```

---

## 输出格式

### Secray XML DataModel

生成的输出为 Secray 格式的 XML：

```xml
<?xml version="1.0" encoding="utf-8"?>
<Secray>
    <!-- Process request function model -->
    <DataModel name="ProcessRequestModel">
        <Blob name="buffer" length="256" />
        <Number name="length" size="32" signed="true" />
    </DataModel>
</Secray>
```

### 支持的元素类型

| 元素 | 说明 | 示例 |
|------|------|------|
| `String` | 字符串 | `<String name="method" value="GET" />` |
| `Number` | 数值 | `<Number name="port" size="16" />` |
| `Blob` | 二进制数据 | `<Blob name="data" length="1024" />` |
| `Block` | 块/引用 | `<Block name="header" ref="HeaderLine" />` |
| `Choice` | 选择 | `<Choice name="type">...</Choice>` |

---

## 故障排除

### MCP 服务器连接失败

**症状**：`MCPConnectionError: Failed to connect to MCP server`

**解决方案**：

1. 确认 MCP 服务器已启动：
   ```bash
   curl http://localhost:8000/mcp -X POST \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
   ```

2. 检查配置中的 URL 是否正确
3. 确认网络连接

### LLM 响应超时

**症状**：`LLMError: Request timed out`

**解决方案**：

1. 增加超时时间：
   ```yaml
   llm:
     timeout: 300
   ```

2. 使用更快的模型
3. 减少 `max_tokens`

### 函数未找到

**症状**：`AnalysisError: Function 'xxx' not found`

**解决方案**：

1. 确认函数名称拼写正确（区分大小写）
2. 确认源文件路径正确
3. 尝试列出项目中的函数：
   ```bash
   fuzz-generator tools test --list-functions
   ```

### 生成结果不符合预期

**建议**：

1. 提供更多自定义知识
2. 调整 LLM 参数（降低 temperature）
3. 查看中间结果：
   ```bash
   ls .fuzz_generator/results/
   ```

---

## FAQ

### Q: 支持哪些编程语言？

目前主要支持 C/C++。Joern 也支持其他语言，但尚未在本工具中完全测试。

### Q: 可以使用 OpenAI API 吗？

可以，只要 API 兼容 OpenAI 格式：

```yaml
llm:
  base_url: "https://api.openai.com/v1"
  api_key: "sk-xxx"
  model: "gpt-4"
```

### Q: 如何提高分析质量？

1. 提供详细的领域知识
2. 使用更强大的 LLM 模型
3. 调整 Agent 迭代次数
4. 确保代码有良好的注释

### Q: 缓存存储在哪里？

默认存储在 `.fuzz_generator/` 目录：

```
.fuzz_generator/
├── cache/          # 分析结果缓存
├── results/        # 最终结果
└── logs/           # 日志文件
```

### Q: 如何清除缓存？

```bash
# 清除缓存但保留结果
fuzz-generator clean

# 清除所有内容
fuzz-generator clean --all
```

### Q: 支持多线程/并发吗？

支持，通过配置 `batch.max_concurrent` 控制：

```yaml
batch:
  max_concurrent: 4
```

### Q: 如何贡献代码？

请参阅项目 README 中的贡献指南。

---

## 更多资源

- [技术设计文档](TECHNICAL_DESIGN.md)
- [配置示例](config.example.yaml)
- [示例项目](../examples/)
- [API 文档](API_REFERENCE.md)

