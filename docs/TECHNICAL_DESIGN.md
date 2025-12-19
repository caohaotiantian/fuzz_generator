# Fuzz Generator - 技术方案设计文档

**项目名称**: fuzz_generator  
**版本**: v1.0  
**日期**: 2024-12-19  
**状态**: 草案

---

## 1. 项目概述

### 1.1 项目背景

本项目旨在构建一个基于 AI Agent 的自动化工具，用于分析源代码函数并生成 fuzz 测试所需的数据建模（XML 格式）。系统利用大语言模型（LLM）理解代码语义，结合 Joern 静态分析工具提取代码结构信息，最终生成符合 Secray 格式的 DataModel 定义。

### 1.2 核心目标

- **自动化分析**：自动解析目标函数的入参、出参及其数据结构
- **批量处理**：支持一次分析多个函数，提高效率
- **深度代码理解**：通过数据流、控制流分析获取完整的代码上下文
- **智能建模生成**：基于分析结果生成符合规范的 fuzz 测试 DataModel
- **可配置性**：支持通过配置文件自定义 prompt、模型参数等
- **可追溯性**：分析中间结果持久化，便于问题定位和结果复用

### 1.3 系统边界

**输入**：

- 源代码文件夹路径
- 待分析的函数列表（支持批量，JSON/YAML 格式）
- 可配置的背景知识 prompt

**输入格式示例（批量分析）**：

```yaml
# analysis_tasks.yaml
project_path: "/path/to/source"
tasks:
  - source_file: "rtsp/handler.c"
    function_name: "process_request"
    output_name: "RtspRequest" # 可选，DataModel名称
  - source_file: "rtsp/session.c"
    function_name: "handle_options"
    output_name: "OptionsRequest"
  - source_file: "rtsp/parser.c"
    function_name: "parse_header"
```

**输出**：

- XML 格式的 Secray DataModel 定义（仅 DataModel 部分）
- 分析中间结果（JSON 格式，便于复用）
- 分析过程日志

---

## 2. 技术选型

### 2.1 核心框架与工具

| 组件              | 选型                | 版本   | 说明                         |
| ----------------- | ------------------- | ------ | ---------------------------- |
| **AI Agent 框架** | AutoGen (AgentChat) | ≥0.4.x | 微软官方 Multi-Agent 框架    |
| **代码分析工具**  | Joern MCP Server    | 本地   | 基于 Joern 的 MCP 协议服务器 |
| **MCP 通信**      | HTTP 传输           | -      | streamable-http 模式         |
| **LLM 服务**      | OpenAI 兼容 API     | -      | 本地部署模型                 |
| **编程语言**      | Python              | ≥3.10  | 主开发语言                   |

> **注意**：AutoGen 0.4+ 版本采用新的包结构，参考 [官方安装指南](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/installation.html)

### 2.2 依赖库

```
# requirements.txt

# AutoGen 官方包 (0.4+ 新架构)
# 参考: https://microsoft.github.io/autogen/stable/reference/index.html
autogen-agentchat>=0.4.0        # AgentChat API - 多Agent对话
autogen-ext[openai]>=0.4.0      # 扩展包 - 包含OpenAI客户端

# HTTP客户端
httpx>=0.25.0

# 数据验证
pydantic>=2.0.0

# 配置管理
pyyaml>=6.0

# CLI框架
click>=8.0.0

# 日志
loguru>=0.7.0

# 模板引擎
jinja2>=3.0.0

# XML处理
lxml>=5.0.0
```

### 2.3 AutoGen 包说明

根据 [AutoGen GitHub](https://github.com/microsoft/autogen) 官方说明：

| 包名                | 用途                         | PyPI                                                             |
| ------------------- | ---------------------------- | ---------------------------------------------------------------- |
| `autogen-core`      | 核心 API，消息传递、事件驱动 | [autogen-core](https://pypi.org/project/autogen-core/)           |
| `autogen-agentchat` | AgentChat API，多 Agent 对话 | [autogen-agentchat](https://pypi.org/project/autogen-agentchat/) |
| `autogen-ext`       | 扩展 API，LLM 客户端等       | [autogen-ext](https://pypi.org/project/autogen-ext/)             |

本项目主要使用 `autogen-agentchat` 进行多 Agent 协作开发。

### 2.4 技术架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI Interface                                │
│                    (Click-based Command Line)                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Orchestrator Agent                              │
│              (协调多Agent工作流，管理分析任务)                         │
└─────────────────────────────────────────────────────────────────────┘
            │                       │                       │
            ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│  Code Analyzer    │   │  Context Builder  │   │  Model Generator  │
│      Agent        │   │      Agent        │   │      Agent        │
│                   │   │                   │   │                   │
│ • 函数解析        │   │ • 数据流追踪      │   │ • DataModel生成   │
│ • 参数分析        │   │ • 控制流分析      │   │ • XML格式化       │
│ • 类型推断        │   │ • 依赖关系提取    │   │ • 模型验证        │
└───────────────────┘   └───────────────────┘   └───────────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Tool Layer                                    │
│                   (MCP Tools Wrapper)                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ ProjectTools │  │AnalysisTools │  │ QueryTools   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HTTP (streamable-http)
┌─────────────────────────────────────────────────────────────────────┐
│                      Joern MCP Server                                │
│                    (localhost:8000/mcp)                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Joern Server                                  │
│                   (Code Property Graph)                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 系统架构设计

### 3.1 模块划分

```
fuzz_generator/
├── __init__.py
├── __main__.py                 # CLI入口
├── cli/                        # 命令行接口模块
│   ├── __init__.py
│   ├── commands.py             # Click命令定义
│   └── validators.py           # 输入验证器
├── config/                     # 配置管理模块
│   ├── __init__.py
│   ├── settings.py             # 配置模型定义
│   ├── loader.py               # 配置加载器
│   └── defaults/               # 默认配置
│       ├── config.yaml         # 主配置文件
│       └── prompts/            # Prompt模板目录
│           ├── orchestrator.yaml
│           ├── code_analyzer.yaml
│           ├── context_builder.yaml
│           └── model_generator.yaml
├── agents/                     # Agent模块
│   ├── __init__.py
│   ├── base.py                 # Agent基类
│   ├── orchestrator.py         # 编排Agent
│   ├── code_analyzer.py        # 代码分析Agent
│   ├── context_builder.py      # 上下文构建Agent
│   └── model_generator.py      # 模型生成Agent
├── tools/                      # 工具封装模块
│   ├── __init__.py
│   ├── mcp_client.py           # MCP HTTP客户端
│   ├── project_tools.py        # 项目管理工具
│   ├── analysis_tools.py       # 分析工具
│   └── query_tools.py          # 查询工具
├── models/                     # 数据模型模块
│   ├── __init__.py
│   ├── function_info.py        # 函数信息模型
│   ├── xml_models.py           # XML DataModel模型
│   ├── task.py                 # 分析任务模型
│   └── analysis_result.py      # 分析结果模型（含数据流模型）
# 注：dataflow_info.py 的内容已合并到 analysis_result.py 中，
# 包括 DataFlowPath, DataFlowNode, ControlFlowInfo, CallGraphInfo 等类
├── storage/                    # 持久化存储模块
│   ├── __init__.py
│   ├── base.py                 # 存储接口定义
│   ├── json_storage.py         # JSON文件存储实现
│   └── cache.py                # 结果缓存管理
├── batch/                      # 批量任务处理模块
│   ├── __init__.py
│   ├── parser.py               # 任务文件解析器
│   ├── executor.py             # 批量执行器
│   └── state.py                # 任务状态管理
├── generators/                 # 生成器模块
│   ├── __init__.py
│   ├── xml_generator.py        # XML生成器
│   └── templates/              # XML模板
│       └── datamodel.xml.j2
└── utils/                      # 工具模块
    ├── __init__.py
    ├── logger.py               # 日志工具
    ├── xml_utils.py            # XML处理工具
    └── validators.py           # 通用验证器

# 测试模块（项目根目录）
tests/
├── conftest.py                 # pytest配置和共享fixture
├── test_phase1/                # Phase 1 单元测试
├── test_phase2/                # Phase 2 单元测试
├── test_phase3/                # Phase 3 单元测试
├── test_phase4/                # Phase 4 单元测试
├── test_phase5/                # Phase 5 单元测试
├── integration/                # 集成测试
└── performance/                # 性能测试
```

### 3.2 模块职责说明

#### 3.2.1 CLI 模块 (`cli/`)

负责命令行交互，提供用户友好的接口。

**主要功能**：

- 解析命令行参数
- 输入验证（路径存在性、文件有效性等）
- 进度显示和结果输出

#### 3.2.2 配置模块 (`config/`)

管理所有可配置项，支持 YAML 格式配置文件。

**配置项包括**：

- LLM 连接配置（API 地址、密钥、模型名称）
- MCP 服务器配置（地址、超时时间）
- Agent 配置（系统提示词、温度参数等）
- Prompt 模板配置

#### 3.2.3 Agent 模块 (`agents/`)

实现基于 AutoGen AgentChat 的多 Agent 协作系统。

**Agent 角色**：
| Agent | 职责 | 工具权限 |
|-------|------|----------|
| Orchestrator | 任务协调、流程控制、批量任务管理 | 无直接工具调用 |
| CodeAnalyzer | 函数代码解析、参数分析 | get_function_code, list_functions, search_code |
| ContextBuilder | 数据流/控制流分析 | track_dataflow, get_callees, get_callers, get_control_flow_graph |
| ModelGenerator | DataModel 生成 | 无（基于上下文生成） |

#### 3.2.4 工具模块 (`tools/`)

封装与 Joern MCP Server 的 HTTP 交互。

**工具分类**：

- **项目工具**：parse_project, list_projects, switch_project
- **分析工具**：track_dataflow, analyze_variable_flow, find_data_dependencies
- **查询工具**：get_function_code, list_functions, search_code, execute_query

#### 3.2.5 模型模块 (`models/`)

定义数据结构，使用 Pydantic 进行验证。

#### 3.2.6 存储模块 (`storage/`)

负责分析中间结果的持久化存储。

**主要功能**：

- 保存分析任务状态和进度
- 缓存函数分析结果（避免重复分析）
- 存储数据流/控制流分析结果
- 支持断点续传（任务中断后可恢复）

#### 3.2.7 批量任务模块 (`batch/`)

负责批量分析任务的解析、执行和状态管理。

**主要功能**：

- 解析 YAML/JSON 格式的任务文件
- 批量执行分析任务（支持顺序/并发）
- 管理任务状态（pending/running/completed/failed）
- 支持断点续传和进度报告

#### 3.2.8 生成器模块 (`generators/`)

负责最终 XML DataModel 的生成和格式化（仅生成 DataModel 部分，不包含 StateModel 和 Test）。

**存储结构**：

```
.fuzz_generator/                    # 工作目录
├── cache/                          # 缓存目录
│   └── {project_hash}/             # 项目级别缓存
│       ├── functions/              # 函数分析缓存
│       │   └── {func_hash}.json
│       └── dataflow/               # 数据流分析缓存
│           └── {flow_hash}.json
├── results/                        # 分析结果目录
│   └── {task_id}/                  # 任务结果
│       ├── task_meta.json          # 任务元信息
│       ├── intermediate/           # 中间结果
│       │   ├── code_analysis.json
│       │   ├── context_info.json
│       │   └── agent_conversations.json
│       └── output/                 # 最终输出
│           └── datamodel.xml
└── logs/                           # 日志目录
    └── {task_id}.log
```

---

## 4. 详细设计

### 4.1 配置系统设计

#### 4.1.1 配置文件结构

```yaml
# config.yaml
version: "1.0"

# LLM配置
llm:
  base_url: "http://localhost:11434/v1" # 本地模型API地址
  api_key: "ollama" # API密钥（本地可为任意值）
  model: "qwen2.5:32b" # 模型名称
  temperature: 0.7
  max_tokens: 4096
  timeout: 120

# MCP服务器配置
mcp_server:
  url: "http://localhost:8000/mcp"
  timeout: 60
  retry_count: 3
  retry_delay: 2

# Agent配置
agents:
  orchestrator:
    system_prompt_file: "prompts/orchestrator.yaml"
  code_analyzer:
    system_prompt_file: "prompts/code_analyzer.yaml"
    max_iterations: 10
  context_builder:
    system_prompt_file: "prompts/context_builder.yaml"
    max_iterations: 15
  model_generator:
    system_prompt_file: "prompts/model_generator.yaml"
    max_iterations: 5

# 输出配置
output:
  format: "xml"
  encoding: "utf-8"
  indent: 4
  include_comments: true

# 批量任务配置
batch:
  # 并发执行的最大任务数（0表示顺序执行）
  max_concurrent: 1
  # 任务失败后是否继续执行其他任务
  fail_fast: false
  # 任务超时时间（秒）
  task_timeout: 600
  # 是否启用断点续传
  enable_resume: true

# 日志配置
logging:
  level: "INFO"
  file: "logs/fuzz_generator.log"
  rotation: "10 MB"
  retention: "7 days"
```

#### 4.1.2 Prompt 模板设计

Prompt 使用 YAML 格式，支持 Jinja2 模板语法：

```yaml
# prompts/code_analyzer.yaml
name: "CodeAnalyzer"
version: "1.0"

system_prompt: |
  你是一个专业的代码分析专家，负责分析C/C++函数的结构和参数。

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
  请以结构化的JSON格式输出分析结果。

# 自定义背景知识（可由用户配置）
custom_knowledge: |
  {{ custom_knowledge | default('') }}

# 任务模板
task_template: |
  请分析以下函数：
  - 项目路径: {{ project_path }}
  - 源文件: {{ source_file }}
  - 函数名: {{ function_name }}

  {{ additional_context | default('') }}
```

### 4.2 Agent 协作设计

#### 4.2.1 协作流程

```
                        ┌─────────────────┐
                        │  User Request   │
                        │  (CLI Input)    │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Orchestrator   │
                        │    (协调)       │
                        └────────┬────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
           ▼                     │                     │
    ┌─────────────┐              │                     │
    │ Phase 1:    │              │                     │
    │ 代码解析    │              │                     │
    │             │              │                     │
    │ CodeAnalyzer│◄─────────────┤                     │
    │   Agent     │   (多轮对话)  │                     │
    └──────┬──────┘              │                     │
           │                     │                     │
           │ 函数结构信息        │                     │
           ▼                     │                     │
    ┌─────────────┐              │                     │
    │ Phase 2:    │              │                     │
    │ 上下文构建  │              │                     │
    │             │              │                     │
    │ContextBuilder◄─────────────┤                     │
    │   Agent     │   (多轮对话)  │                     │
    └──────┬──────┘              │                     │
           │                     │                     │
           │ 数据流/控制流信息   │                     │
           ▼                     │                     │
    ┌─────────────┐              │                     │
    │ Phase 3:    │              │                     │
    │ 模型生成    │              │                     │
    │             │◄─────────────┘                     │
    │ModelGenerator                                    │
    │   Agent     │◄───────────────────────────────────┘
    └──────┬──────┘        (汇总所有信息)
           │
           │ XML DataModel
           ▼
    ┌─────────────┐
    │   Output    │
    │  (XML File) │
    └─────────────┘
```

#### 4.2.2 Agent 间消息传递

使用 pyautogen 的 GroupChat 机制实现 Agent 协作：

```python
# agents/orchestrator.py 伪代码
class OrchestratorAgent:
    def create_group_chat(self):
        return GroupChat(
            agents=[
                self.orchestrator,
                self.code_analyzer,
                self.context_builder,
                self.model_generator
            ],
            messages=[],
            max_round=50,
            speaker_selection_method="auto"
        )
```

#### 4.2.3 工具调用流程

每个 Agent 通过注册的工具函数与 Joern MCP Server 交互：

```python
# 工具注册示例
@user_proxy.register_for_execution()
@code_analyzer.register_for_llm(description="获取指定函数的源代码")
async def get_function_code(function_name: str, project_name: str) -> dict:
    """获取函数源代码"""
    return await mcp_client.call_tool("get_function_code", {
        "function_name": function_name,
        "project_name": project_name
    })
```

### 4.3 MCP 客户端设计

#### 4.3.1 HTTP 客户端封装

````python
# tools/mcp_client.py
from dataclasses import dataclass
from typing import Any
import httpx

@dataclass
class MCPClientConfig:
    url: str
    timeout: int = 60
    retry_count: int = 3
    retry_delay: float = 2.0

class MCPHttpClient:
    """Joern MCP Server HTTP客户端"""

    def __init__(self, config: MCPClientConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.config.url,
            timeout=self.config.timeout
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict:
        """调用MCP工具"""
        # 实现MCP协议的工具调用
        ...

    async def list_tools(self) -> list[dict]:
        """列出可用工具"""
        ...

#### 4.3.2 MCP协议交互

MCP协议使用JSON-RPC 2.0格式进行通信：

**请求格式**：
```json
{
    "jsonrpc": "2.0",
    "id": "request-id",
    "method": "tools/call",
    "params": {
        "name": "get_function_code",
        "arguments": {
            "function_name": "process_request",
            "project_name": "my_project"
        }
    }
}
````

**响应格式**：

```json
{
  "jsonrpc": "2.0",
  "id": "request-id",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"success\": true, \"code\": \"...\"}"
      }
    ]
  }
}
```

**错误处理**：

```json
{
  "jsonrpc": "2.0",
  "id": "request-id",
  "error": {
    "code": -32600,
    "message": "Invalid Request",
    "data": { "details": "..." }
  }
}
```

````

### 4.4 XML DataModel生成设计

#### 4.4.1 数据模型定义

```python
# models/xml_models.py
from pydantic import BaseModel
from typing import Literal

class StringElement(BaseModel):
    """String元素定义"""
    name: str
    value: str | None = None
    token: bool = False
    mutable: bool = True

class ChoiceElement(BaseModel):
    """Choice元素定义"""
    name: str
    options: list[StringElement]

class BlockElement(BaseModel):
    """Block元素定义"""
    name: str
    ref: str | None = None
    min_occurs: int = 1
    max_occurs: int | str = 1  # 可以是数字或"unbounded"
    children: list["DataModelElement"] = []

DataModelElement = StringElement | ChoiceElement | BlockElement

class NumberElement(BaseModel):
    """Number元素定义"""
    name: str
    size: int = 32                     # 位数: 8, 16, 32, 64
    signed: bool = True                # 是否有符号
    endian: Literal["big", "little"] = "big"
    value: int | None = None

class BlobElement(BaseModel):
    """Blob元素定义（二进制数据）"""
    name: str
    length: int | str | None = None    # 固定长度或引用其他字段
    min_length: int | None = None
    max_length: int | None = None

DataModelElement = StringElement | ChoiceElement | BlockElement | NumberElement | BlobElement

class DataModel(BaseModel):
    """DataModel定义"""
    name: str
    elements: list[DataModelElement]
````

#### 4.4.2 XML 生成器

```python
# generators/xml_generator.py
from jinja2 import Environment, FileSystemLoader
import xml.etree.ElementTree as ET
from xml.dom import minidom

class XMLGenerator:
    """XML DataModel生成器"""

    def __init__(self, template_dir: str):
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def generate(self, data_models: list[DataModel]) -> str:
        """生成XML字符串"""
        template = self.env.get_template("datamodel.xml.j2")
        xml_str = template.render(data_models=data_models)
        return self._format_xml(xml_str)

    def _format_xml(self, xml_str: str) -> str:
        """格式化XML"""
        parsed = minidom.parseString(xml_str)
        return parsed.toprettyxml(indent="    ")
```

### 4.5 命令行接口设计

```python
# cli/commands.py
import click
from pathlib import Path

@click.group()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='配置文件路径')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
@click.option('--work-dir', '-w', type=click.Path(),
              default='.fuzz_generator', help='工作目录（存储中间结果）')
@click.pass_context
def cli(ctx, config, verbose, work_dir):
    """Fuzz Generator - Fuzz测试数据建模工具"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    ctx.obj['work_dir'] = work_dir

@cli.command()
@click.option('--project-path', '-p', required=True,
              type=click.Path(exists=True),
              help='源代码项目路径')
@click.option('--source-file', '-f',
              help='待分析的源文件（单函数模式，相对于项目路径）')
@click.option('--function', '-fn',
              help='待分析的函数名（单函数模式）')
@click.option('--task-file', '-t', type=click.Path(exists=True),
              help='批量分析任务文件（YAML/JSON格式）')
@click.option('--output', '-o', type=click.Path(),
              help='输出目录或文件路径')
@click.option('--knowledge-file', '-k', type=click.Path(exists=True),
              help='自定义背景知识文件')
@click.option('--resume', is_flag=True,
              help='从上次中断处继续执行')
@click.pass_context
async def analyze(ctx, project_path, source_file, function, task_file,
                  output, knowledge_file, resume):
    """分析函数并生成DataModel

    支持两种模式：
    1. 单函数模式：使用 -f 和 -fn 参数指定单个函数
    2. 批量模式：使用 -t 参数指定任务文件
    """
    # 实现分析逻辑
    ...

@cli.command()
@click.option('--project-path', '-p', required=True,
              type=click.Path(exists=True))
@click.pass_context
async def parse(ctx, project_path):
    """解析项目生成CPG（预处理步骤）"""
    ...

@cli.command()
@click.option('--task-id', '-t', help='查看指定任务的结果')
@click.option('--list', '-l', 'list_all', is_flag=True, help='列出所有任务')
@click.pass_context
def results(ctx, task_id, list_all):
    """查看分析结果和中间数据"""
    ...

@cli.command()
@click.option('--all', '-a', 'clear_all', is_flag=True, help='清理所有缓存')
@click.option('--task-id', '-t', help='清理指定任务')
@click.pass_context
def clean(ctx, clear_all, task_id):
    """清理缓存和中间结果"""
    ...

@cli.command()
def list_tools():
    """列出可用的MCP工具"""
    ...
```

**使用示例**：

```bash
# 单函数分析模式
python -m fuzz_generator analyze \
    --project-path /path/to/source \
    --source-file rtsp/handler.c \
    --function process_request \
    --output output/request_model.xml \
    --knowledge-file knowledge/rtsp.md

# 批量分析模式（推荐）
python -m fuzz_generator analyze \
    --project-path /path/to/source \
    --task-file tasks.yaml \
    --output output/ \
    --knowledge-file knowledge/rtsp.md

# 从中断处恢复执行
python -m fuzz_generator analyze \
    --project-path /path/to/source \
    --task-file tasks.yaml \
    --resume

# 仅解析项目（构建CPG）
python -m fuzz_generator parse --project-path /path/to/source

# 查看分析结果
python -m fuzz_generator results --list
python -m fuzz_generator results --task-id abc123

# 清理缓存
python -m fuzz_generator clean --all

# 使用自定义配置
python -m fuzz_generator -c custom_config.yaml analyze ...
```

### 4.6 批量任务文件格式

支持 YAML 和 JSON 两种格式：

**YAML 格式 (tasks.yaml)**：

```yaml
# 项目基础信息
project_path: "/path/to/source" # 可在命令行覆盖
description: "RTSP协议处理函数分析"

# 分析任务列表
tasks:
  - source_file: "rtsp/handler.c"
    function_name: "process_request"
    output_name: "RtspRequest" # 生成的DataModel名称
    priority: 1 # 可选，执行优先级

  - source_file: "rtsp/handler.c"
    function_name: "handle_options"
    output_name: "OptionsRequest"
    depends_on: ["process_request"] # 可选，依赖关系

  - source_file: "rtsp/parser.c"
    function_name: "parse_header"
    output_name: "HeaderLine"

  - source_file: "rtsp/session.c"
    function_name: "create_session"
    output_name: "SessionData"

# 可选的全局配置覆盖
config_overrides:
  llm:
    temperature: 0.5
  analysis:
    max_dataflow_depth: 15
```

**JSON 格式 (tasks.json)**：

```json
{
  "project_path": "/path/to/source",
  "description": "RTSP协议处理函数分析",
  "tasks": [
    {
      "source_file": "rtsp/handler.c",
      "function_name": "process_request",
      "output_name": "RtspRequest"
    },
    {
      "source_file": "rtsp/handler.c",
      "function_name": "handle_options",
      "output_name": "OptionsRequest"
    }
  ]
}
```

### 4.7 中间结果持久化设计

#### 4.7.1 缓存策略

**缓存键计算**：

```python
import hashlib

def compute_cache_key(project_path: str, function_name: str, source_content: str) -> str:
    """计算缓存键

    基于项目路径、函数名和源代码内容生成唯一键
    当源代码变更时，缓存自动失效
    """
    content = f"{project_path}:{function_name}:{source_content}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

**缓存失效策略**：

- **代码变更失效**：源代码内容变更时自动失效
- **时间过期失效**：可配置过期时间（默认永不过期）
- **手动失效**：支持通过 CLI 命令清理缓存

**缓存分类**：

- `functions/`：函数分析结果缓存
- `dataflow/`：数据流分析结果缓存
- `callgraph/`：调用图分析结果缓存

#### 4.7.2 存储接口

```python
# storage/base.py
from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel

class AnalysisCache(BaseModel):
    """分析缓存条目"""
    key: str
    data: dict
    created_at: str
    expires_at: Optional[str] = None

class StorageBackend(ABC):
    """存储后端抽象接口"""

    @abstractmethod
    async def save(self, category: str, key: str, data: Any) -> None:
        """保存数据"""
        pass

    @abstractmethod
    async def load(self, category: str, key: str) -> Optional[Any]:
        """加载数据"""
        pass

    @abstractmethod
    async def exists(self, category: str, key: str) -> bool:
        """检查是否存在"""
        pass

    @abstractmethod
    async def delete(self, category: str, key: str) -> None:
        """删除数据"""
        pass

    @abstractmethod
    async def list_keys(self, category: str) -> list[str]:
        """列出所有键"""
        pass
```

#### 4.7.3 任务状态管理

```python
# models/task.py
from enum import Enum
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AnalysisTask(BaseModel):
    """单个分析任务"""
    task_id: str
    source_file: str
    function_name: str
    output_name: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class BatchTask(BaseModel):
    """批量任务"""
    batch_id: str
    project_path: str
    description: Optional[str] = None
    tasks: list[AnalysisTask]
    created_at: datetime
    completed_count: int = 0
    failed_count: int = 0

class IntermediateResult(BaseModel):
    """中间结果"""
    task_id: str
    stage: str  # code_analysis, context_building, model_generation
    data: dict
    timestamp: datetime
```

#### 4.7.4 中间结果内容

每个分析任务会保存以下中间结果：

| 阶段       | 文件名                     | 内容                              |
| ---------- | -------------------------- | --------------------------------- |
| 代码分析   | `code_analysis.json`       | 函数签名、参数列表、代码片段      |
| 上下文构建 | `context_info.json`        | 数据流路径、控制流信息、调用关系  |
| Agent 对话 | `agent_conversations.json` | 完整的 Agent 对话记录（用于调试） |
| 模型生成   | `model_draft.json`         | DataModel 的中间表示              |
| 最终输出   | `datamodel.xml`            | 最终生成的 XML                    |

---

## 5. 数据流设计

### 5.1 分析流程时序图

```
User        CLI        Orchestrator    CodeAnalyzer    ContextBuilder    ModelGenerator    MCPServer
 │           │              │               │                │                 │               │
 │──analyze──►│              │               │                │                 │               │
 │           │──init────────►│               │                │                 │               │
 │           │              │───parse_proj──────────────────────────────────────────────────────►│
 │           │              │◄──────────────────────────────────────────────────────success─────│
 │           │              │               │                │                 │               │
 │           │              │──────────────►│                │                 │               │
 │           │              │  analyze_code │                │                 │               │
 │           │              │               │──get_func_code─────────────────────────────────────►│
 │           │              │               │◄───────────────────────────────────code────────────│
 │           │              │               │    (多轮)      │                 │               │
 │           │              │◄──func_info───│                │                 │               │
 │           │              │               │                │                 │               │
 │           │              │──────────────────────────────►│                 │               │
 │           │              │         build_context         │                 │               │
 │           │              │               │                │──track_dataflow────────────────────►│
 │           │              │               │                │◄───────────────────flows──────────│
 │           │              │               │                │──get_callers───────────────────────►│
 │           │              │               │                │◄───────────────────callers────────│
 │           │              │               │                │    (多轮)       │               │
 │           │              │◄────────────────context_info──│                 │               │
 │           │              │               │                │                 │               │
 │           │              │────────────────────────────────────────────────►│               │
 │           │              │                  generate_model                  │               │
 │           │              │               │                │                 │    (多轮)     │
 │           │              │◄──────────────────────────────────xml_model─────│               │
 │           │◄─────────────│               │                │                 │               │
 │◄──XML─────│              │               │                │                 │               │
 │           │              │               │                │                 │               │
```

### 5.2 核心数据结构

```python
# models/analysis_result.py
from pydantic import BaseModel
from typing import Any
from datetime import datetime

class ParameterInfo(BaseModel):
    """函数参数信息"""
    name: str
    type: str
    direction: Literal["in", "out", "inout"]
    constraints: list[str] = []  # 约束条件
    description: str = ""

class FunctionInfo(BaseModel):
    """函数分析结果"""
    name: str
    file_path: str
    line_number: int
    return_type: str
    parameters: list[ParameterInfo]
    source_code: str
    description: str = ""

class DataFlowPath(BaseModel):
    """数据流路径"""
    source: dict[str, Any]
    sink: dict[str, Any]
    path_length: int
    path_details: list[dict[str, Any]] = []

class ControlFlowInfo(BaseModel):
    """控制流信息"""
    has_loops: bool
    has_conditions: bool
    branches: list[dict[str, Any]] = []
    complexity: int = 0

class AnalysisContext(BaseModel):
    """完整分析上下文"""
    function_info: FunctionInfo
    data_flows: list[DataFlowPath]
    control_flow: ControlFlowInfo
    call_graph: dict[str, Any]
    dependencies: list[str]
    timestamp: datetime

class GenerationResult(BaseModel):
    """生成结果"""
    success: bool
    xml_content: str | None
    data_models: list[dict[str, Any]]
    errors: list[str] = []
    warnings: list[str] = []
```

---

## 6. 错误处理设计

### 6.1 错误分类

| 错误类型           | 描述               | 处理策略               |
| ------------------ | ------------------ | ---------------------- |
| ConfigError        | 配置错误           | 终止并提示修正         |
| MCPConnectionError | MCP 服务器连接失败 | 重试 N 次后终止        |
| MCPToolError       | MCP 工具调用失败   | 记录日志，尝试替代方案 |
| LLMError           | LLM 服务错误       | 重试或降级处理         |
| AnalysisError      | 代码分析失败       | 记录详情，生成部分结果 |
| ValidationError    | 输出验证失败       | 重新生成或人工干预     |

### 6.2 错误处理流程

```python
# utils/error_handler.py
from enum import Enum
from typing import Callable
import functools

class ErrorSeverity(Enum):
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

class FuzzGeneratorError(Exception):
    """基础异常类"""
    severity: ErrorSeverity = ErrorSeverity.ERROR

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

class MCPConnectionError(FuzzGeneratorError):
    severity = ErrorSeverity.FATAL

class AnalysisError(FuzzGeneratorError):
    severity = ErrorSeverity.ERROR

def with_retry(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            raise last_error
        return wrapper
    return decorator
```

---

## 7. 日志与监控

### 7.1 日志设计

```python
# utils/logger.py
from loguru import logger
import sys

def setup_logger(log_level: str = "INFO", log_file: str = None):
    """配置日志"""
    logger.remove()

    # 控制台输出
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>"
    )

    # 文件输出
    if log_file:
        logger.add(
            log_file,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} | {message}"
        )

    return logger
```

### 7.2 日志内容规范

```
# 分析开始
2024-12-19 10:00:00 | INFO     | Starting analysis for function: process_request
2024-12-19 10:00:00 | INFO     | Project path: /path/to/source
2024-12-19 10:00:00 | DEBUG    | Loading config from: config.yaml

# MCP工具调用
2024-12-19 10:00:01 | DEBUG    | MCP Tool Call: parse_project
2024-12-19 10:00:05 | INFO     | Project parsed successfully: my-project
2024-12-19 10:00:05 | DEBUG    | MCP Tool Call: get_function_code(process_request)
2024-12-19 10:00:06 | DEBUG    | Function code retrieved: 45 lines

# Agent交互
2024-12-19 10:00:07 | INFO     | CodeAnalyzer: Starting function analysis
2024-12-19 10:00:10 | DEBUG    | CodeAnalyzer: Identified 3 parameters
2024-12-19 10:00:15 | INFO     | ContextBuilder: Building dataflow context
2024-12-19 10:00:20 | DEBUG    | ContextBuilder: Found 5 dataflow paths

# 生成结果
2024-12-19 10:00:30 | INFO     | ModelGenerator: Generating DataModel
2024-12-19 10:00:35 | INFO     | Generated 3 DataModel definitions
2024-12-19 10:00:36 | INFO     | Output saved to: output/request_model.xml
```

---

## 8. 测试策略

### 8.1 测试分层

| 层级     | 测试类型         | 覆盖范围       | 工具          |
| -------- | ---------------- | -------------- | ------------- |
| 单元测试 | Unit Test        | 各模块独立功能 | pytest        |
| 集成测试 | Integration Test | 模块间交互     | pytest + mock |
| E2E 测试 | End-to-End Test  | 完整分析流程   | pytest        |

### 8.2 测试用例设计

```python
# tests/test_agents/test_code_analyzer.py
import pytest
from unittest.mock import AsyncMock, patch

class TestCodeAnalyzerAgent:

    @pytest.fixture
    def mock_mcp_client(self):
        client = AsyncMock()
        client.call_tool.return_value = {
            "success": True,
            "code": "void process(char* buf, int len) {...}"
        }
        return client

    @pytest.mark.asyncio
    async def test_analyze_function_success(self, mock_mcp_client):
        """测试函数分析成功场景"""
        ...

    @pytest.mark.asyncio
    async def test_analyze_function_not_found(self, mock_mcp_client):
        """测试函数不存在场景"""
        ...
```

### 8.3 Mock 策略

- **LLM Mock**：使用预定义响应模拟 LLM 输出
- **MCP Mock**：模拟 Joern MCP Server 响应
- **File Mock**：使用临时文件系统

---

## 9. 部署与运行

### 9.1 环境要求

| 组件             | 要求            |
| ---------------- | --------------- |
| Python           | ≥3.10           |
| Joern            | ≥2.0.0          |
| Joern MCP Server | 运行中          |
| 本地 LLM 服务    | OpenAI 兼容 API |

### 9.2 安装步骤

```bash
# 1. 克隆项目
git clone <repository>
cd fuzz_data_modeler

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -e ".[dev]"

# 4. 配置
cp config/defaults/config.yaml config/config.yaml
# 编辑配置文件，配置LLM API地址等...

# 5. 启动Joern MCP Server（另一终端）
cd ../joern_mcp
python -m joern_mcp

# 6. 运行
python -m fuzz_generator analyze ...
```

### 9.3 配置示例

```yaml
# config/config.yaml
llm:
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
  model: "qwen2.5:32b"
  temperature: 0.7
  max_tokens: 4096

mcp_server:
  url: "http://localhost:8000/mcp"
  timeout: 120
```

---

## 10. 扩展性设计

### 10.1 扩展点

1. **新增 Agent**：通过继承 BaseAgent 实现新的专门化 Agent
2. **新增工具**：在 tools 模块添加新的 MCP 工具封装
3. **自定义 Prompt**：通过配置文件覆盖默认 Prompt
4. **输出格式**：通过实现新的 Generator 支持其他输出格式

### 10.2 插件机制（预留）

```python
# 未来可扩展的插件接口
class PluginInterface:
    """插件接口"""

    def on_analysis_start(self, context: dict): pass
    def on_analysis_complete(self, result: dict): pass
    def transform_output(self, xml: str) -> str: pass
```

---

## 11. 待确定事项

| 编号  | 事项                        | 状态      | 备注                              |
| ----- | --------------------------- | --------- | --------------------------------- |
| 1     | 具体的 DataModel 生成规则   | 待定      | 需要用户通过 prompt 提供背景知识  |
| ~~2~~ | ~~StateModel 是否需要生成~~ | ✅ 已确认 | 仅生成 DataModel 部分             |
| ~~3~~ | ~~是否需要支持批量分析~~    | ✅ 已确认 | 支持 YAML/JSON 格式的批量任务文件 |
| ~~4~~ | ~~分析结果是否需要持久化~~  | ✅ 已确认 | 中间结果持久化存储                |

---

## 12. 项目计划（建议）

| 阶段    | 内容                                  | 预计时间 |
| ------- | ------------------------------------- | -------- |
| Phase 1 | 基础框架搭建（CLI、配置、日志、存储） | 3-4 天   |
| Phase 2 | MCP 客户端与工具封装                  | 2-3 天   |
| Phase 3 | Agent 实现与协作流程                  | 5-6 天   |
| Phase 4 | 批量任务处理与断点续传                | 2-3 天   |
| Phase 5 | XML 生成器与模板                      | 2-3 天   |
| Phase 6 | 集成测试与优化                        | 3-4 天   |
| Phase 7 | 文档与示例                            | 1-2 天   |

**总计**：约 18-25 天

---

## 附录

### A. Joern MCP 工具清单

| 工具名                 | 功能             | 使用场景   |
| ---------------------- | ---------------- | ---------- |
| parse_project          | 解析项目生成 CPG | 初始化阶段 |
| list_projects          | 列出已解析项目   | 项目管理   |
| switch_project         | 切换活动项目     | 多项目场景 |
| get_function_code      | 获取函数源码     | 代码分析   |
| list_functions         | 列出函数         | 函数发现   |
| search_code            | 代码搜索         | 模式匹配   |
| get_callers            | 获取调用者       | 调用图分析 |
| get_callees            | 获取被调用函数   | 调用图分析 |
| track_dataflow         | 数据流追踪       | 数据流分析 |
| analyze_variable_flow  | 变量流分析       | 数据流分析 |
| get_control_flow_graph | 获取 CFG         | 控制流分析 |
| find_vulnerabilities   | 漏洞检测         | 安全分析   |

### B. 参考资料

1. [AutoGen 官方文档](https://microsoft.github.io/autogen/stable/)
2. [AutoGen API 参考](https://microsoft.github.io/autogen/stable/reference/index.html)
3. [AutoGen GitHub 仓库](https://github.com/microsoft/autogen)
4. [Joern 文档](https://docs.joern.io/)
5. [MCP 协议规范](https://modelcontextprotocol.io/)
6. [Pydantic v2 文档](https://docs.pydantic.dev/)

---

**文档维护**：

- 创建日期：2024-12-19
- 最后更新：2024-12-19
- 维护者：[待填写]
