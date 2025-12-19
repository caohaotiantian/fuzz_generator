# Fuzz Generator

基于 AI Agent 的 Fuzz 测试数据建模工具。

## 简介

Fuzz Generator 是一个自动化工具，用于分析源代码函数并生成 fuzz 测试所需的数据建模（Secray XML 格式）。系统利用大语言模型（LLM）理解代码语义，结合 Joern 静态分析工具提取代码结构信息，最终生成符合规范的 DataModel 定义。

## 特性

- 🤖 **AI 驱动**：基于 AutoGen 框架的多 Agent 协作系统
- 🔍 **深度分析**：通过 Joern 进行数据流、控制流分析
- 📦 **批量处理**：支持一次分析多个函数
- 💾 **断点续传**：任务中断后可恢复
- ⚙️ **高度可配置**：支持自定义 Prompt 和参数配置

## 安装

### 环境要求

- Python >= 3.10
- Joern >= 2.0.0
- Joern MCP Server（运行中）
- 本地 LLM 服务（OpenAI 兼容 API）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-org/fuzz_generator.git
cd fuzz_generator

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .

# 安装开发依赖（可选）
pip install -e ".[dev]"
```

## 快速开始

### 1. 配置

创建配置文件 `config.yaml`：

```yaml
version: "1.0"

llm:
  base_url: "http://localhost:11434/v1"
  model: "qwen2.5:32b"
  temperature: 0.7

mcp_server:
  url: "http://localhost:8000/mcp"
```

### 2. 解析项目

```bash
# 解析源代码项目
fuzz-generator parse -p ./your_project
```

### 3. 分析函数

```bash
# 单函数分析
fuzz-generator analyze -p ./your_project -f main.c -fn process_request -o output.xml

# 批量分析
fuzz-generator analyze -p ./your_project -t tasks.yaml -o ./output/
```

### 4. 查看结果

```bash
# 列出所有结果
fuzz-generator results --list

# 查看特定任务结果
fuzz-generator results -t task_001
```

## 命令参考

| 命令 | 说明 |
|------|------|
| `analyze` | 分析函数并生成 DataModel |
| `parse` | 解析项目生成 CPG |
| `results` | 查看分析结果 |
| `clean` | 清理缓存和中间结果 |
| `tools` | 列出可用的 MCP 工具 |
| `status` | 显示当前状态 |

## 批量任务格式

创建 `tasks.yaml` 文件：

```yaml
project_path: "/path/to/source"
description: "RTSP 协议处理函数分析"

tasks:
  - source_file: "rtsp/handler.c"
    function_name: "process_request"
    output_name: "RtspRequest"
    
  - source_file: "rtsp/parser.c"
    function_name: "parse_header"
    output_name: "HeaderLine"
```

## 配置选项

详见 [配置文档](docs/config.example.yaml)。

## 开发

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行 Phase 1 测试
pytest tests/test_phase1/ -v

# 生成覆盖率报告
pytest tests/ --cov=fuzz_generator --cov-report=html
```

### 代码检查

```bash
# 代码格式化
ruff format .

# 代码检查
ruff check .

# 类型检查
mypy fuzz_generator/
```

## 项目结构

```
fuzz_generator/
├── cli/                # 命令行接口
├── config/             # 配置管理
├── agents/             # AI Agent 实现
├── tools/              # MCP 工具封装
├── models/             # 数据模型
├── storage/            # 持久化存储
├── generators/         # XML 生成器
├── batch/              # 批量任务处理
└── utils/              # 工具函数
```

## 文档

- [技术设计文档](docs/TECHNICAL_DESIGN.md)
- [开发计划](docs/DEVELOPMENT_PLAN.md)
- [配置示例](docs/config.example.yaml)

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

