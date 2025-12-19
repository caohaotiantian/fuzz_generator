# 基础分析示例

本示例演示如何使用 `fuzz_generator` 对单个 C 函数进行分析并生成 DataModel。

## 目录结构

```
basic_analysis/
├── README.md           # 本文件
├── sample_code/        # 示例 C 代码
│   └── handler.c       # 包含待分析函数的源文件
├── tasks.yaml          # 分析任务配置
├── knowledge.md        # 自定义领域知识
└── output/             # 生成的输出（运行后创建）
```

## 快速开始

### 1. 确保依赖已安装

```bash
# 安装 fuzz_generator
pip install -e /path/to/fuzz_generator

# 确保 Joern MCP 服务器正在运行
# 参见 joern_mcp 文档
```

### 2. 运行分析

```bash
# 进入示例目录
cd examples/basic_analysis

# 执行分析
fuzz-generator analyze --task-file tasks.yaml --output output/

# 或使用 Python 模块方式
python -m fuzz_generator analyze --task-file tasks.yaml --output output/
```

### 3. 查看结果

分析完成后，输出文件将生成在 `output/` 目录：

```bash
ls output/
# ProcessRequestModel.xml
```

## 任务配置说明

`tasks.yaml` 文件定义了要分析的函数：

```yaml
project_path: "./sample_code"
description: "基础分析示例"
tasks:
  - source_file: "handler.c"
    function_name: "process_request"
    output_name: "ProcessRequestModel"
```

## 自定义知识配置

可以通过 `knowledge.md` 文件提供领域特定的知识，帮助 LLM 更好地理解代码语义：

```markdown
# 协议解析知识

- buffer 参数通常是网络接收的原始数据
- length 参数表示有效数据长度
- 返回值 0 表示成功，-1 表示失败
```

使用自定义知识运行：

```bash
fuzz-generator analyze \
  --task-file tasks.yaml \
  --knowledge knowledge.md \
  --output output/
```

## 输出说明

生成的 XML DataModel 格式如下：

```xml
<?xml version="1.0" encoding="utf-8"?>
<Secray>
    <DataModel name="ProcessRequestModel">
        <Blob name="buffer" />
        <Number name="length" size="32" signed="true" />
    </DataModel>
</Secray>
```

## 故障排除

### MCP 服务器连接失败

确保 Joern MCP 服务器正在运行：

```bash
# 检查 MCP 服务器状态
curl http://localhost:8000/mcp -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
```

### 函数未找到

确保函数名称与代码中的定义完全匹配（区分大小写）。

### 生成结果不符合预期

尝试提供更多自定义知识，或调整 LLM 参数（如 temperature）。

