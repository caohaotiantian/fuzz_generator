# 批量分析示例

本示例演示如何使用 `fuzz_generator` 对多个函数进行批量分析。

## 目录结构

```
batch_analysis/
├── README.md           # 本文件
├── rtsp_server/        # 模拟 RTSP 服务器代码
│   ├── handler.c       # 请求处理
│   ├── parser.c        # 协议解析
│   └── session.c       # 会话管理
├── tasks.yaml          # 批量任务配置
└── output/             # 生成的输出
```

## 运行批量分析

### 1. 执行分析

```bash
cd examples/batch_analysis

# 运行批量分析
fuzz-generator analyze --task-file tasks.yaml --output output/
```

### 2. 监控进度

```bash
# 查看当前状态
fuzz-generator status
```

### 3. 断点续传

如果分析中断，可以从上次位置继续：

```bash
# 恢复执行
fuzz-generator analyze --task-file tasks.yaml --output output/ --resume
```

## 任务配置

### 基本配置

```yaml
project_path: "./rtsp_server"
tasks:
  - source_file: "handler.c"
    function_name: "handle_request"
  - source_file: "parser.c"
    function_name: "parse_rtsp_request"
```

### 带依赖的配置

```yaml
tasks:
  - source_file: "parser.c"
    function_name: "parse_rtsp_request"
    output_name: "RTSPRequest"
  
  - source_file: "handler.c"
    function_name: "handle_request"
    output_name: "RequestHandler"
    depends_on:
      - "0"  # 依赖第一个任务
```

### 并发配置

在配置文件中设置并发数：

```yaml
# config.yaml
batch:
  max_concurrent: 4  # 最多同时执行4个任务
  fail_fast: false   # 单个失败不影响其他任务
```

## 输出结果

批量分析完成后，输出目录结构如下：

```
output/
├── RTSPRequest.xml
├── RequestHandler.xml
└── SessionManager.xml
```

## 查看历史结果

```bash
# 列出所有结果
fuzz-generator results

# 查看特定任务结果
fuzz-generator results --task-id <task_id>
```

## 清理

```bash
# 清理中间文件
fuzz-generator clean

# 清理所有缓存（包括分析结果）
fuzz-generator clean --all
```

