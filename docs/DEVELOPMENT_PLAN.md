# Fuzz Generator - 阶段性开发计划

**版本**: v1.0  
**日期**: 2024-12-19  
**关联文档**: [技术方案设计](./TECHNICAL_DESIGN.md)

---

## 目录

- [开发阶段概览](#开发阶段概览)
- [Phase 1: 基础框架](#phase-1-基础框架)
- [Phase 2: MCP 客户端](#phase-2-mcp客户端)
- [Phase 3: Agent 实现](#phase-3-agent实现)
- [Phase 4: 批量任务处理](#phase-4-批量任务处理)
- [Phase 5: XML 生成器](#phase-5-xml生成器)
- [Phase 6: 集成与优化](#phase-6-集成与优化)
- [Phase 7: 文档与示例](#phase-7-文档与示例)
- [测试策略总览](#测试策略总览)

---

## 开发阶段概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           开发阶段依赖关系                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Phase 1 ──────► Phase 2 ──────► Phase 3 ──────┐                          │
│   基础框架         MCP客户端        Agent实现     │                          │
│                                                  ▼                          │
│                                              Phase 6 ──────► Phase 7       │
│                                              集成优化         文档示例       │
│                                                  ▲                          │
│   Phase 4 ◄─────────────────────────────────────┘                          │
│   批量任务         Phase 5                                                  │
│       ▲            XML生成器                                                │
│       └────────────────┘                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| 阶段    | 名称         | 预计时间 | 前置依赖         |
| ------- | ------------ | -------- | ---------------- |
| Phase 1 | 基础框架     | 3-4 天   | 无               |
| Phase 2 | MCP 客户端   | 2-3 天   | Phase 1          |
| Phase 3 | Agent 实现   | 5-6 天   | Phase 1, Phase 2 |
| Phase 4 | 批量任务处理 | 2-3 天   | Phase 1          |
| Phase 5 | XML 生成器   | 2-3 天   | Phase 1          |
| Phase 6 | 集成与优化   | 3-4 天   | Phase 1-5        |
| Phase 7 | 文档与示例   | 1-2 天   | Phase 6          |

---

## Phase 1: 基础框架

### 1.1 阶段目标

搭建项目基础设施，包括 CLI 框架、配置管理、日志系统和持久化存储模块。

### 1.2 交付物清单

| 序号 | 交付物     | 文件路径                             | 说明              |
| ---- | ---------- | ------------------------------------ | ----------------- |
| 1.1  | 项目初始化 | `pyproject.toml`, `requirements.txt` | 项目配置和依赖    |
| 1.2  | CLI 框架   | `fuzz_generator/cli/`                | 命令行接口        |
| 1.3  | 配置模块   | `fuzz_generator/config/`             | 配置加载和验证    |
| 1.4  | 日志模块   | `fuzz_generator/utils/logger.py`     | 日志配置          |
| 1.5  | 存储模块   | `fuzz_generator/storage/`            | 持久化存储        |
| 1.6  | 数据模型   | `fuzz_generator/models/`             | Pydantic 模型定义 |
| 1.7  | 单元测试   | `tests/test_phase1/`                 | Phase 1 测试用例  |

### 1.3 详细任务分解

#### 1.3.1 项目初始化 (0.5 天)

**任务描述**: 创建项目结构和依赖配置

**交付文件**:

```
fuzz_generator/
├── __init__.py
├── __main__.py
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

**验收标准**:

- [ ] `pip install -e .` 安装成功
- [ ] `python -m fuzz_generator --help` 显示帮助信息
- [ ] 所有依赖版本锁定且可正常安装

**自动化测试**:

```python
# tests/test_phase1/test_installation.py
import subprocess
import sys

def test_package_installable():
    """测试包可以正常安装"""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        capture_output=True
    )
    assert result.returncode == 0

def test_cli_help():
    """测试CLI帮助命令"""
    result = subprocess.run(
        [sys.executable, "-m", "fuzz_generator", "--help"],
        capture_output=True
    )
    assert result.returncode == 0
    assert b"Fuzz Generator" in result.stdout
```

#### 1.3.2 配置模块 (1 天)

**任务描述**: 实现 YAML 配置文件加载、验证和默认值处理

**交付文件**:

```
fuzz_generator/config/
├── __init__.py
├── settings.py      # Pydantic Settings模型
├── loader.py        # 配置加载器
└── defaults/
    └── config.yaml  # 默认配置
```

**验收标准**:

- [ ] 支持从 YAML 文件加载配置
- [ ] 支持环境变量覆盖配置
- [ ] 配置验证失败时抛出明确错误
- [ ] 未指定配置时使用默认值

**自动化测试**:

```python
# tests/test_phase1/test_config.py
import pytest
from fuzz_generator.config import Settings, load_config

class TestConfigLoading:
    def test_load_default_config(self):
        """测试加载默认配置"""
        config = load_config()
        assert config.llm.base_url is not None
        assert config.mcp_server.url is not None

    def test_load_custom_config(self, tmp_path):
        """测试加载自定义配置"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
        llm:
          base_url: "http://custom:8000/v1"
          model: "custom-model"
        """)
        config = load_config(str(config_file))
        assert config.llm.base_url == "http://custom:8000/v1"
        assert config.llm.model == "custom-model"

    def test_config_validation_error(self, tmp_path):
        """测试配置验证失败"""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("""
        llm:
          temperature: 2.0  # 超出范围
        """)
        with pytest.raises(ValueError):
            load_config(str(config_file))

    def test_env_override(self, monkeypatch):
        """测试环境变量覆盖"""
        monkeypatch.setenv("FUZZ_GENERATOR_LLM__BASE_URL", "http://env:9000")
        config = load_config()
        assert config.llm.base_url == "http://env:9000"

class TestSettingsModel:
    def test_llm_settings_defaults(self):
        """测试LLM配置默认值"""
        from fuzz_generator.config.settings import LLMSettings
        settings = LLMSettings()
        assert settings.temperature == 0.7
        assert settings.max_tokens == 4096

    def test_llm_settings_validation(self):
        """测试LLM配置验证"""
        from fuzz_generator.config.settings import LLMSettings
        with pytest.raises(ValueError):
            LLMSettings(temperature=1.5)  # 超出0-1范围
```

#### 1.3.3 日志模块 (0.5 天)

**任务描述**: 基于 loguru 实现统一的日志配置

**交付文件**:

```
fuzz_generator/utils/
├── __init__.py
└── logger.py
```

**验收标准**:

- [ ] 支持控制台和文件双输出
- [ ] 支持日志级别配置
- [ ] 支持日志轮转和压缩
- [ ] 日志格式包含时间、级别、模块、消息

**自动化测试**:

```python
# tests/test_phase1/test_logger.py
import pytest
from pathlib import Path
from fuzz_generator.utils.logger import setup_logger, get_logger

class TestLogger:
    def test_setup_logger(self, tmp_path):
        """测试日志配置"""
        log_file = tmp_path / "test.log"
        logger = setup_logger(log_level="DEBUG", log_file=str(log_file))
        logger.info("Test message")
        assert log_file.exists()
        assert "Test message" in log_file.read_text()

    def test_log_levels(self, tmp_path):
        """测试日志级别过滤"""
        log_file = tmp_path / "test.log"
        logger = setup_logger(log_level="WARNING", log_file=str(log_file))
        logger.debug("Debug message")
        logger.warning("Warning message")
        content = log_file.read_text()
        assert "Debug message" not in content
        assert "Warning message" in content

    def test_get_logger(self):
        """测试获取模块logger"""
        logger = get_logger("test_module")
        assert logger is not None
```

#### 1.3.4 存储模块 (1 天)

**任务描述**: 实现中间结果的持久化存储

**交付文件**:

```
fuzz_generator/storage/
├── __init__.py
├── base.py           # 存储接口定义
├── json_storage.py   # JSON文件存储实现
└── cache.py          # 缓存管理
```

**验收标准**:

- [ ] 支持保存和加载 JSON 数据
- [ ] 支持按 category/key 组织数据
- [ ] 支持列出和删除数据
- [ ] 支持缓存过期机制

**自动化测试**:

```python
# tests/test_phase1/test_storage.py
import pytest
import asyncio
from fuzz_generator.storage import JsonStorage, CacheManager

class TestJsonStorage:
    @pytest.fixture
    def storage(self, tmp_path):
        return JsonStorage(base_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_save_and_load(self, storage):
        """测试保存和加载数据"""
        data = {"key": "value", "number": 42}
        await storage.save("test_category", "test_key", data)
        loaded = await storage.load("test_category", "test_key")
        assert loaded == data

    @pytest.mark.asyncio
    async def test_exists(self, storage):
        """测试检查数据存在"""
        assert not await storage.exists("category", "nonexistent")
        await storage.save("category", "key", {"data": 1})
        assert await storage.exists("category", "key")

    @pytest.mark.asyncio
    async def test_delete(self, storage):
        """测试删除数据"""
        await storage.save("category", "key", {"data": 1})
        await storage.delete("category", "key")
        assert not await storage.exists("category", "key")

    @pytest.mark.asyncio
    async def test_list_keys(self, storage):
        """测试列出所有键"""
        await storage.save("category", "key1", {"data": 1})
        await storage.save("category", "key2", {"data": 2})
        keys = await storage.list_keys("category")
        assert set(keys) == {"key1", "key2"}

class TestCacheManager:
    @pytest.fixture
    def cache(self, tmp_path):
        return CacheManager(base_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_cache_hit(self, cache):
        """测试缓存命中"""
        await cache.set("func_hash_123", {"result": "data"})
        result = await cache.get("func_hash_123")
        assert result == {"result": "data"}

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """测试缓存未命中"""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_invalidate(self, cache):
        """测试缓存失效"""
        await cache.set("key", {"data": 1})
        await cache.invalidate("key")
        assert await cache.get("key") is None
```

#### 1.3.5 数据模型 (0.5 天)

**任务描述**: 定义核心 Pydantic 数据模型

**交付文件**:

```
fuzz_generator/models/
├── __init__.py
├── task.py           # 任务模型
├── function_info.py  # 函数信息模型
└── analysis_result.py # 分析结果模型
```

**验收标准**:

- [ ] 所有模型支持 JSON 序列化/反序列化
- [ ] 模型字段有完整的类型注解
- [ ] 模型有合理的默认值和验证规则

**自动化测试**:

```python
# tests/test_phase1/test_models.py
import pytest
from datetime import datetime
from fuzz_generator.models import (
    AnalysisTask, BatchTask, TaskStatus,
    FunctionInfo, ParameterInfo, AnalysisResult
)

class TestTaskModels:
    def test_analysis_task_creation(self):
        """测试创建分析任务"""
        task = AnalysisTask(
            task_id="task_001",
            source_file="main.c",
            function_name="process",
            created_at=datetime.now()
        )
        assert task.status == TaskStatus.PENDING
        assert task.output_name is None

    def test_analysis_task_serialization(self):
        """测试任务序列化"""
        task = AnalysisTask(
            task_id="task_001",
            source_file="main.c",
            function_name="process",
            created_at=datetime.now()
        )
        json_str = task.model_dump_json()
        loaded = AnalysisTask.model_validate_json(json_str)
        assert loaded.task_id == task.task_id

    def test_batch_task_creation(self):
        """测试批量任务创建"""
        tasks = [
            AnalysisTask(
                task_id=f"task_{i}",
                source_file=f"file{i}.c",
                function_name=f"func{i}",
                created_at=datetime.now()
            )
            for i in range(3)
        ]
        batch = BatchTask(
            batch_id="batch_001",
            project_path="/path/to/project",
            tasks=tasks,
            created_at=datetime.now()
        )
        assert len(batch.tasks) == 3
        assert batch.completed_count == 0

class TestFunctionModels:
    def test_parameter_info(self):
        """测试参数信息模型"""
        param = ParameterInfo(
            name="buffer",
            type="char*",
            direction="in",
            description="Input buffer"
        )
        assert param.constraints == []

    def test_function_info(self):
        """测试函数信息模型"""
        func = FunctionInfo(
            name="process",
            file_path="main.c",
            line_number=10,
            return_type="int",
            parameters=[],
            source_code="int process() { return 0; }"
        )
        assert func.description == ""
```

#### 1.3.6 CLI 框架 (0.5 天)

**任务描述**: 实现命令行接口框架

**交付文件**:

```
fuzz_generator/cli/
├── __init__.py
├── commands.py      # 命令定义
└── validators.py    # 输入验证
```

**验收标准**:

- [ ] 支持 `--help` 显示帮助
- [ ] 支持 `--version` 显示版本
- [ ] 支持 `--config` 指定配置文件
- [ ] 支持 `--verbose` 详细输出
- [ ] 命令参数验证失败时给出明确提示

**自动化测试**:

```python
# tests/test_phase1/test_cli.py
import pytest
from click.testing import CliRunner
from fuzz_generator.cli import cli

class TestCLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_help(self, runner):
        """测试帮助命令"""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Fuzz Generator" in result.output

    def test_version(self, runner):
        """测试版本命令"""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

    def test_analyze_help(self, runner):
        """测试analyze子命令帮助"""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--project-path" in result.output
        assert "--task-file" in result.output

    def test_analyze_missing_required(self, runner):
        """测试缺少必需参数"""
        result = runner.invoke(cli, ["analyze"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_analyze_invalid_path(self, runner):
        """测试无效路径"""
        result = runner.invoke(cli, [
            "analyze",
            "--project-path", "/nonexistent/path",
            "--source-file", "main.c",
            "--function", "test"
        ])
        assert result.exit_code != 0

    def test_parse_command(self, runner, tmp_path):
        """测试parse命令"""
        # 创建一个临时目录作为项目路径
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "main.c").write_text("int main() { return 0; }")

        result = runner.invoke(cli, [
            "parse",
            "--project-path", str(project_dir)
        ])
        # 这里只验证命令可以被解析，实际执行需要MCP服务器
        # 具体行为在集成测试中验证
```

### 1.4 Phase 1 验收清单

| 序号 | 验收项       | 验收方式   | 通过标准                |
| ---- | ------------ | ---------- | ----------------------- |
| 1    | 项目可安装   | 自动化测试 | `pip install -e .` 成功 |
| 2    | CLI 基本功能 | 自动化测试 | 所有 CLI 测试通过       |
| 3    | 配置加载     | 自动化测试 | 所有配置测试通过        |
| 4    | 日志系统     | 自动化测试 | 所有日志测试通过        |
| 5    | 存储模块     | 自动化测试 | 所有存储测试通过        |
| 6    | 数据模型     | 自动化测试 | 所有模型测试通过        |
| 7    | 测试覆盖率   | pytest-cov | 覆盖率 ≥ 80%            |

**验收命令**:

```bash
# 运行Phase 1所有测试
pytest tests/test_phase1/ -v --cov=fuzz_generator --cov-report=term-missing

# 验收标准：所有测试通过，覆盖率>=80%
```

---

## Phase 2: MCP 客户端

### 2.1 阶段目标

实现与 Joern MCP Server 的 HTTP 通信客户端，封装所有 MCP 工具调用。

### 2.2 交付物清单

| 序号 | 交付物     | 文件路径                                 | 说明             |
| ---- | ---------- | ---------------------------------------- | ---------------- |
| 2.1  | MCP 客户端 | `fuzz_generator/tools/mcp_client.py`     | HTTP 客户端实现  |
| 2.2  | 项目工具   | `fuzz_generator/tools/project_tools.py`  | 项目管理工具封装 |
| 2.3  | 分析工具   | `fuzz_generator/tools/analysis_tools.py` | 分析工具封装     |
| 2.4  | 查询工具   | `fuzz_generator/tools/query_tools.py`    | 查询工具封装     |
| 2.5  | 单元测试   | `tests/test_phase2/`                     | Phase 2 测试用例 |

### 2.3 详细任务分解

#### 2.3.1 MCP HTTP 客户端 (1 天)

**任务描述**: 实现基于 httpx 的异步 MCP 客户端

**交付文件**:

```
fuzz_generator/tools/
├── __init__.py
└── mcp_client.py
```

**验收标准**:

- [ ] 支持异步 HTTP 请求
- [ ] 支持请求超时和重试
- [ ] 支持连接池管理
- [ ] 错误响应有明确的异常类型

**自动化测试**:

```python
# tests/test_phase2/test_mcp_client.py
import pytest
import httpx
from unittest.mock import AsyncMock, patch
from fuzz_generator.tools.mcp_client import MCPHttpClient, MCPClientConfig

class TestMCPHttpClient:
    @pytest.fixture
    def config(self):
        return MCPClientConfig(
            url="http://localhost:8000/mcp",
            timeout=30,
            retry_count=3
        )

    @pytest.mark.asyncio
    async def test_client_context_manager(self, config):
        """测试客户端上下文管理器"""
        async with MCPHttpClient(config) as client:
            assert client._client is not None
        assert client._client is None

    @pytest.mark.asyncio
    async def test_call_tool_success(self, config):
        """测试成功调用工具"""
        mock_response = {
            "content": [{"type": "text", "text": '{"success": true}'}]
        }
        with patch.object(httpx.AsyncClient, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.status_code = 200

            async with MCPHttpClient(config) as client:
                result = await client.call_tool("test_tool", {"arg": "value"})
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_call_tool_retry_on_error(self, config):
        """测试错误时重试"""
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection failed")
            response = AsyncMock()
            response.json.return_value = {"content": [{"type": "text", "text": '{"success": true}'}]}
            response.status_code = 200
            return response

        with patch.object(httpx.AsyncClient, 'post', side_effect=mock_post):
            async with MCPHttpClient(config) as client:
                result = await client.call_tool("test_tool", {})
                assert result["success"] is True
                assert call_count == 3

    @pytest.mark.asyncio
    async def test_call_tool_max_retries_exceeded(self, config):
        """测试超过最大重试次数"""
        with patch.object(httpx.AsyncClient, 'post', side_effect=httpx.ConnectError("Failed")):
            async with MCPHttpClient(config) as client:
                with pytest.raises(Exception):  # MCPConnectionError
                    await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_list_tools(self, config):
        """测试列出可用工具"""
        mock_response = {
            "tools": [
                {"name": "tool1", "description": "desc1"},
                {"name": "tool2", "description": "desc2"}
            ]
        }
        with patch.object(httpx.AsyncClient, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.status_code = 200

            async with MCPHttpClient(config) as client:
                tools = await client.list_tools()
                assert len(tools) == 2
```

#### 2.3.2 工具封装 (1-2 天)

**任务描述**: 封装 Joern MCP 工具为类型安全的 Python 函数

**交付文件**:

```
fuzz_generator/tools/
├── project_tools.py   # parse_project, list_projects, etc.
├── analysis_tools.py  # track_dataflow, get_callers, etc.
└── query_tools.py     # get_function_code, list_functions, etc.
```

**验收标准**:

- [ ] 所有工具有类型注解
- [ ] 工具参数有验证
- [ ] 工具返回值有统一的结构
- [ ] 工具调用失败时抛出明确异常

**自动化测试**:

```python
# tests/test_phase2/test_tools.py
import pytest
from unittest.mock import AsyncMock, patch
from fuzz_generator.tools import (
    parse_project, list_projects, get_function_code,
    track_dataflow, get_callers, get_callees
)

class TestProjectTools:
    @pytest.mark.asyncio
    async def test_parse_project(self):
        """测试解析项目"""
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = {
            "success": True,
            "project_name": "test_project",
            "message": "Project parsed successfully"
        }

        result = await parse_project(
            mock_client,
            source_path="/path/to/source",
            project_name="test_project"
        )

        assert result.success is True
        assert result.project_name == "test_project"
        mock_client.call_tool.assert_called_once_with(
            "parse_project",
            {
                "source_path": "/path/to/source",
                "project_name": "test_project",
                "language": "auto"
            }
        )

    @pytest.mark.asyncio
    async def test_list_projects(self):
        """测试列出项目"""
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = {
            "success": True,
            "projects": [
                {"name": "proj1", "inputPath": "/path1"},
                {"name": "proj2", "inputPath": "/path2"}
            ],
            "count": 2
        }

        result = await list_projects(mock_client)
        assert result.success is True
        assert len(result.projects) == 2

class TestQueryTools:
    @pytest.mark.asyncio
    async def test_get_function_code(self):
        """测试获取函数代码"""
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = {
            "success": True,
            "function_name": "main",
            "code": "int main() { return 0; }",
            "file": "main.c",
            "line_number": 1
        }

        result = await get_function_code(
            mock_client,
            function_name="main",
            project_name="test_project"
        )

        assert result.success is True
        assert "int main()" in result.code

    @pytest.mark.asyncio
    async def test_get_function_code_not_found(self):
        """测试函数不存在"""
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = {
            "success": False,
            "error": "Function not found"
        }

        result = await get_function_code(
            mock_client,
            function_name="nonexistent",
            project_name="test_project"
        )

        assert result.success is False

class TestAnalysisTools:
    @pytest.mark.asyncio
    async def test_track_dataflow(self):
        """测试数据流追踪"""
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = {
            "success": True,
            "flows": [
                {
                    "source": {"code": "gets(buf)", "file": "main.c", "line": 10},
                    "sink": {"code": "system(buf)", "file": "main.c", "line": 20},
                    "pathLength": 3
                }
            ],
            "count": 1
        }

        result = await track_dataflow(
            mock_client,
            project_name="test_project",
            source_method="gets",
            sink_method="system"
        )

        assert result.success is True
        assert len(result.flows) == 1

    @pytest.mark.asyncio
    async def test_get_callers(self):
        """测试获取调用者"""
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = {
            "success": True,
            "function": "process",
            "callers": [
                {"name": "main", "filename": "main.c", "lineNumber": 10}
            ],
            "count": 1
        }

        result = await get_callers(
            mock_client,
            project_name="test_project",
            function_name="process"
        )

        assert result.success is True
        assert len(result.callers) == 1
```

### 2.4 Phase 2 验收清单

| 序号 | 验收项         | 验收方式        | 通过标准         |
| ---- | -------------- | --------------- | ---------------- |
| 1    | MCP 客户端连接 | 单元测试 + Mock | 所有连接测试通过 |
| 2    | 重试机制       | 单元测试        | 重试逻辑测试通过 |
| 3    | 工具封装       | 单元测试 + Mock | 所有工具测试通过 |
| 4    | 错误处理       | 单元测试        | 异常测试通过     |
| 5    | 测试覆盖率     | pytest-cov      | 覆盖率 ≥ 85%     |

**验收命令**:

```bash
pytest tests/test_phase2/ -v --cov=fuzz_generator/tools --cov-report=term-missing
```

**集成测试（需要 MCP 服务器）**:

```bash
# 启动MCP服务器后运行
pytest tests/test_phase2/test_integration.py -v --mcp-server=http://localhost:8000/mcp
```

---

## Phase 3: Agent 实现

### 3.1 阶段目标

基于 AutoGen AgentChat 实现多 Agent 协作系统。

### 3.2 交付物清单

| 序号 | 交付物         | 文件路径                                   | 说明             |
| ---- | -------------- | ------------------------------------------ | ---------------- |
| 3.1  | Agent 基类     | `fuzz_generator/agents/base.py`            | Agent 抽象基类   |
| 3.2  | Orchestrator   | `fuzz_generator/agents/orchestrator.py`    | 编排 Agent       |
| 3.3  | CodeAnalyzer   | `fuzz_generator/agents/code_analyzer.py`   | 代码分析 Agent   |
| 3.4  | ContextBuilder | `fuzz_generator/agents/context_builder.py` | 上下文构建 Agent |
| 3.5  | ModelGenerator | `fuzz_generator/agents/model_generator.py` | 模型生成 Agent   |
| 3.6  | Prompt 模板    | `fuzz_generator/config/defaults/prompts/`  | Agent 提示词     |
| 3.7  | 单元测试       | `tests/test_phase3/`                       | Phase 3 测试用例 |

### 3.3 详细任务分解

#### 3.3.1 Agent 基类设计 (0.5 天)

**交付文件**:

```
fuzz_generator/agents/
├── __init__.py
└── base.py
```

**验收标准**:

- [ ] 定义 Agent 接口规范
- [ ] 支持工具注册
- [ ] 支持 Prompt 模板加载

**自动化测试**:

```python
# tests/test_phase3/test_base_agent.py
import pytest
from fuzz_generator.agents.base import BaseAgent

class TestBaseAgent:
    def test_agent_creation(self):
        """测试Agent创建"""
        # 测试基类接口
        pass

    def test_tool_registration(self):
        """测试工具注册"""
        pass

    def test_prompt_loading(self):
        """测试Prompt加载"""
        pass
```

#### 3.3.2 CodeAnalyzer Agent (1.5 天)

**任务描述**: 实现代码分析 Agent，负责解析函数结构

**验收标准**:

- [ ] 能够获取并解析函数代码
- [ ] 能够识别函数参数
- [ ] 能够输出结构化的分析结果
- [ ] 支持多轮对话获取更多信息

**自动化测试**:

```python
# tests/test_phase3/test_code_analyzer.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from fuzz_generator.agents.code_analyzer import CodeAnalyzerAgent

class TestCodeAnalyzerAgent:
    @pytest.fixture
    def mock_llm_client(self):
        """模拟LLM客户端"""
        client = MagicMock()
        return client

    @pytest.fixture
    def mock_mcp_client(self):
        """模拟MCP客户端"""
        client = AsyncMock()
        client.call_tool.return_value = {
            "success": True,
            "code": """
            int process_request(char* buffer, int length) {
                // process the request
                return 0;
            }
            """,
            "file": "handler.c",
            "line_number": 10
        }
        return client

    @pytest.mark.asyncio
    async def test_analyze_function_basic(self, mock_llm_client, mock_mcp_client):
        """测试基本函数分析"""
        agent = CodeAnalyzerAgent(
            llm_client=mock_llm_client,
            mcp_client=mock_mcp_client
        )

        # 模拟LLM响应
        mock_llm_client.create.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(
                    content='''
                    {
                        "function_name": "process_request",
                        "return_type": "int",
                        "parameters": [
                            {"name": "buffer", "type": "char*", "direction": "in"},
                            {"name": "length", "type": "int", "direction": "in"}
                        ]
                    }
                    '''
                )
            )]
        )

        result = await agent.analyze(
            project_name="test_project",
            function_name="process_request"
        )

        assert result.function_name == "process_request"
        assert len(result.parameters) == 2

    @pytest.mark.asyncio
    async def test_analyze_with_tool_calls(self, mock_llm_client, mock_mcp_client):
        """测试带工具调用的分析"""
        # 测试Agent调用MCP工具的场景
        pass
```

#### 3.3.3 ContextBuilder Agent (1.5 天)

**任务描述**: 实现上下文构建 Agent，负责数据流和控制流分析

**验收标准**:

- [ ] 能够追踪数据流
- [ ] 能够分析控制流
- [ ] 能够获取调用关系
- [ ] 输出完整的上下文信息

**自动化测试**:

```python
# tests/test_phase3/test_context_builder.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from fuzz_generator.agents.context_builder import ContextBuilderAgent

class TestContextBuilderAgent:
    @pytest.fixture
    def mock_mcp_client(self):
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_build_context(self, mock_mcp_client):
        """测试构建上下文"""
        # 配置mock返回值
        mock_mcp_client.call_tool.side_effect = [
            # track_dataflow 结果
            {"success": True, "flows": [], "count": 0},
            # get_callers 结果
            {"success": True, "callers": [], "count": 0},
            # get_callees 结果
            {"success": True, "callees": [], "count": 0}
        ]

        agent = ContextBuilderAgent(mcp_client=mock_mcp_client)

        function_info = MagicMock()
        function_info.name = "process_request"
        function_info.parameters = []

        result = await agent.build_context(
            project_name="test_project",
            function_info=function_info
        )

        assert result is not None
```

#### 3.3.4 ModelGenerator Agent (1.5 天)

**任务描述**: 实现模型生成 Agent，负责生成 DataModel

**验收标准**:

- [ ] 能够根据分析结果生成 DataModel
- [ ] 输出符合 Secray 格式的结构
- [ ] 支持自定义背景知识注入

**自动化测试**:

```python
# tests/test_phase3/test_model_generator.py
import pytest
from unittest.mock import MagicMock
from fuzz_generator.agents.model_generator import ModelGeneratorAgent

class TestModelGeneratorAgent:
    @pytest.mark.asyncio
    async def test_generate_datamodel(self):
        """测试生成DataModel"""
        mock_llm_client = MagicMock()
        mock_llm_client.create.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(
                    content='''
                    <DataModel name="Request">
                        <String name="Method" />
                        <String name="Space" value=" " token="true" />
                    </DataModel>
                    '''
                )
            )]
        )

        agent = ModelGeneratorAgent(llm_client=mock_llm_client)

        # 构造输入上下文
        context = MagicMock()
        context.function_info.name = "process_request"
        context.function_info.parameters = []

        result = await agent.generate(context)

        assert result is not None
        assert "<DataModel" in result.xml_content
```

#### 3.3.5 Orchestrator Agent (1 天)

**任务描述**: 实现编排 Agent，协调其他 Agent 工作

**验收标准**:

- [ ] 能够按顺序调用各 Agent
- [ ] 能够传递中间结果
- [ ] 能够处理错误和重试
- [ ] 能够保存中间状态

**自动化测试**:

```python
# tests/test_phase3/test_orchestrator.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from fuzz_generator.agents.orchestrator import OrchestratorAgent

class TestOrchestratorAgent:
    @pytest.fixture
    def mock_agents(self):
        return {
            "code_analyzer": AsyncMock(),
            "context_builder": AsyncMock(),
            "model_generator": AsyncMock()
        }

    @pytest.mark.asyncio
    async def test_run_analysis_flow(self, mock_agents):
        """测试完整分析流程"""
        # 配置mock返回值
        mock_agents["code_analyzer"].analyze.return_value = MagicMock()
        mock_agents["context_builder"].build_context.return_value = MagicMock()
        mock_agents["model_generator"].generate.return_value = MagicMock(
            xml_content="<DataModel>...</DataModel>"
        )

        orchestrator = OrchestratorAgent(**mock_agents)

        result = await orchestrator.run(
            project_name="test_project",
            source_file="main.c",
            function_name="process"
        )

        assert result is not None
        mock_agents["code_analyzer"].analyze.assert_called_once()
        mock_agents["context_builder"].build_context.assert_called_once()
        mock_agents["model_generator"].generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_agents):
        """测试错误处理"""
        mock_agents["code_analyzer"].analyze.side_effect = Exception("Analysis failed")

        orchestrator = OrchestratorAgent(**mock_agents)

        with pytest.raises(Exception):
            await orchestrator.run(
                project_name="test_project",
                source_file="main.c",
                function_name="process"
            )
```

### 3.4 Phase 3 验收清单

| 序号 | 验收项         | 验收方式        | 通过标准           |
| ---- | -------------- | --------------- | ------------------ |
| 1    | Agent 基类     | 单元测试        | 接口测试通过       |
| 2    | CodeAnalyzer   | 单元测试 + Mock | 分析功能测试通过   |
| 3    | ContextBuilder | 单元测试 + Mock | 上下文构建测试通过 |
| 4    | ModelGenerator | 单元测试 + Mock | 生成功能测试通过   |
| 5    | Orchestrator   | 单元测试 + Mock | 流程编排测试通过   |
| 6    | 测试覆盖率     | pytest-cov      | 覆盖率 ≥ 80%       |

**验收命令**:

```bash
pytest tests/test_phase3/ -v --cov=fuzz_generator/agents --cov-report=term-missing
```

---

## Phase 4: 批量任务处理

### 4.1 阶段目标

实现批量分析任务的管理和执行，包括任务解析、并发控制、断点续传。

### 4.2 交付物清单

| 序号 | 交付物     | 文件路径                           | 说明                    |
| ---- | ---------- | ---------------------------------- | ----------------------- |
| 4.1  | 任务解析器 | `fuzz_generator/batch/parser.py`   | 解析 YAML/JSON 任务文件 |
| 4.2  | 任务执行器 | `fuzz_generator/batch/executor.py` | 批量任务执行            |
| 4.3  | 状态管理   | `fuzz_generator/batch/state.py`    | 任务状态管理            |
| 4.4  | 单元测试   | `tests/test_phase4/`               | Phase 4 测试用例        |

### 4.3 详细任务分解

#### 4.3.1 任务解析器 (0.5 天)

**验收标准**:

- [ ] 支持 YAML 格式
- [ ] 支持 JSON 格式
- [ ] 验证任务文件格式
- [ ] 处理相对路径和绝对路径

**自动化测试**:

```python
# tests/test_phase4/test_parser.py
import pytest
from fuzz_generator.batch.parser import TaskParser

class TestTaskParser:
    def test_parse_yaml(self, tmp_path):
        """测试解析YAML文件"""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text("""
project_path: "/path/to/source"
tasks:
  - source_file: "main.c"
    function_name: "process"
    output_name: "ProcessModel"
  - source_file: "handler.c"
    function_name: "handle"
        """)

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        assert batch.project_path == "/path/to/source"
        assert len(batch.tasks) == 2
        assert batch.tasks[0].output_name == "ProcessModel"
        assert batch.tasks[1].output_name is None

    def test_parse_json(self, tmp_path):
        """测试解析JSON文件"""
        task_file = tmp_path / "tasks.json"
        task_file.write_text('''
{
    "project_path": "/path/to/source",
    "tasks": [
        {"source_file": "main.c", "function_name": "process"}
    ]
}
        ''')

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        assert len(batch.tasks) == 1

    def test_parse_invalid_file(self, tmp_path):
        """测试解析无效文件"""
        task_file = tmp_path / "invalid.yaml"
        task_file.write_text("invalid: yaml: content:")

        parser = TaskParser()
        with pytest.raises(ValueError):
            parser.parse(str(task_file))

    def test_validate_required_fields(self, tmp_path):
        """测试验证必需字段"""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text("""
tasks:
  - source_file: "main.c"
        """)  # 缺少function_name

        parser = TaskParser()
        with pytest.raises(ValueError):
            parser.parse(str(task_file))
```

#### 4.3.2 任务执行器 (1 天)

**验收标准**:

- [ ] 支持顺序执行
- [ ] 支持并发执行（可配置）
- [ ] 支持单任务失败不影响其他任务
- [ ] 支持进度报告

**自动化测试**:

```python
# tests/test_phase4/test_executor.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from fuzz_generator.batch.executor import BatchExecutor

class TestBatchExecutor:
    @pytest.fixture
    def mock_orchestrator(self):
        orchestrator = AsyncMock()
        orchestrator.run.return_value = MagicMock(
            xml_content="<DataModel>...</DataModel>"
        )
        return orchestrator

    @pytest.mark.asyncio
    async def test_execute_batch(self, mock_orchestrator):
        """测试批量执行"""
        batch = MagicMock()
        batch.tasks = [
            MagicMock(source_file="a.c", function_name="func_a"),
            MagicMock(source_file="b.c", function_name="func_b")
        ]
        batch.project_path = "/path"

        executor = BatchExecutor(orchestrator=mock_orchestrator)
        results = await executor.execute(batch)

        assert len(results) == 2
        assert mock_orchestrator.run.call_count == 2

    @pytest.mark.asyncio
    async def test_partial_failure(self, mock_orchestrator):
        """测试部分任务失败"""
        mock_orchestrator.run.side_effect = [
            MagicMock(xml_content="<DataModel>...</DataModel>"),
            Exception("Failed"),
            MagicMock(xml_content="<DataModel>...</DataModel>")
        ]

        batch = MagicMock()
        batch.tasks = [MagicMock() for _ in range(3)]
        batch.project_path = "/path"

        executor = BatchExecutor(orchestrator=mock_orchestrator)
        results = await executor.execute(batch, fail_fast=False)

        assert len([r for r in results if r.success]) == 2
        assert len([r for r in results if not r.success]) == 1

    @pytest.mark.asyncio
    async def test_progress_callback(self, mock_orchestrator):
        """测试进度回调"""
        progress_calls = []

        def on_progress(completed, total, task):
            progress_calls.append((completed, total))

        batch = MagicMock()
        batch.tasks = [MagicMock() for _ in range(3)]
        batch.project_path = "/path"

        executor = BatchExecutor(orchestrator=mock_orchestrator)
        await executor.execute(batch, on_progress=on_progress)

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)
```

#### 4.3.3 断点续传 (0.5 天)

**验收标准**:

- [ ] 任务中断后能保存状态
- [ ] 支持从上次中断处继续
- [ ] 跳过已完成的任务

**自动化测试**:

```python
# tests/test_phase4/test_resume.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from fuzz_generator.batch.state import BatchStateManager
from fuzz_generator.models import TaskStatus

class TestBatchStateManager:
    @pytest.fixture
    def storage(self, tmp_path):
        from fuzz_generator.storage import JsonStorage
        return JsonStorage(base_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_save_and_load_state(self, storage):
        """测试保存和加载状态"""
        manager = BatchStateManager(storage=storage)

        batch_id = "batch_001"
        state = {
            "completed": ["task_1", "task_2"],
            "failed": ["task_3"],
            "pending": ["task_4", "task_5"]
        }

        await manager.save_state(batch_id, state)
        loaded = await manager.load_state(batch_id)

        assert loaded == state

    @pytest.mark.asyncio
    async def test_get_pending_tasks(self, storage):
        """测试获取待处理任务"""
        manager = BatchStateManager(storage=storage)

        batch_id = "batch_001"
        await manager.save_state(batch_id, {
            "completed": ["task_1"],
            "failed": [],
            "pending": ["task_2", "task_3"]
        })

        pending = await manager.get_pending_tasks(batch_id)
        assert pending == ["task_2", "task_3"]

    @pytest.mark.asyncio
    async def test_mark_completed(self, storage):
        """测试标记完成"""
        manager = BatchStateManager(storage=storage)

        batch_id = "batch_001"
        await manager.save_state(batch_id, {
            "completed": [],
            "failed": [],
            "pending": ["task_1", "task_2"]
        })

        await manager.mark_completed(batch_id, "task_1")

        state = await manager.load_state(batch_id)
        assert "task_1" in state["completed"]
        assert "task_1" not in state["pending"]
```

### 4.4 Phase 4 验收清单

| 序号 | 验收项     | 验收方式        | 通过标准         |
| ---- | ---------- | --------------- | ---------------- |
| 1    | YAML 解析  | 单元测试        | 解析测试通过     |
| 2    | JSON 解析  | 单元测试        | 解析测试通过     |
| 3    | 批量执行   | 单元测试 + Mock | 执行测试通过     |
| 4    | 断点续传   | 单元测试        | 状态管理测试通过 |
| 5    | 测试覆盖率 | pytest-cov      | 覆盖率 ≥ 85%     |

**验收命令**:

```bash
pytest tests/test_phase4/ -v --cov=fuzz_generator/batch --cov-report=term-missing
```

---

## Phase 5: XML 生成器

### 5.1 阶段目标

实现 Secray 格式的 XML DataModel 生成器。

### 5.2 交付物清单

| 序号 | 交付物     | 文件路径                                     | 说明               |
| ---- | ---------- | -------------------------------------------- | ------------------ |
| 5.1  | XML 模型   | `fuzz_generator/models/xml_models.py`        | DataModel 数据结构 |
| 5.2  | XML 生成器 | `fuzz_generator/generators/xml_generator.py` | XML 生成逻辑       |
| 5.3  | XML 模板   | `fuzz_generator/generators/templates/`       | Jinja2 模板        |
| 5.4  | XML 验证器 | `fuzz_generator/generators/validator.py`     | XML 格式验证       |
| 5.5  | 单元测试   | `tests/test_phase5/`                         | Phase 5 测试用例   |

### 5.3 详细任务分解

#### 5.3.1 XML 数据模型 (0.5 天)

**验收标准**:

- [ ] 定义 String、Block、Choice 等元素模型
- [ ] 支持嵌套结构
- [ ] 支持序列化为 dict

**自动化测试**:

```python
# tests/test_phase5/test_xml_models.py
import pytest
from fuzz_generator.models.xml_models import (
    StringElement, BlockElement, ChoiceElement, DataModel
)

class TestXMLModels:
    def test_string_element(self):
        """测试String元素"""
        elem = StringElement(
            name="Method",
            value="GET",
            token=True,
            mutable=False
        )
        assert elem.name == "Method"
        assert elem.token is True

    def test_block_element(self):
        """测试Block元素"""
        elem = BlockElement(
            name="Header",
            ref="HeaderLine",
            min_occurs=0,
            max_occurs="unbounded"
        )
        assert elem.ref == "HeaderLine"

    def test_choice_element(self):
        """测试Choice元素"""
        elem = ChoiceElement(
            name="EndChoice",
            options=[
                StringElement(name="CRLF", value="\\r\\n", token=True),
                StringElement(name="LF", value="\\n", token=True)
            ]
        )
        assert len(elem.options) == 2

    def test_datamodel(self):
        """测试DataModel"""
        model = DataModel(
            name="Request",
            elements=[
                StringElement(name="Method"),
                StringElement(name="Space", value=" ", token=True)
            ]
        )
        assert model.name == "Request"
        assert len(model.elements) == 2

    def test_nested_structure(self):
        """测试嵌套结构"""
        inner = DataModel(
            name="CrLf",
            elements=[StringElement(name="End", value="\\r\\n")]
        )
        outer = DataModel(
            name="Request",
            elements=[
                StringElement(name="Method"),
                BlockElement(name="End", ref="CrLf")
            ]
        )
        assert outer.elements[1].ref == "CrLf"
```

#### 5.3.2 XML 生成器 (1 天)

**验收标准**:

- [ ] 能够生成有效的 XML
- [ ] 支持格式化输出
- [ ] 支持注释添加
- [ ] XML 编码正确

**自动化测试**:

```python
# tests/test_phase5/test_xml_generator.py
import pytest
import xml.etree.ElementTree as ET
from fuzz_generator.generators.xml_generator import XMLGenerator
from fuzz_generator.models.xml_models import DataModel, StringElement, BlockElement

class TestXMLGenerator:
    @pytest.fixture
    def generator(self):
        return XMLGenerator()

    def test_generate_simple_datamodel(self, generator):
        """测试生成简单DataModel"""
        model = DataModel(
            name="Request",
            elements=[
                StringElement(name="Method", value="GET"),
                StringElement(name="Space", value=" ", token=True)
            ]
        )

        xml_str = generator.generate([model])

        # 验证XML有效性
        root = ET.fromstring(xml_str)
        assert root.tag == "Secray"

        datamodel = root.find("DataModel")
        assert datamodel.get("name") == "Request"

        strings = datamodel.findall("String")
        assert len(strings) == 2

    def test_generate_with_block(self, generator):
        """测试生成带Block的DataModel"""
        models = [
            DataModel(
                name="CrLf",
                elements=[StringElement(name="End", value="\\r\\n", token=True)]
            ),
            DataModel(
                name="Request",
                elements=[
                    StringElement(name="Method"),
                    BlockElement(name="End", ref="CrLf")
                ]
            )
        ]

        xml_str = generator.generate(models)
        root = ET.fromstring(xml_str)

        datamodels = root.findall("DataModel")
        assert len(datamodels) == 2

    def test_generate_with_choice(self, generator):
        """测试生成带Choice的DataModel"""
        from fuzz_generator.models.xml_models import ChoiceElement

        model = DataModel(
            name="LineEnd",
            elements=[
                ChoiceElement(
                    name="EndChoice",
                    options=[
                        StringElement(name="CRLF", value="\\r\\n", token=True),
                        StringElement(name="LF", value="\\n", token=True)
                    ]
                )
            ]
        )

        xml_str = generator.generate([model])
        root = ET.fromstring(xml_str)

        choice = root.find(".//Choice")
        assert choice is not None
        assert len(choice.findall("String")) == 2

    def test_xml_formatting(self, generator):
        """测试XML格式化"""
        model = DataModel(
            name="Test",
            elements=[StringElement(name="Field")]
        )

        xml_str = generator.generate([model], indent=4)

        # 验证缩进
        lines = xml_str.split('\n')
        assert any(line.startswith('    ') for line in lines)

    def test_xml_encoding(self, generator):
        """测试XML编码"""
        model = DataModel(
            name="Test",
            elements=[StringElement(name="Field", value="中文")]
        )

        xml_str = generator.generate([model])

        assert 'encoding="utf-8"' in xml_str
        assert "中文" in xml_str
```

#### 5.3.3 XML 验证器 (0.5 天)

**验收标准**:

- [ ] 验证 XML 语法
- [ ] 验证 DataModel 结构
- [ ] 提供详细的错误信息

**自动化测试**:

```python
# tests/test_phase5/test_validator.py
import pytest
from fuzz_generator.generators.validator import XMLValidator

class TestXMLValidator:
    @pytest.fixture
    def validator(self):
        return XMLValidator()

    def test_validate_valid_xml(self, validator):
        """测试验证有效XML"""
        xml_str = '''<?xml version="1.0" encoding="utf-8"?>
        <Secray>
            <DataModel name="Test">
                <String name="Field" value="value" />
            </DataModel>
        </Secray>
        '''

        result = validator.validate(xml_str)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_invalid_syntax(self, validator):
        """测试验证无效语法"""
        xml_str = '<Secray><DataModel></Secray>'

        result = validator.validate(xml_str)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_missing_name(self, validator):
        """测试验证缺少name属性"""
        xml_str = '''<?xml version="1.0"?>
        <Secray>
            <DataModel>
                <String value="value" />
            </DataModel>
        </Secray>
        '''

        result = validator.validate(xml_str)
        assert result.is_valid is False
        assert any("name" in err.lower() for err in result.errors)

    def test_validate_invalid_ref(self, validator):
        """测试验证无效引用"""
        xml_str = '''<?xml version="1.0"?>
        <Secray>
            <DataModel name="Test">
                <Block name="Invalid" ref="NonExistent" />
            </DataModel>
        </Secray>
        '''

        result = validator.validate(xml_str)
        assert result.is_valid is False
        assert any("ref" in err.lower() or "reference" in err.lower() for err in result.errors)
```

### 5.4 Phase 5 验收清单

| 序号 | 验收项     | 验收方式   | 通过标准     |
| ---- | ---------- | ---------- | ------------ |
| 1    | XML 模型   | 单元测试   | 模型测试通过 |
| 2    | XML 生成   | 单元测试   | 生成测试通过 |
| 3    | XML 验证   | 单元测试   | 验证测试通过 |
| 4    | 格式化输出 | 单元测试   | 格式测试通过 |
| 5    | 测试覆盖率 | pytest-cov | 覆盖率 ≥ 90% |

**验收命令**:

```bash
pytest tests/test_phase5/ -v --cov=fuzz_generator/generators --cov-report=term-missing
```

---

## Phase 6: 集成与优化

### 6.1 阶段目标

集成所有模块，完成端到端测试，性能优化。

### 6.2 交付物清单

| 序号 | 交付物   | 文件路径             | 说明         |
| ---- | -------- | -------------------- | ------------ |
| 6.1  | 集成测试 | `tests/integration/` | 端到端测试   |
| 6.2  | 性能测试 | `tests/performance/` | 性能基准测试 |
| 6.3  | 示例项目 | `examples/`          | 使用示例     |
| 6.4  | 用户文档 | `docs/USER_GUIDE.md` | 用户指南     |

### 6.3 集成测试设计

```python
# tests/integration/test_e2e.py
import pytest
import subprocess
from pathlib import Path

class TestEndToEnd:
    @pytest.fixture
    def sample_project(self, tmp_path):
        """创建示例C项目"""
        project_dir = tmp_path / "sample_project"
        project_dir.mkdir()

        # 创建测试C文件
        (project_dir / "handler.c").write_text('''
        #include <stdio.h>
        #include <string.h>

        int process_request(char* buffer, int length) {
            char local_buf[256];
            if (length > 0 && length < 256) {
                strncpy(local_buf, buffer, length);
                printf("Processing: %s\\n", local_buf);
                return 0;
            }
            return -1;
        }
        ''')

        return project_dir

    @pytest.fixture
    def task_file(self, tmp_path, sample_project):
        """创建任务文件"""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(f'''
project_path: "{sample_project}"
tasks:
  - source_file: "handler.c"
    function_name: "process_request"
    output_name: "RequestModel"
        ''')
        return task_file

    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("/path/to/joern").exists(),
        reason="Joern not available"
    )
    def test_full_analysis_flow(self, sample_project, task_file, tmp_path):
        """测试完整分析流程"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # 运行分析命令
        result = subprocess.run([
            "python", "-m", "fuzz_generator",
            "analyze",
            "--task-file", str(task_file),
            "--output", str(output_dir)
        ], capture_output=True, text=True)

        # 验证结果
        assert result.returncode == 0

        # 验证输出文件存在
        output_files = list(output_dir.glob("*.xml"))
        assert len(output_files) > 0

        # 验证XML内容
        xml_content = output_files[0].read_text()
        assert "<DataModel" in xml_content
        assert "RequestModel" in xml_content

    @pytest.mark.integration
    def test_resume_interrupted_batch(self, task_file, tmp_path):
        """测试断点续传"""
        # 第一次运行（模拟中断）
        # 第二次运行（恢复执行）
        pass
```

### 6.4 Phase 6 验收清单

| 序号 | 验收项     | 验收方式 | 通过标准         |
| ---- | ---------- | -------- | ---------------- |
| 1    | 端到端流程 | 集成测试 | E2E 测试通过     |
| 2    | CLI 完整性 | 手动测试 | 所有命令可用     |
| 3    | 错误处理   | 集成测试 | 错误场景处理正确 |
| 4    | 性能基准   | 性能测试 | 满足性能要求     |
| 5    | 文档完整   | 人工审核 | 文档覆盖所有功能 |

**验收命令**:

```bash
# 运行所有测试
pytest tests/ -v --cov=fuzz_generator --cov-report=html

# 运行集成测试（需要MCP服务器）
pytest tests/integration/ -v -m integration

# 运行性能测试
pytest tests/performance/ -v
```

---

## Phase 7: 文档与示例

### 7.1 阶段目标

完善项目文档，提供完整的使用示例和 API 文档。

### 7.2 交付物清单

| 序号 | 交付物   | 文件路径                | 说明         |
| ---- | -------- | ----------------------- | ------------ |
| 7.1  | 用户指南 | `docs/USER_GUIDE.md`    | 详细使用指南 |
| 7.2  | API 文档 | `docs/API_REFERENCE.md` | 编程接口文档 |
| 7.3  | 配置指南 | `docs/CONFIGURATION.md` | 配置项说明   |
| 7.4  | 示例项目 | `examples/`             | 完整使用示例 |
| 7.5  | README   | `README.md`             | 项目主页文档 |

### 7.3 详细任务分解

#### 7.3.1 用户指南 (0.5 天)

**交付内容**:

- 快速开始指南
- 命令行使用说明
- 批量分析使用说明
- 自定义 Prompt 配置说明
- 常见问题解答（FAQ）

**验收标准**:

- [ ] 新用户可根据文档完成首次分析
- [ ] 所有 CLI 命令有示例
- [ ] 配置项有详细说明

#### 7.3.2 示例项目 (0.5 天)

**交付文件**:

```
examples/
├── basic_analysis/           # 基础分析示例
│   ├── sample_code/
│   │   └── handler.c
│   ├── tasks.yaml
│   ├── knowledge.md
│   └── README.md
├── batch_analysis/           # 批量分析示例
│   ├── rtsp_server/
│   │   ├── handler.c
│   │   ├── parser.c
│   │   └── session.c
│   ├── tasks.yaml
│   └── README.md
└── custom_prompts/           # 自定义Prompt示例
    ├── prompts/
    │   └── protocol_analyzer.yaml
    └── README.md
```

**验收标准**:

- [ ] 每个示例包含完整的运行说明
- [ ] 示例可独立运行
- [ ] 示例覆盖主要使用场景

#### 7.3.3 API 文档 (0.5 天)

**交付内容**:

- 核心类和函数文档
- Agent 接口说明
- 扩展开发指南

**验收标准**:

- [ ] 所有公开 API 有文档
- [ ] 有代码示例
- [ ] 有类型说明

### 7.4 Phase 7 验收清单

| 序号 | 验收项   | 验收方式 | 通过标准       |
| ---- | -------- | -------- | -------------- |
| 1    | 用户指南 | 人工审核 | 内容完整、准确 |
| 2    | 示例项目 | 实际运行 | 所有示例可运行 |
| 3    | API 文档 | 人工审核 | 覆盖公开 API   |
| 4    | README   | 人工审核 | 信息完整       |

**验收方式**:

```bash
# 验证示例可运行
cd examples/basic_analysis
python -m fuzz_generator analyze --task-file tasks.yaml --output output/
```

---

## 测试策略总览

### 测试分层

```
┌─────────────────────────────────────────────────────────┐
│                    E2E Tests (Phase 6)                  │
│                  端到端测试，验证完整流程                  │
├─────────────────────────────────────────────────────────┤
│              Integration Tests (Phase 2-6)              │
│            集成测试，验证模块间交互（需要外部服务）          │
├─────────────────────────────────────────────────────────┤
│               Unit Tests (Phase 1-5)                    │
│              单元测试，验证独立模块功能                    │
└─────────────────────────────────────────────────────────┘
```

### 测试目录结构

```
tests/
├── conftest.py               # pytest配置和共享fixture
├── test_phase1/              # Phase 1 单元测试
│   ├── test_config.py
│   ├── test_logger.py
│   ├── test_storage.py
│   ├── test_models.py
│   └── test_cli.py
├── test_phase2/              # Phase 2 单元测试
│   ├── test_mcp_client.py
│   └── test_tools.py
├── test_phase3/              # Phase 3 单元测试
│   ├── test_base_agent.py
│   ├── test_code_analyzer.py
│   ├── test_context_builder.py
│   ├── test_model_generator.py
│   └── test_orchestrator.py
├── test_phase4/              # Phase 4 单元测试
│   ├── test_parser.py
│   ├── test_executor.py
│   └── test_resume.py
├── test_phase5/              # Phase 5 单元测试
│   ├── test_xml_models.py
│   ├── test_xml_generator.py
│   └── test_validator.py
├── integration/              # 集成测试
│   ├── conftest.py
│   ├── test_mcp_integration.py
│   └── test_e2e.py
└── performance/              # 性能测试
    └── test_benchmarks.py
```

### CI/CD 配置

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run unit tests
        run: pytest tests/test_phase*/ -v --cov=fuzz_generator --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      joern:
        image: joern/joern
        ports:
          - 8080:8080
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run integration tests
        run: pytest tests/integration/ -v -m integration
```

### 验收自动化脚本

```bash
#!/bin/bash
# scripts/verify_phase.sh

PHASE=$1

if [ -z "$PHASE" ]; then
    echo "Usage: ./verify_phase.sh <phase_number>"
    exit 1
fi

echo "=========================================="
echo "Verifying Phase $PHASE"
echo "=========================================="

# 运行对应阶段的测试
pytest tests/test_phase${PHASE}/ -v --cov=fuzz_generator --cov-report=term-missing --cov-fail-under=80

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Phase $PHASE PASSED ✓"
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "Phase $PHASE FAILED ✗"
    echo "=========================================="
    exit 1
fi
```

---

## 附录：快速参考

### 各阶段验收命令速查

```bash
# Phase 1: 基础框架
pytest tests/test_phase1/ -v --cov-fail-under=80

# Phase 2: MCP客户端
pytest tests/test_phase2/ -v --cov-fail-under=85

# Phase 3: Agent实现
pytest tests/test_phase3/ -v --cov-fail-under=80

# Phase 4: 批量任务
pytest tests/test_phase4/ -v --cov-fail-under=85

# Phase 5: XML生成器
pytest tests/test_phase5/ -v --cov-fail-under=90

# Phase 6: 集成测试
pytest tests/integration/ -v -m integration

# 全量测试
pytest tests/ -v --cov=fuzz_generator --cov-report=html
```

### 开发环境准备

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装开发依赖
pip install -e ".[dev]"

# 安装pre-commit hooks
pre-commit install
```

---

**文档维护**：

- 创建日期：2024-12-19
- 最后更新：2024-12-19
