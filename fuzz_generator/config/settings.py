"""Configuration settings models using Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LLMSettings(BaseModel):
    """LLM service configuration."""

    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="LLM API base URL",
    )
    api_key: str = Field(
        default="ollama",
        description="API key for LLM service",
    )
    model: str = Field(
        default="qwen2.5:32b",
        description="Model name to use",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Generation temperature (0.0-1.0)",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum tokens in response",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter",
    )
    timeout: int = Field(
        default=120,
        ge=1,
        description="Request timeout in seconds",
    )
    retry_count: int = Field(
        default=3,
        ge=0,
        description="Number of retries on failure",
    )
    retry_delay: float = Field(
        default=2.0,
        ge=0.0,
        description="Delay between retries in seconds",
    )


class MCPServerSettings(BaseModel):
    """MCP Server configuration."""

    url: str = Field(
        default="http://localhost:8000/mcp",
        description="Joern MCP Server URL",
    )
    timeout: int = Field(
        default=60,
        ge=1,
        description="Request timeout in seconds",
    )
    retry_count: int = Field(
        default=3,
        ge=0,
        description="Number of retries on failure",
    )
    retry_delay: float = Field(
        default=2.0,
        ge=0.0,
        description="Delay between retries in seconds",
    )
    max_connections: int = Field(
        default=10,
        ge=1,
        description="Maximum connections in pool",
    )
    keepalive_expiry: int = Field(
        default=30,
        ge=0,
        description="Connection keepalive expiry in seconds",
    )


class SingleAgentSettings(BaseModel):
    """Configuration for a single agent."""

    system_prompt_file: str | None = Field(
        default=None,
        description="Path to system prompt file (relative to prompts directory)",
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        description="Maximum iterations for agent",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="List of tools available to this agent",
    )


class AgentCommonSettings(BaseModel):
    """Common settings for all agents."""

    verbose: bool = Field(
        default=True,
        description="Whether to show agent thinking process",
    )
    human_input_mode: Literal["NEVER", "ALWAYS", "TERMINATE"] = Field(
        default="NEVER",
        description="Human input mode for agents",
    )


class AgentSettings(BaseModel):
    """Agent configuration."""

    common: AgentCommonSettings = Field(default_factory=AgentCommonSettings)
    orchestrator: SingleAgentSettings = Field(
        default_factory=lambda: SingleAgentSettings(
            system_prompt_file="orchestrator.yaml",
            max_iterations=50,
        )
    )
    code_analyzer: SingleAgentSettings = Field(
        default_factory=lambda: SingleAgentSettings(
            system_prompt_file="code_analyzer.yaml",
            max_iterations=10,
            tools=["get_function_code", "list_functions", "search_code"],
        )
    )
    context_builder: SingleAgentSettings = Field(
        default_factory=lambda: SingleAgentSettings(
            system_prompt_file="context_builder.yaml",
            max_iterations=15,
            tools=[
                "track_dataflow",
                "analyze_variable_flow",
                "find_data_dependencies",
                "get_callers",
                "get_callees",
                "get_call_chain",
                "get_control_flow_graph",
            ],
        )
    )
    model_generator: SingleAgentSettings = Field(
        default_factory=lambda: SingleAgentSettings(
            system_prompt_file="model_generator.yaml",
            max_iterations=5,
        )
    )


class BatchSettings(BaseModel):
    """Batch task processing configuration."""

    max_concurrent: int = Field(
        default=1,
        ge=0,
        description="Maximum concurrent tasks (0 = sequential)",
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop on first task failure",
    )
    task_timeout: int = Field(
        default=600,
        ge=1,
        description="Single task timeout in seconds",
    )
    progress_interval: int = Field(
        default=10,
        ge=1,
        description="Progress report interval in seconds",
    )


class StorageSettings(BaseModel):
    """Storage configuration."""

    work_dir: str = Field(
        default=".fuzz_generator",
        description="Working directory for cache and results",
    )
    enable_cache: bool = Field(
        default=True,
        description="Whether to enable result caching",
    )
    cache_expiry_hours: int = Field(
        default=0,
        ge=0,
        description="Cache expiry in hours (0 = never expire)",
    )
    save_conversations: bool = Field(
        default=True,
        description="Whether to save agent conversations",
    )
    enable_resume: bool = Field(
        default=True,
        description="Whether to enable task resume",
    )


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    file: str | None = Field(
        default="logs/fuzz_generator.log",
        description="Log file path",
    )
    rotation: str = Field(
        default="10 MB",
        description="Log rotation size",
    )
    retention: str = Field(
        default="7 days",
        description="Log retention period",
    )
    compression: bool = Field(
        default=True,
        description="Whether to compress old logs",
    )
    console_format: str = Field(
        default="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        description="Console log format",
    )


class OutputSettings(BaseModel):
    """Output configuration."""

    format: Literal["xml"] = Field(
        default="xml",
        description="Output format",
    )
    encoding: str = Field(
        default="utf-8",
        description="Output encoding",
    )
    indent: int = Field(
        default=4,
        ge=0,
        description="Indentation spaces",
    )
    include_comments: bool = Field(
        default=True,
        description="Whether to include comments in output",
    )
    datamodel_only: bool = Field(
        default=True,
        description="Only generate DataModel (exclude StateModel and Test)",
    )
    default_output_dir: str = Field(
        default="./output",
        description="Default output directory",
    )


class AnalysisSettings(BaseModel):
    """Analysis configuration."""

    max_dataflow_depth: int = Field(
        default=10,
        ge=1,
        description="Maximum dataflow analysis depth",
    )
    max_call_chain_depth: int = Field(
        default=5,
        ge=1,
        description="Maximum call chain depth",
    )
    max_flows: int = Field(
        default=20,
        ge=1,
        description="Maximum flows per analysis",
    )
    include_indirect_calls: bool = Field(
        default=True,
        description="Whether to include indirect calls",
    )
    timeout: int = Field(
        default=300,
        ge=1,
        description="Analysis timeout in seconds",
    )


class KnowledgeSettings(BaseModel):
    """Custom knowledge configuration."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable custom knowledge",
    )
    knowledge_dir: str = Field(
        default="./knowledge",
        description="Knowledge files directory",
    )
    default_file: str | None = Field(
        default=None,
        description="Default knowledge file",
    )


class Settings(BaseModel):
    """Main configuration settings."""

    version: str = Field(
        default="1.0",
        description="Configuration version",
    )
    llm: LLMSettings = Field(default_factory=LLMSettings)
    mcp_server: MCPServerSettings = Field(default_factory=MCPServerSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    batch: BatchSettings = Field(default_factory=BatchSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate configuration version."""
        supported_versions = ["1.0"]
        if v not in supported_versions:
            raise ValueError(f"Unsupported config version: {v}. Supported: {supported_versions}")
        return v

    def get_work_dir(self) -> Path:
        """Get the working directory as Path."""
        return Path(self.storage.work_dir)

    def get_cache_dir(self) -> Path:
        """Get the cache directory as Path."""
        return self.get_work_dir() / "cache"

    def get_results_dir(self) -> Path:
        """Get the results directory as Path."""
        return self.get_work_dir() / "results"

    def get_logs_dir(self) -> Path:
        """Get the logs directory as Path."""
        return self.get_work_dir() / "logs"
