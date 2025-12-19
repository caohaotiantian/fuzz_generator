"""Test configuration module."""

from pathlib import Path

import pytest

from fuzz_generator.config import (
    LLMSettings,
    MCPServerSettings,
    Settings,
    get_settings,
    load_config,
)
from fuzz_generator.config.loader import ConfigurationError, reset_settings


class TestLLMSettings:
    """Test LLM settings model."""

    def test_default_values(self):
        """Test default values."""
        settings = LLMSettings()
        assert settings.base_url == "http://localhost:11434/v1"
        assert settings.model == "qwen2.5:32b"
        assert settings.temperature == 0.7
        assert settings.max_tokens == 4096

    def test_custom_values(self):
        """Test custom values."""
        settings = LLMSettings(
            base_url="http://custom:8080/v1",
            model="custom-model",
            temperature=0.5,
        )
        assert settings.base_url == "http://custom:8080/v1"
        assert settings.model == "custom-model"
        assert settings.temperature == 0.5

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        LLMSettings(temperature=0.0)
        LLMSettings(temperature=0.5)
        LLMSettings(temperature=1.0)

        # Invalid temperature
        with pytest.raises(ValueError):
            LLMSettings(temperature=1.5)

        with pytest.raises(ValueError):
            LLMSettings(temperature=-0.1)


class TestMCPServerSettings:
    """Test MCP server settings model."""

    def test_default_values(self):
        """Test default values."""
        settings = MCPServerSettings()
        assert settings.url == "http://localhost:8000/mcp"
        assert settings.timeout == 60
        assert settings.retry_count == 3

    def test_timeout_validation(self):
        """Test timeout validation."""
        MCPServerSettings(timeout=1)
        MCPServerSettings(timeout=300)

        with pytest.raises(ValueError):
            MCPServerSettings(timeout=0)


class TestSettings:
    """Test main Settings model."""

    def test_default_settings(self):
        """Test default settings creation."""
        settings = Settings()
        assert settings.version == "1.0"
        assert settings.llm is not None
        assert settings.mcp_server is not None
        assert settings.agents is not None

    def test_get_work_dir(self):
        """Test get_work_dir method."""
        settings = Settings()
        work_dir = settings.get_work_dir()
        assert isinstance(work_dir, Path)
        assert str(work_dir) == ".fuzz_generator"

    def test_get_cache_dir(self):
        """Test get_cache_dir method."""
        settings = Settings()
        cache_dir = settings.get_cache_dir()
        assert str(cache_dir) == ".fuzz_generator/cache"


class TestConfigLoading:
    """Test configuration loading."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        reset_settings()
        config = load_config()
        assert config.llm.base_url is not None
        assert config.mcp_server.url is not None

    def test_load_custom_config(self, tmp_path: Path):
        """Test loading custom configuration."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
version: "1.0"
llm:
  base_url: "http://custom:8000/v1"
  model: "custom-model"
  temperature: 0.5
""")
        config = load_config(str(config_file))
        assert config.llm.base_url == "http://custom:8000/v1"
        assert config.llm.model == "custom-model"
        assert config.llm.temperature == 0.5

    def test_load_partial_config(self, tmp_path: Path):
        """Test loading partial configuration with defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
version: "1.0"
llm:
  model: "partial-model"
""")
        config = load_config(str(config_file))
        # Custom value
        assert config.llm.model == "partial-model"
        # Default values
        assert config.llm.temperature == 0.7
        assert config.mcp_server.url == "http://localhost:8000/mcp"

    def test_config_validation_error(self, tmp_path: Path):
        """Test configuration validation error."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("""
version: "1.0"
llm:
  temperature: 2.0
""")
        with pytest.raises(ConfigurationError):
            load_config(str(config_file))

    def test_config_file_not_found(self):
        """Test loading non-existent config file."""
        with pytest.raises(ConfigurationError):
            load_config("/nonexistent/config.yaml")

    def test_invalid_yaml(self, tmp_path: Path):
        """Test loading invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content:")
        with pytest.raises(ConfigurationError):
            load_config(str(config_file))

    def test_env_override(self, monkeypatch, tmp_path: Path):
        """Test environment variable override."""
        monkeypatch.setenv("FUZZ_GENERATOR_LLM__BASE_URL", "http://env:9000")

        config = load_config()
        assert config.llm.base_url == "http://env:9000"

    def test_env_override_nested(self, monkeypatch):
        """Test nested environment variable override."""
        monkeypatch.setenv("FUZZ_GENERATOR_MCP_SERVER__TIMEOUT", "120")

        config = load_config()
        assert config.mcp_server.timeout == 120

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        reset_settings()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_reset_settings(self):
        """Test settings reset."""
        reset_settings()
        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()
        # Should be equal but not the same instance
        assert settings1.version == settings2.version
