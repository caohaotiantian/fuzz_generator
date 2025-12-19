"""Configuration loader module."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from fuzz_generator.config.settings import Settings


class ConfigurationError(Exception):
    """Configuration related errors."""

    pass


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary (values take precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML content as dictionary

    Raises:
        ConfigurationError: If file cannot be loaded or parsed
    """
    try:
        with open(path, encoding="utf-8") as f:
            content = yaml.safe_load(f)
            return content if content else {}
    except FileNotFoundError as e:
        raise ConfigurationError(f"Configuration file not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML file {path}: {e}") from e


def _get_default_config() -> dict[str, Any]:
    """Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return Settings().model_dump()


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to configuration.

    Environment variables should be prefixed with FUZZ_GENERATOR_ and use
    double underscore (__) to separate nested keys.

    Example:
        FUZZ_GENERATOR_LLM__BASE_URL=http://localhost:8080/v1
        FUZZ_GENERATOR_MCP_SERVER__TIMEOUT=120

    Args:
        config: Base configuration dictionary

    Returns:
        Configuration with environment overrides applied
    """
    prefix = "FUZZ_GENERATOR_"
    result = config.copy()

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Remove prefix and convert to lowercase
        config_key = key[len(prefix) :].lower()

        # Split by double underscore for nested keys
        parts = config_key.split("__")

        # Navigate to the correct nested location
        current = result
        for _i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # Skip if the path doesn't lead to a dict
                break
            current = current[part]
        else:
            # Set the value, attempting type conversion
            final_key = parts[-1]
            if final_key in current:
                # Try to preserve the original type
                original_value = current[final_key]
                if isinstance(original_value, bool):
                    current[final_key] = value.lower() in ("true", "1", "yes")
                elif isinstance(original_value, int):
                    try:
                        current[final_key] = int(value)
                    except ValueError:
                        current[final_key] = value
                elif isinstance(original_value, float):
                    try:
                        current[final_key] = float(value)
                    except ValueError:
                        current[final_key] = value
                else:
                    current[final_key] = value
            else:
                current[final_key] = value

    return result


def load_config(config_path: str | Path | None = None) -> Settings:
    """Load configuration from file with defaults and environment overrides.

    Loading order (later overrides earlier):
    1. Default configuration
    2. Configuration file (if provided)
    3. Environment variables

    Args:
        config_path: Optional path to configuration file

    Returns:
        Validated Settings object

    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Start with defaults
    config = _get_default_config()

    # Load from file if provided
    if config_path:
        path = Path(config_path)
        file_config = _load_yaml_file(path)
        config = _deep_merge(config, file_config)

    # Apply environment overrides
    config = _apply_env_overrides(config)

    # Validate and return
    try:
        return Settings.model_validate(config)
    except Exception as e:
        raise ConfigurationError(f"Invalid configuration: {e}") from e


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance.

    This function returns a cached Settings instance. The first call
    will load from the default location or environment.

    Returns:
        Cached Settings object
    """
    # Check for config file in environment
    config_path = os.environ.get("FUZZ_GENERATOR_CONFIG")

    # Check for default locations
    if not config_path:
        default_locations = [
            Path("config.yaml"),
            Path("config/config.yaml"),
            Path.home() / ".fuzz_generator" / "config.yaml",
        ]
        for location in default_locations:
            if location.exists():
                config_path = str(location)
                break

    return load_config(config_path)


def reset_settings() -> None:
    """Reset cached settings.

    This clears the settings cache, allowing reload on next get_settings() call.
    """
    get_settings.cache_clear()
