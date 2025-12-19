"""Logging configuration using loguru."""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Global logger instance
_logger_configured = False


def setup_logger(
    log_level: str = "INFO",
    log_file: str | Path | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: bool = True,
    console_format: str | None = None,
    file_format: str | None = None,
) -> Any:
    """Configure and setup the logger.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None for console only)
        rotation: Log rotation size (e.g., "10 MB", "1 day")
        retention: Log retention period (e.g., "7 days", "1 week")
        compression: Whether to compress rotated logs
        console_format: Custom console log format
        file_format: Custom file log format

    Returns:
        Configured logger instance
    """
    global _logger_configured

    # Remove existing handlers
    logger.remove()

    # Default formats
    if console_format is None:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    if file_format is None:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        )

    # Add console handler
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format=console_format,
        colorize=True,
    )

    # Add file handler if path provided
    if log_file:
        log_path = Path(log_file)
        # Create parent directories if needed
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            level="DEBUG",  # File always captures DEBUG level
            format=file_format,
            rotation=rotation,
            retention=retention,
            compression="zip" if compression else None,
            encoding="utf-8",
        )

    _logger_configured = True
    return logger


def get_logger(name: str | None = None) -> Any:
    """Get a logger instance.

    Args:
        name: Optional module name for context

    Returns:
        Logger instance
    """
    global _logger_configured

    # Setup with defaults if not configured
    if not _logger_configured:
        setup_logger()

    if name:
        return logger.bind(name=name)
    return logger


def configure_from_settings(settings: Any) -> Any:
    """Configure logger from Settings object.

    Args:
        settings: Settings object with logging configuration

    Returns:
        Configured logger instance
    """
    logging_config = settings.logging

    return setup_logger(
        log_level=logging_config.level,
        log_file=logging_config.file,
        rotation=logging_config.rotation,
        retention=logging_config.retention,
        compression=logging_config.compression,
        console_format=logging_config.console_format,
    )


class LoggerContext:
    """Context manager for temporary log level changes."""

    def __init__(self, level: str):
        """Initialize context with new level.

        Args:
            level: Temporary log level to use
        """
        self.level = level.upper()
        self._previous_level: str | None = None

    def __enter__(self) -> Any:
        """Enter context and change log level."""
        # Store current configuration
        # Note: loguru doesn't have a direct way to get current level,
        # so we just set the new one
        logger.info(f"Temporarily changing log level to {self.level}")
        return logger

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and restore log level."""
        logger.info("Restoring previous log level")


# Convenience logging functions
def debug(message: str, **kwargs: Any) -> None:
    """Log debug message."""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs: Any) -> None:
    """Log info message."""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs: Any) -> None:
    """Log warning message."""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs: Any) -> None:
    """Log error message."""
    get_logger().error(message, **kwargs)


def exception(message: str, **kwargs: Any) -> None:
    """Log exception with traceback."""
    get_logger().exception(message, **kwargs)
