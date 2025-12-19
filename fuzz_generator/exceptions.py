"""Exception classes for fuzz_generator."""

from enum import Enum
from typing import Any


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class FuzzGeneratorError(Exception):
    """Base exception class for fuzz_generator."""

    severity: ErrorSeverity = ErrorSeverity.ERROR

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        """Initialize exception.

        Args:
            message: Error message
            details: Additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        """Return string representation."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(FuzzGeneratorError):
    """Configuration related errors."""

    severity = ErrorSeverity.FATAL


class MCPConnectionError(FuzzGeneratorError):
    """MCP server connection errors."""

    severity = ErrorSeverity.FATAL


class MCPToolError(FuzzGeneratorError):
    """MCP tool invocation errors."""

    severity = ErrorSeverity.ERROR


class LLMError(FuzzGeneratorError):
    """LLM service errors."""

    severity = ErrorSeverity.ERROR


class AnalysisError(FuzzGeneratorError):
    """Code analysis errors."""

    severity = ErrorSeverity.ERROR


class ValidationError(FuzzGeneratorError):
    """Output validation errors."""

    severity = ErrorSeverity.WARNING


class StorageError(FuzzGeneratorError):
    """Storage related errors."""

    severity = ErrorSeverity.ERROR


class TaskError(FuzzGeneratorError):
    """Task execution errors."""

    severity = ErrorSeverity.ERROR
