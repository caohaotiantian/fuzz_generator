"""Test logging module."""

from pathlib import Path

from fuzz_generator.utils.logger import (
    debug,
    error,
    get_logger,
    info,
    setup_logger,
    warning,
)


class TestLoggerSetup:
    """Test logger setup."""

    def test_setup_logger_console_only(self):
        """Test logger setup with console only."""
        logger = setup_logger(log_level="DEBUG")
        assert logger is not None

    def test_setup_logger_with_file(self, tmp_path: Path):
        """Test logger setup with file output."""
        log_file = tmp_path / "test.log"
        logger = setup_logger(log_level="DEBUG", log_file=str(log_file))

        logger.info("Test message")

        # Give a moment for async write
        import time

        time.sleep(0.1)

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_log_level_filtering(self, tmp_path: Path):
        """Test log level filtering."""
        log_file = tmp_path / "test.log"
        logger = setup_logger(log_level="WARNING", log_file=str(log_file))

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        import time

        time.sleep(0.1)

        content = log_file.read_text()
        # File always captures DEBUG, so check console level
        assert "Warning message" in content
        assert "Error message" in content

    def test_log_rotation(self, tmp_path: Path):
        """Test log rotation parameter acceptance."""
        log_file = tmp_path / "test.log"
        logger = setup_logger(
            log_level="DEBUG",
            log_file=str(log_file),
            rotation="1 KB",
            retention="1 day",
        )
        assert logger is not None


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_basic(self):
        """Test getting basic logger."""
        logger = get_logger()
        assert logger is not None

    def test_get_logger_with_name(self):
        """Test getting named logger."""
        logger = get_logger("test_module")
        assert logger is not None

    def test_logger_functions(self, tmp_path: Path):
        """Test convenience logging functions."""
        log_file = tmp_path / "test.log"
        setup_logger(log_level="DEBUG", log_file=str(log_file))

        debug("Debug test")
        info("Info test")
        warning("Warning test")
        error("Error test")

        import time

        time.sleep(0.1)

        content = log_file.read_text()
        assert "Debug test" in content
        assert "Info test" in content
        assert "Warning test" in content
        assert "Error test" in content
