"""Test package installation and CLI entry point."""

import subprocess
import sys


class TestInstallation:
    """Test package installation."""

    def test_package_importable(self):
        """Test that package can be imported."""
        import fuzz_generator

        assert fuzz_generator.__version__ is not None
        assert fuzz_generator.__version__ == "0.1.0"

    def test_cli_entry_point(self):
        """Test CLI entry point."""
        from fuzz_generator.cli import cli

        assert cli is not None

    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(
            [sys.executable, "-m", "fuzz_generator", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Fuzz Generator" in result.stdout

    def test_cli_version(self):
        """Test CLI version command."""
        result = subprocess.run(
            [sys.executable, "-m", "fuzz_generator", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout


class TestModuleImports:
    """Test that all modules can be imported."""

    def test_import_config(self):
        """Test config module import."""
        from fuzz_generator.config import Settings, load_config

        assert Settings is not None
        assert load_config is not None

    def test_import_utils(self):
        """Test utils module import."""
        from fuzz_generator.utils import get_logger, setup_logger

        assert setup_logger is not None
        assert get_logger is not None

    def test_import_storage(self):
        """Test storage module import."""
        from fuzz_generator.storage import CacheManager, JsonStorage

        assert JsonStorage is not None
        assert CacheManager is not None

    def test_import_models(self):
        """Test models module import."""
        from fuzz_generator.models import (
            AnalysisTask,
            BatchTask,
            DataModel,
            FunctionInfo,
        )

        assert AnalysisTask is not None
        assert BatchTask is not None
        assert FunctionInfo is not None
        assert DataModel is not None

    def test_import_cli(self):
        """Test CLI module import."""
        from fuzz_generator.cli import cli

        assert cli is not None
