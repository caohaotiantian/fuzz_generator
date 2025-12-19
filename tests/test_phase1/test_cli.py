"""Test CLI commands."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from fuzz_generator.cli import cli


class TestCLIBasic:
    """Test basic CLI functionality."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_help(self, runner: CliRunner):
        """Test help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Fuzz Generator" in result.output
        assert "analyze" in result.output
        assert "parse" in result.output

    def test_version(self, runner: CliRunner):
        """Test version command."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_analyze_help(self, runner: CliRunner):
        """Test analyze subcommand help."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--project-path" in result.output
        assert "--task-file" in result.output
        assert "--function" in result.output

    def test_parse_help(self, runner: CliRunner):
        """Test parse subcommand help."""
        result = runner.invoke(cli, ["parse", "--help"])
        assert result.exit_code == 0
        assert "--project-path" in result.output
        assert "--language" in result.output

    def test_results_help(self, runner: CliRunner):
        """Test results subcommand help."""
        result = runner.invoke(cli, ["results", "--help"])
        assert result.exit_code == 0
        assert "--task-id" in result.output
        assert "--list" in result.output

    def test_clean_help(self, runner: CliRunner):
        """Test clean subcommand help."""
        result = runner.invoke(cli, ["clean", "--help"])
        assert result.exit_code == 0
        assert "--all" in result.output
        assert "--cache-only" in result.output

    def test_tools_help(self, runner: CliRunner):
        """Test tools subcommand help."""
        result = runner.invoke(cli, ["tools", "--help"])
        assert result.exit_code == 0

    def test_status(self, runner: CliRunner):
        """Test status command."""
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Version" in result.output


class TestAnalyzeCommand:
    """Test analyze command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_analyze_missing_required(self, runner: CliRunner):
        """Test analyze with missing required parameters."""
        result = runner.invoke(cli, ["analyze"])
        assert result.exit_code != 0

    def test_analyze_invalid_path(self, runner: CliRunner):
        """Test analyze with invalid project path."""
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--project-path",
                "/nonexistent/path",
                "--source-file",
                "main.c",
                "--function",
                "test",
            ],
        )
        assert result.exit_code != 0

    def test_analyze_missing_function(self, runner: CliRunner, sample_project: Path):
        """Test analyze with missing function option."""
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--project-path",
                str(sample_project),
                "--source-file",
                "main.c",
            ],
        )
        assert result.exit_code != 0
        # Should indicate mode selection error
        assert (
            "single function mode" in result.output.lower() or "batch mode" in result.output.lower()
        )

    def test_analyze_single_mode(self, runner: CliRunner, sample_project: Path):
        """Test analyze in single function mode."""
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--project-path",
                str(sample_project),
                "--source-file",
                "main.c",
                "--function",
                "process_request",
            ],
        )
        # Command should parse successfully (actual analysis not implemented yet)
        assert result.exit_code == 0
        assert "Analyzing" in result.output or "not yet implemented" in result.output

    def test_analyze_batch_mode(self, runner: CliRunner, sample_project: Path, sample_tasks: Path):
        """Test analyze in batch mode."""
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--project-path",
                str(sample_project),
                "--task-file",
                str(sample_tasks),
            ],
        )
        assert result.exit_code == 0
        assert "batch" in result.output.lower() or "not yet implemented" in result.output

    def test_analyze_conflicting_modes(
        self, runner: CliRunner, sample_project: Path, sample_tasks: Path
    ):
        """Test analyze with both modes specified."""
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--project-path",
                str(sample_project),
                "--source-file",
                "main.c",
                "--function",
                "test",
                "--task-file",
                str(sample_tasks),
            ],
        )
        assert result.exit_code != 0
        assert "Cannot use both" in result.output


class TestParseCommand:
    """Test parse command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_parse_valid_project(self, runner: CliRunner, sample_project: Path):
        """Test parse with valid project."""
        result = runner.invoke(
            cli,
            ["parse", "--project-path", str(sample_project)],
        )
        assert result.exit_code == 0
        assert "Parsing" in result.output or "not yet implemented" in result.output

    def test_parse_with_name(self, runner: CliRunner, sample_project: Path):
        """Test parse with custom project name."""
        result = runner.invoke(
            cli,
            [
                "parse",
                "--project-path",
                str(sample_project),
                "--project-name",
                "my_project",
            ],
        )
        assert result.exit_code == 0


class TestCleanCommand:
    """Test clean command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_clean_missing_option(self, runner: CliRunner):
        """Test clean without specifying what to clean."""
        result = runner.invoke(cli, ["clean"])
        assert result.exit_code != 0
        assert "specify what to clean" in result.output.lower() or "Missing" in result.output

    def test_clean_cache_only(self, runner: CliRunner):
        """Test clean cache only."""
        result = runner.invoke(cli, ["clean", "--cache-only"])
        assert result.exit_code == 0
        assert "cache" in result.output.lower()


class TestToolsCommand:
    """Test tools command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_list_tools(self, runner: CliRunner):
        """Test listing tools."""
        result = runner.invoke(cli, ["tools"])
        assert result.exit_code == 0
        assert "parse_project" in result.output
        assert "get_function_code" in result.output

    def test_list_tools_detailed(self, runner: CliRunner):
        """Test listing tools with details."""
        result = runner.invoke(cli, ["tools", "--detailed"])
        assert result.exit_code == 0
        assert "Description" in result.output


class TestValidators:
    """Test input validators."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_invalid_config_file(self, runner: CliRunner, tmp_path: Path):
        """Test with invalid config file."""
        invalid_config = tmp_path / "invalid.txt"
        invalid_config.write_text("not yaml")

        result = runner.invoke(
            cli,
            ["--config", str(invalid_config), "status"],
        )
        assert result.exit_code != 0

    def test_invalid_task_file(self, runner: CliRunner, sample_project: Path, invalid_tasks: Path):
        """Test with invalid task file."""
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--project-path",
                str(sample_project),
                "--task-file",
                str(invalid_tasks),
            ],
        )
        assert result.exit_code != 0
        assert "function_name" in result.output.lower()
