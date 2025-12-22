"""Test CLI commands."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from fuzz_generator.cli import cli
from fuzz_generator.models import TaskResult


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
        assert "--detailed" in result.output

    def test_status(self, runner: CliRunner, tmp_path: Path):
        """Test status command."""
        # Mock the MCP connection check to avoid actual connection
        with patch("fuzz_generator.tools.mcp_client.MCPHttpClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.list_tools.side_effect = Exception("Not connected")
            mock_client.return_value = mock_instance

            result = runner.invoke(cli, ["--work-dir", str(tmp_path), "status"])
            assert result.exit_code == 0
            assert "Fuzz Generator Status" in result.output


class TestAnalyzeCommand:
    """Test analyze command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_project(self, tmp_path: Path) -> Path:
        """Create sample project directory."""
        project_dir = tmp_path / "sample_project"
        project_dir.mkdir()
        (project_dir / "main.c").write_text(
            """
            int process_request(char* buffer, int size) {
                return 0;
            }
            """
        )
        return project_dir

    @pytest.fixture
    def sample_tasks(self, tmp_path: Path, sample_project: Path) -> Path:
        """Create sample tasks file."""
        tasks_file = tmp_path / "tasks.yaml"
        tasks_file.write_text(
            f"""
project_path: "{sample_project}"
tasks:
  - source_file: "main.c"
    function_name: "process_request"
    output_name: "RequestModel"
"""
        )
        return tasks_file

    def test_analyze_missing_required(self, runner: CliRunner):
        """Test analyze without required options."""
        result = runner.invoke(cli, ["analyze"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_analyze_invalid_path(self, runner: CliRunner, tmp_path: Path):
        """Test analyze with invalid project path."""
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--project-path",
                str(tmp_path / "nonexistent"),
                "--source-file",
                "test.c",
                "--function",
                "test",
            ],
        )
        assert result.exit_code != 0

    def test_analyze_missing_function(self, runner: CliRunner, sample_project: Path):
        """Test analyze with source file but no function."""
        result = runner.invoke(
            cli,
            ["analyze", "--project-path", str(sample_project), "--source-file", "main.c"],
        )
        assert result.exit_code != 0
        assert "Please specify" in result.output

    def test_analyze_single_mode(self, runner: CliRunner, sample_project: Path):
        """Test analyze in single function mode."""
        # Mock the analysis runner to avoid actual MCP connection
        with patch("fuzz_generator.cli.runner.AnalysisRunner") as mock_runner_class:
            mock_runner = AsyncMock()
            mock_runner.analyze_single_function.return_value = TaskResult(
                task_id="test",
                success=True,
                xml_content="<DataModel />",
            )
            mock_runner_class.return_value = mock_runner

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

            # With mocking, the command should succeed
            assert result.exit_code == 0

    def test_analyze_batch_mode(self, runner: CliRunner, sample_project: Path, sample_tasks: Path):
        """Test analyze in batch mode."""
        # Mock the analysis runner to avoid actual MCP connection
        with patch("fuzz_generator.cli.runner.AnalysisRunner") as mock_runner_class:
            mock_runner = AsyncMock()
            mock_runner.analyze_batch.return_value = {
                "batch_id": "test_batch",
                "total": 1,
                "completed": 1,
                "failed": 0,
            }
            mock_runner_class.return_value = mock_runner

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

            # With mocking, the command should succeed
            assert result.exit_code == 0

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

    @pytest.fixture
    def sample_project(self, tmp_path: Path) -> Path:
        """Create sample project directory."""
        project_dir = tmp_path / "sample_project"
        project_dir.mkdir()
        (project_dir / "main.c").write_text("int main() { return 0; }")
        return project_dir

    def test_parse_valid_project(self, runner: CliRunner, sample_project: Path):
        """Test parse with valid project."""
        # Mock the analysis runner to avoid actual MCP connection
        with patch("fuzz_generator.cli.runner.AnalysisRunner") as mock_runner_class:
            mock_runner = AsyncMock()
            mock_runner.parse_project.return_value = True
            mock_runner_class.return_value = mock_runner

            result = runner.invoke(
                cli,
                ["parse", "--project-path", str(sample_project)],
            )

            assert result.exit_code == 0

    def test_parse_with_name(self, runner: CliRunner, sample_project: Path):
        """Test parse with custom project name."""
        with patch("fuzz_generator.cli.runner.AnalysisRunner") as mock_runner_class:
            mock_runner = AsyncMock()
            mock_runner.parse_project.return_value = True
            mock_runner_class.return_value = mock_runner

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
        assert "Please specify" in result.output

    def test_clean_cache_only(self, runner: CliRunner, tmp_path: Path):
        """Test clean cache only."""
        with patch("fuzz_generator.cli.runner.CacheCleaner") as mock_cleaner_class:
            mock_cleaner = AsyncMock()
            mock_cleaner.clear_cache_only.return_value = 5
            mock_cleaner_class.return_value = mock_cleaner

            result = runner.invoke(
                cli,
                ["--work-dir", str(tmp_path), "clean", "--cache-only"],
            )

            assert result.exit_code == 0
            assert "Cleared" in result.output


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
        assert "Available MCP Tools" in result.output
        assert "get_function_code" in result.output
        assert "track_dataflow" in result.output

    def test_list_tools_detailed(self, runner: CliRunner):
        """Test listing tools with details."""
        result = runner.invoke(cli, ["tools", "--detailed"])
        assert result.exit_code == 0
        assert "Available MCP Tools" in result.output
        assert "Description:" in result.output


class TestValidators:
    """Test CLI validators."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_invalid_config_file(self, runner: CliRunner, tmp_path: Path):
        """Test invalid config file."""
        # Create an invalid config file
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content:")

        result = runner.invoke(
            cli,
            ["--config", str(invalid_config), "status"],
        )
        # Should fail due to invalid YAML
        assert result.exit_code != 0

    def test_invalid_task_file(self, runner: CliRunner, tmp_path: Path):
        """Test invalid task file in analyze command."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "main.c").write_text("int main() {}")

        # Create invalid task file
        invalid_tasks = tmp_path / "tasks.yaml"
        invalid_tasks.write_text("tasks: not_a_list")

        result = runner.invoke(
            cli,
            [
                "analyze",
                "--project-path",
                str(project_dir),
                "--task-file",
                str(invalid_tasks),
            ],
        )
        assert result.exit_code != 0
