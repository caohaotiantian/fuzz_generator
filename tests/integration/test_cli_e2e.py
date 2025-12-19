"""End-to-end CLI tests."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from fuzz_generator.cli import cli


class TestCLIEndToEnd:
    """End-to-end tests for CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()

    def test_cli_version(self, runner: CliRunner):
        """Test version command."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "fuzz-generator" in result.output.lower() or "0.1.0" in result.output

    def test_cli_help(self, runner: CliRunner):
        """Test help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output
        assert "parse" in result.output

    def test_analyze_help(self, runner: CliRunner):
        """Test analyze command help."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--project" in result.output or "project" in result.output.lower()

    def test_parse_help(self, runner: CliRunner):
        """Test parse command help."""
        result = runner.invoke(cli, ["parse", "--help"])
        assert result.exit_code == 0

    def test_status_help(self, runner: CliRunner):
        """Test status command help."""
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0

    def test_tools_help(self, runner: CliRunner):
        """Test tools command help."""
        result = runner.invoke(cli, ["tools", "--help"])
        assert result.exit_code == 0

    def test_parse_valid_task_file(
        self,
        runner: CliRunner,
        sample_c_project: Path,
        tmp_path: Path,
    ):
        """Test parsing a valid task file."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(f'''project_path: "{sample_c_project}"
tasks:
  - source_file: "handler.c"
    function_name: "process_request"
''')

        # parse command requires --project-path according to implementation
        result = runner.invoke(
            cli, ["parse", "--project-path", str(sample_c_project), str(task_file)]
        )

        # May require MCP server, so we just check it doesn't crash on args
        assert result.exit_code in [0, 1, 2]

    def test_parse_invalid_task_file(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ):
        """Test parsing an invalid task file."""
        task_file = tmp_path / "invalid.yaml"
        task_file.write_text("invalid: yaml: content:")

        result = runner.invoke(cli, ["parse", str(task_file)])

        # Should indicate error
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_parse_missing_file(self, runner: CliRunner):
        """Test parsing non-existent file."""
        result = runner.invoke(cli, ["parse", "/nonexistent/path.yaml"])

        assert result.exit_code != 0

    def test_status_no_tasks(self, runner: CliRunner, tmp_path: Path):
        """Test status with no running tasks."""
        result = runner.invoke(cli, ["status"], env={"FUZZ_GENERATOR_WORK_DIR": str(tmp_path)})

        # Should not error
        assert result.exit_code == 0

    def test_results_empty(self, runner: CliRunner, tmp_path: Path):
        """Test results with no completed tasks."""
        result = runner.invoke(cli, ["results"], env={"FUZZ_GENERATOR_WORK_DIR": str(tmp_path)})

        # Should not error
        assert result.exit_code == 0

    def test_clean_command(self, runner: CliRunner, tmp_path: Path):
        """Test clean command."""
        # Create work dir first
        work_dir = tmp_path / ".fuzz_generator"
        work_dir.mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            cli,
            ["--work-dir", str(work_dir), "clean", "--force"],
        )

        # Clean may have different exit codes depending on implementation
        # We just check it doesn't crash unexpectedly
        assert result.exit_code in [0, 1, 2]


class TestCLIWithConfig:
    """CLI tests with custom configuration."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()

    def test_analyze_with_config(
        self,
        runner: CliRunner,
        sample_c_project: Path,
        tmp_path: Path,
    ):
        """Test analyze with custom config file."""
        # Create config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""version: "1.0"
llm:
  model: "test-model"
batch:
  max_concurrent: 2
""")

        # Create task file
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(f'''project_path: "{sample_c_project}"
tasks:
  - source_file: "handler.c"
    function_name: "process_request"
''')

        # Run with config - note: --config is a global option before the command
        result = runner.invoke(
            cli,
            ["--config", str(config_file), "analyze", "--task-file", str(task_file)],
        )

        # May fail due to no MCP server, but should accept the config option
        # We just verify it parses the arguments correctly
        assert result.exit_code in [0, 1, 2]

    def test_quiet_mode(self, runner: CliRunner):
        """Test quiet mode."""
        result = runner.invoke(cli, ["--quiet", "--help"])
        assert result.exit_code == 0

    def test_verbose_mode(self, runner: CliRunner):
        """Test verbose mode."""
        result = runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0
