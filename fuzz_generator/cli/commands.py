"""CLI commands for fuzz_generator."""

from pathlib import Path

import click

from fuzz_generator import __version__
from fuzz_generator.cli.validators import (
    validate_config_file,
    validate_project_path,
    validate_task_file,
)

# Context settings for better help formatting
CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "max_content_width": 120,
}


class AliasedGroup(click.Group):
    """Click group with command aliases support."""

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        # Support common aliases
        aliases = {
            "a": "analyze",
            "p": "parse",
            "r": "results",
            "c": "clean",
            "t": "tools",
        }
        cmd_name = aliases.get(cmd_name, cmd_name)
        return super().get_command(ctx, cmd_name)


@click.group(cls=AliasedGroup, context_settings=CONTEXT_SETTINGS)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to configuration file",
    callback=validate_config_file,
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output",
)
@click.option(
    "--work-dir",
    "-w",
    type=click.Path(path_type=Path),
    default=".fuzz_generator",
    help="Working directory for intermediate results",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress non-essential output",
)
@click.version_option(version=__version__, prog_name="fuzz_generator")
@click.pass_context
def cli(
    ctx: click.Context,
    config: Path | None,
    verbose: bool,
    work_dir: Path,
    quiet: bool,
) -> None:
    """Fuzz Generator - AI Agent-based Fuzz Test Data Modeling Tool

    Generate fuzz test data models from source code functions using
    LLM-powered analysis and Joern static analysis.

    \b
    Examples:
        # Analyze a single function
        fuzz-generator analyze -p ./src -f handler.c -fn process_request

        # Batch analyze from task file
        fuzz-generator analyze -p ./src -t tasks.yaml

        # Parse project first
        fuzz-generator parse -p ./src

    For more information, see: https://github.com/your-org/fuzz_generator
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Store global options
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose
    ctx.obj["work_dir"] = work_dir
    ctx.obj["quiet"] = quiet

    # Setup logging based on verbosity
    if not quiet:
        from fuzz_generator.utils.logger import setup_logger

        log_level = "DEBUG" if verbose else "INFO"
        log_file = work_dir / "logs" / "fuzz_generator.log" if not quiet else None
        setup_logger(log_level=log_level, log_file=log_file)


@cli.command()
@click.option(
    "--project-path",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to source code project",
    callback=validate_project_path,
)
@click.option(
    "--source-file",
    "-f",
    type=str,
    help="Source file to analyze (relative to project path, for single function mode)",
)
@click.option(
    "--function",
    "-fn",
    type=str,
    help="Function name to analyze (for single function mode)",
)
@click.option(
    "--task-file",
    "-t",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Batch task file (YAML/JSON format)",
    callback=validate_task_file,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory or file path",
)
@click.option(
    "--knowledge-file",
    "-k",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Custom background knowledge file",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from last interrupted batch",
)
@click.option(
    "--output-name",
    "-n",
    type=str,
    help="Custom name for generated DataModel (single function mode)",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    project_path: Path,
    source_file: str | None,
    function: str | None,
    task_file: Path | None,
    output: Path | None,
    knowledge_file: Path | None,
    resume: bool,
    output_name: str | None,
) -> None:
    """Analyze functions and generate DataModel definitions.

    \b
    Supports two modes:
    1. Single function mode: use -f and -fn options
    2. Batch mode: use -t option with task file

    \b
    Examples:
        # Single function analysis
        fuzz-generator analyze -p ./src -f handler.c -fn process_request -o output.xml

        # Batch analysis
        fuzz-generator analyze -p ./src -t tasks.yaml -o ./output/

        # Resume interrupted batch
        fuzz-generator analyze -p ./src -t tasks.yaml --resume
    """
    # Validate mode options
    single_mode = source_file is not None and function is not None
    batch_mode = task_file is not None

    if not single_mode and not batch_mode:
        raise click.UsageError(
            "Please specify either single function mode (-f, -fn) or batch mode (-t task_file)"
        )

    if single_mode and batch_mode:
        raise click.UsageError(
            "Cannot use both single function mode and batch mode. Please choose one."
        )

    # Run analysis
    if not ctx.obj.get("quiet"):
        if single_mode:
            click.echo(f"Analyzing function '{function}' in {source_file}...")
        else:
            click.echo(f"Running batch analysis from {task_file}...")

    # TODO: Implement actual analysis logic in Phase 3
    click.echo(click.style("Analysis not yet implemented (Phase 3)", fg="yellow"))


@cli.command()
@click.option(
    "--project-path",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to source code project",
    callback=validate_project_path,
)
@click.option(
    "--project-name",
    "-n",
    type=str,
    help="Custom project name (default: directory name)",
)
@click.option(
    "--language",
    "-l",
    type=click.Choice(["c", "cpp", "java", "auto"]),
    default="auto",
    help="Source language (auto-detect by default)",
)
@click.pass_context
def parse(
    ctx: click.Context,
    project_path: Path,
    project_name: str | None,
    language: str,
) -> None:
    """Parse project and generate Code Property Graph (CPG).

    This is a pre-processing step that prepares the project for analysis.
    The CPG is cached for subsequent analysis operations.

    \b
    Examples:
        fuzz-generator parse -p ./src
        fuzz-generator parse -p ./src -n my_project -l c
    """
    if not ctx.obj.get("quiet"):
        click.echo(f"Parsing project at {project_path}...")

    # TODO: Implement actual parsing logic in Phase 2
    click.echo(click.style("Parsing not yet implemented (Phase 2)", fg="yellow"))


@cli.command()
@click.option(
    "--task-id",
    "-t",
    type=str,
    help="View results for specific task",
)
@click.option(
    "--list",
    "-l",
    "list_all",
    is_flag=True,
    default=False,
    help="List all available results",
)
@click.option(
    "--batch-id",
    "-b",
    type=str,
    help="View results for specific batch",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.pass_context
def results(
    ctx: click.Context,
    task_id: str | None,
    list_all: bool,
    batch_id: str | None,
    format: str,
) -> None:
    """View analysis results and intermediate data.

    \b
    Examples:
        fuzz-generator results --list
        fuzz-generator results -t task_001
        fuzz-generator results -b batch_001 -f json
    """
    ctx.obj.get("work_dir", Path(".fuzz_generator"))

    if list_all:
        click.echo("Available results:")
        # TODO: Implement results listing
        click.echo(click.style("Results listing not yet implemented", fg="yellow"))
    elif task_id:
        click.echo(f"Results for task: {task_id}")
        # TODO: Implement task results display
    elif batch_id:
        click.echo(f"Results for batch: {batch_id}")
        # TODO: Implement batch results display
    else:
        click.echo("Use --list to view all results or specify --task-id or --batch-id")


@cli.command()
@click.option(
    "--all",
    "-a",
    "clear_all",
    is_flag=True,
    default=False,
    help="Clear all cache and intermediate data",
)
@click.option(
    "--task-id",
    "-t",
    type=str,
    help="Clear data for specific task",
)
@click.option(
    "--cache-only",
    is_flag=True,
    default=False,
    help="Only clear analysis cache",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt",
)
@click.pass_context
def clean(
    ctx: click.Context,
    clear_all: bool,
    task_id: str | None,
    cache_only: bool,
    force: bool,
) -> None:
    """Clean cache and intermediate results.


    Examples:
        fuzz-generator clean --all
        fuzz-generator clean -t task_001
        fuzz-generator clean --cache-only
    """
    ctx.obj.get("work_dir", Path(".fuzz_generator"))

    if not clear_all and not task_id and not cache_only:
        raise click.UsageError("Please specify what to clean: --all, --task-id, or --cache-only")

    if clear_all:
        if not force:
            if not click.confirm("This will delete all cached data. Continue?"):
                click.echo("Aborted.")
                return

        click.echo("Clearing all data...")
        # TODO: Implement actual cleanup

    elif task_id:
        click.echo(f"Clearing data for task: {task_id}")
        # TODO: Implement task-specific cleanup

    elif cache_only:
        click.echo("Clearing cache...")
        # TODO: Implement cache cleanup

    click.echo(click.style("Cleanup completed", fg="green"))


@cli.command(name="tools")
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    default=False,
    help="Show detailed tool information",
)
@click.pass_context
def list_tools(ctx: click.Context, detailed: bool) -> None:
    """List available MCP tools from Joern server.

    \b
    Examples:
        fuzz-generator tools
        fuzz-generator tools --detailed
    """
    click.echo("Available MCP Tools:")
    click.echo("-" * 40)

    # Static list for now, will be dynamic in Phase 2
    tools = [
        ("parse_project", "Parse project and generate CPG"),
        ("list_projects", "List parsed projects"),
        ("switch_project", "Switch active project"),
        ("get_function_code", "Get function source code"),
        ("list_functions", "List project functions"),
        ("search_code", "Search code patterns"),
        ("get_callers", "Get function callers"),
        ("get_callees", "Get called functions"),
        ("track_dataflow", "Track data flow"),
        ("analyze_variable_flow", "Analyze variable flow"),
        ("get_control_flow_graph", "Get CFG"),
        ("find_vulnerabilities", "Find vulnerabilities"),
    ]

    for name, desc in tools:
        if detailed:
            click.echo(f"\n{click.style(name, fg='cyan', bold=True)}")
            click.echo(f"  Description: {desc}")
            click.echo("  Parameters: (not yet available)")
        else:
            click.echo(f"  {click.style(name, fg='cyan'):30} {desc}")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current status and configuration.

    Displays information about:
    - Current configuration
    - MCP server connection status
    - Running tasks
    - Cache statistics
    """
    work_dir = ctx.obj.get("work_dir", Path(".fuzz_generator"))
    # work_dir will be used when implementing results display
    # work_dir = ctx.obj.get("work_dir", Path(".fuzz_generator"))
    config_path = ctx.obj.get("config_path")

    click.echo(click.style("Fuzz Generator Status", fg="cyan", bold=True))
    click.echo("=" * 40)

    click.echo(f"\nVersion: {__version__}")
    click.echo(f"Work Directory: {work_dir}")
    click.echo(f"Config File: {config_path or 'Using defaults'}")

    # TODO: Add actual status checks
    click.echo("\nMCP Server: " + click.style("Not connected", fg="yellow"))
    click.echo("Running Tasks: 0")
    click.echo("Cached Results: N/A")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
