"""CLI commands for fuzz_generator."""

import asyncio
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


def _get_settings(ctx: click.Context):
    """Get settings from context or load defaults."""
    from fuzz_generator.config import load_config

    config_path = ctx.obj.get("config_path")
    return load_config(config_path)


def _run_async(coro):
    """Run an async coroutine."""
    return asyncio.run(coro)


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
    from fuzz_generator.cli.runner import AnalysisRunner

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

    # Get settings and create runner
    settings = _get_settings(ctx)
    work_dir = ctx.obj.get("work_dir", Path(".fuzz_generator"))
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    runner = AnalysisRunner(
        settings=settings,
        work_dir=work_dir,
        verbose=verbose,
        quiet=quiet,
    )

    # Run analysis
    if single_mode:
        result = _run_async(
            runner.analyze_single_function(
                project_path=project_path,
                source_file=source_file,
                function_name=function,
                output_path=output,
                output_name=output_name,
                knowledge_file=knowledge_file,
            )
        )

        # Exit with error code if result is None (exception) or analysis failed
        if result is None or not result.success:
            ctx.exit(1)

    else:  # batch_mode
        # Determine output directory
        output_dir = output if output else Path("output")

        result = _run_async(
            runner.analyze_batch(
                project_path=project_path,
                task_file=task_file,
                output_dir=output_dir,
                knowledge_file=knowledge_file,
                resume=resume,
            )
        )

        if result.get("error"):
            ctx.exit(1)


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
    from fuzz_generator.cli.runner import AnalysisRunner

    settings = _get_settings(ctx)
    work_dir = ctx.obj.get("work_dir", Path(".fuzz_generator"))
    quiet = ctx.obj.get("quiet", False)

    runner = AnalysisRunner(
        settings=settings,
        work_dir=work_dir,
        verbose=ctx.obj.get("verbose", False),
        quiet=quiet,
    )

    success = _run_async(
        runner.parse_project(
            project_path=project_path,
            project_name=project_name,
            language=language,
        )
    )

    if not success:
        ctx.exit(1)


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
    import json as json_lib

    import yaml

    from fuzz_generator.cli.runner import ResultsViewer
    from fuzz_generator.storage import JsonStorage

    work_dir = ctx.obj.get("work_dir", Path(".fuzz_generator"))
    storage = JsonStorage(base_dir=work_dir)
    viewer = ResultsViewer(storage=storage, work_dir=work_dir)

    if list_all:
        results_list = _run_async(viewer.list_results())

        if not results_list:
            click.echo("No results found.")
            return

        if format == "json":
            click.echo(json_lib.dumps(results_list, indent=2))
        elif format == "yaml":
            click.echo(yaml.dump(results_list, default_flow_style=False))
        else:
            click.echo("Available results:")
            click.echo("-" * 60)
            for r in results_list:
                status = (
                    click.style("✓", fg="green") if r.get("success") else click.style("✗", fg="red")
                )
                click.echo(f"  {status} {r.get('task_id', 'unknown')}")

    elif task_id:
        result = _run_async(viewer.get_result(task_id))

        if not result:
            click.echo(f"No result found for task: {task_id}")
            ctx.exit(1)

        if format == "json":
            click.echo(json_lib.dumps(result, indent=2))
        elif format == "yaml":
            click.echo(yaml.dump(result, default_flow_style=False))
        else:
            click.echo(f"Result for task: {task_id}")
            click.echo("-" * 40)
            click.echo(f"  Success: {result.get('success', False)}")
            if result.get("xml_content"):
                click.echo("  XML Content:")
                click.echo(result.get("xml_content"))

    elif batch_id:
        results_list = _run_async(viewer.get_batch_results(batch_id))

        if not results_list:
            click.echo(f"No results found for batch: {batch_id}")
            return

        if format == "json":
            click.echo(json_lib.dumps(results_list, indent=2))
        elif format == "yaml":
            click.echo(yaml.dump(results_list, default_flow_style=False))
        else:
            click.echo(f"Results for batch: {batch_id}")
            click.echo("-" * 40)
            for r in results_list:
                status = (
                    click.style("✓", fg="green") if r.get("success") else click.style("✗", fg="red")
                )
                click.echo(f"  {status} {r.get('task_id', 'unknown')}")

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

    \b
    Examples:
        fuzz-generator clean --all
        fuzz-generator clean -t task_001
        fuzz-generator clean --cache-only
    """
    from fuzz_generator.cli.runner import CacheCleaner
    from fuzz_generator.storage import JsonStorage

    work_dir = ctx.obj.get("work_dir", Path(".fuzz_generator"))
    storage = JsonStorage(base_dir=work_dir)
    cleaner = CacheCleaner(storage=storage, work_dir=work_dir)

    if not clear_all and not task_id and not cache_only:
        raise click.UsageError("Please specify what to clean: --all, --task-id, or --cache-only")

    if clear_all:
        if not force:
            if not click.confirm("This will delete all cached data. Continue?"):
                click.echo("Aborted.")
                return

        count = _run_async(cleaner.clear_all())
        click.echo(click.style(f"Cleared {count} items", fg="green"))

    elif task_id:
        success = _run_async(cleaner.clear_task(task_id))
        if success:
            click.echo(click.style(f"Cleared data for task: {task_id}", fg="green"))
        else:
            click.echo(f"No data found for task: {task_id}")

    elif cache_only:
        count = _run_async(cleaner.clear_cache_only())
        click.echo(click.style(f"Cleared {count} cache items", fg="green"))


@cli.command(name="tools")
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    default=False,
    help="Show detailed tool information",
)
@click.option(
    "--test",
    is_flag=True,
    default=False,
    help="Test MCP server connection",
)
@click.pass_context
def list_tools(ctx: click.Context, detailed: bool, test: bool) -> None:
    """List available MCP tools from Joern server.

    \b
    Examples:
        fuzz-generator tools
        fuzz-generator tools --detailed
        fuzz-generator tools --test
    """
    if test:
        # Test MCP connection
        from fuzz_generator.tools.mcp_client import MCPClientConfig, MCPHttpClient

        settings = _get_settings(ctx)
        config = MCPClientConfig(
            url=settings.mcp_server.url,
            timeout=settings.mcp_server.timeout,
        )

        click.echo(f"Testing connection to: {config.url}")

        async def test_connection():
            try:
                async with MCPHttpClient(config) as client:
                    tools = await client.list_tools()
                    return tools
            except Exception as e:
                return str(e)

        result = _run_async(test_connection())

        if isinstance(result, str):
            click.echo(click.style(f"✗ Connection failed: {result}", fg="red"))
            ctx.exit(1)
        else:
            click.echo(click.style("✓ Connection successful", fg="green"))
            click.echo(f"  Available tools: {len(result)}")
            return

    click.echo("Available MCP Tools:")
    click.echo("-" * 40)

    # Tool list with descriptions
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
    from fuzz_generator.storage import JsonStorage
    from fuzz_generator.tools.mcp_client import MCPClientConfig, MCPHttpClient

    work_dir = ctx.obj.get("work_dir", Path(".fuzz_generator"))
    config_path = ctx.obj.get("config_path")

    click.echo(click.style("Fuzz Generator Status", fg="cyan", bold=True))
    click.echo("=" * 40)

    click.echo(f"\nVersion: {__version__}")
    click.echo(f"Work Directory: {work_dir}")
    click.echo(f"Config File: {config_path or 'Using defaults'}")

    # Load settings
    settings = _get_settings(ctx)
    click.echo(f"\nLLM Model: {settings.llm.model}")
    click.echo(f"LLM URL: {settings.llm.base_url}")
    click.echo(f"MCP Server: {settings.mcp_server.url}")

    # Test MCP connection
    click.echo("\nMCP Server Status:")
    config = MCPClientConfig(
        url=settings.mcp_server.url,
        timeout=5,  # Short timeout for status check
    )

    async def check_mcp():
        try:
            async with MCPHttpClient(config) as client:
                await client.list_tools()
                return True
        except Exception:
            return False

    mcp_ok = _run_async(check_mcp())
    if mcp_ok:
        click.echo(click.style("  ✓ Connected", fg="green"))
    else:
        click.echo(click.style("  ✗ Not available", fg="red"))

    # Storage stats
    click.echo("\nStorage:")
    storage = JsonStorage(base_dir=work_dir)

    async def get_storage_stats():
        categories = await storage.list_categories()
        stats = {}
        for cat in categories:
            keys = await storage.list_keys(cat)
            stats[cat] = len(keys)
        return stats

    stats = _run_async(get_storage_stats())
    if stats:
        for cat, count in stats.items():
            click.echo(f"  {cat}: {count} items")
    else:
        click.echo("  No cached data")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
