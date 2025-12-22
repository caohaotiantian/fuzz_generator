"""CLI runner module for executing analysis workflows.

This module provides the integration between CLI commands and the
AutoGen-based multi-agent analysis system.

Design Reference: docs/TECHNICAL_DESIGN.md Section 4.5 (CLI Design)
"""

from pathlib import Path
from typing import Any

import click

from fuzz_generator.batch import BatchStateManager, TaskParser
from fuzz_generator.config import Settings
from fuzz_generator.exceptions import MCPConnectionError
from fuzz_generator.generators import XMLValidator
from fuzz_generator.models import AnalysisTask, TaskResult
from fuzz_generator.storage import JsonStorage
from fuzz_generator.tools.mcp_client import MCPClientConfig, MCPHttpClient
from fuzz_generator.tools.project_tools import parse_project, switch_project
from fuzz_generator.utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisRunner:
    """Runner for analysis operations.

    Implements CLI integration as per docs/TECHNICAL_DESIGN.md Section 4.5
    """

    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        verbose: bool = False,
        quiet: bool = False,
    ):
        """Initialize analysis runner.

        Args:
            settings: Application settings
            work_dir: Working directory for intermediate results
            verbose: Enable verbose output
            quiet: Suppress non-essential output
        """
        self.settings = settings
        self.work_dir = work_dir
        self.verbose = verbose
        self.quiet = quiet

        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage
        self.storage = JsonStorage(base_dir=work_dir)

        # MCP client config
        self.mcp_config = MCPClientConfig(
            url=settings.mcp_server.url,
            timeout=settings.mcp_server.timeout,
            retry_count=settings.mcp_server.retry_count,
            retry_delay=settings.mcp_server.retry_delay,
        )

    async def _ensure_project_parsed(
        self,
        mcp_client: MCPHttpClient,
        project_path: Path,
        project_name: str,
        language: str = "auto",
        auto_parse: bool = True,
    ) -> tuple[bool, str]:
        """Ensure project is parsed, auto-parse if needed.

        Args:
            mcp_client: MCP client instance
            project_path: Path to source code project
            project_name: Project name
            language: Source language
            auto_parse: Whether to auto-parse if project not found

        Returns:
            (success, effective_project_name) tuple
        """
        # First, check available projects to verify if project actually exists
        from fuzz_generator.tools.project_tools import list_projects

        projects_result = await list_projects(mcp_client)
        project_exists = False
        available = []

        if projects_result.success and projects_result.projects:
            available = [p.name for p in projects_result.projects]
            project_exists = project_name in available

        # If project exists, switch to it and return
        if project_exists:
            switch_result = await switch_project(mcp_client, project_name)
            if switch_result.success:
                return (True, project_name)
            else:
                logger.warning(
                    f"Project {project_name} exists but failed to switch: {switch_result.error}"
                )

        # Project doesn't exist - decide whether to auto-parse
        if not auto_parse:
            if available:
                click.echo(
                    click.style(
                        f"Project '{project_name}' not found. Available: {available}",
                        fg="yellow",
                    )
                )
                click.echo("  Please specify correct project name with --project-name")
            else:
                click.echo(
                    click.style("No projects found. Please run 'parse' command first.", fg="red")
                )
            return (False, project_name)

        # Auto-parse the project
        if not self.quiet:
            if available:
                click.echo(
                    click.style(
                        f"Project '{project_name}' not found in {available}, auto-parsing...",
                        fg="yellow",
                    )
                )
            else:
                click.echo(
                    click.style(f"No projects found, auto-parsing '{project_name}'...", fg="yellow")
                )

        # Parse project (using the same client, ensure absolute path)
        absolute_path = project_path.resolve()
        lang_param = None if language == "auto" else language
        result = await parse_project(
            mcp_client,
            source_path=str(absolute_path),
            project_name=project_name,
            language=lang_param,
        )

        if result.success:
            if not self.quiet:
                click.echo(click.style("✓ Project parsed successfully", fg="green"))
            # Switch to newly parsed project
            await switch_project(mcp_client, project_name)
            return (True, project_name)
        else:
            click.echo(click.style(f"✗ Failed to parse project: {result.error}", fg="red"))
            return (False, project_name)

    async def parse_project(
        self,
        project_path: Path,
        project_name: str | None = None,
        language: str = "auto",
    ) -> bool:
        """Parse a project to generate CPG.

        Implements: fuzz-generator parse -p ./src

        Note: This is an optional pre-processing step. The analyze command
        will automatically parse projects if needed.

        Args:
            project_path: Path to source code project
            project_name: Custom project name
            language: Source language (auto-detect by default)

        Returns:
            True if successful
        """
        name = project_name or project_path.name

        if not self.quiet:
            click.echo(f"Parsing project: {name}")
            click.echo(f"  Path: {project_path}")

        try:
            async with MCPHttpClient(self.mcp_config) as client:
                # Only pass language if not "auto" (let MCP server auto-detect)
                lang_param = None if language == "auto" else language
                result = await parse_project(
                    client,
                    source_path=str(project_path),
                    project_name=name,
                    language=lang_param,
                )

                if result.success:
                    if not self.quiet:
                        click.echo(click.style("✓ Project parsed successfully", fg="green"))
                        click.echo(f"  Project name: {result.project_name}")
                    return True
                else:
                    click.echo(click.style(f"✗ Failed to parse project: {result.error}", fg="red"))
                    return False

        except MCPConnectionError as e:
            click.echo(click.style(f"✗ MCP connection error: {e}", fg="red"))
            click.echo("  Make sure the Joern MCP server is running.")
            return False
        except Exception as e:
            click.echo(click.style(f"✗ Error: {e}", fg="red"))
            return False

    async def analyze_single_function(
        self,
        project_path: Path,
        source_file: str,
        function_name: str,
        output_path: Path | None = None,
        output_name: str | None = None,
        knowledge_file: Path | None = None,
        project_name: str | None = None,
    ) -> TaskResult | None:
        """Analyze a single function.

        Implements: fuzz-generator analyze -p ./src -f handler.c -fn process_request

        Design Reference: docs/TECHNICAL_DESIGN.md Section 4.5

        Args:
            project_path: Path to source code project
            source_file: Source file containing the function
            function_name: Name of the function to analyze
            output_path: Output file path
            output_name: Custom name for generated DataModel
            knowledge_file: Path to custom knowledge file
            project_name: Project name in MCP server (if different from directory name)

        Returns:
            TaskResult if successful, None otherwise
        """
        # Use provided project_name or fall back to directory name
        effective_project_name = project_name or project_path.name

        # Load custom knowledge
        custom_knowledge = ""
        if knowledge_file and knowledge_file.exists():
            custom_knowledge = knowledge_file.read_text()

        # Create task
        task = AnalysisTask(
            task_id=f"{effective_project_name}_{function_name}",
            source_file=source_file,
            function_name=function_name,
            output_name=output_name or f"{function_name}Model",
        )

        if not self.quiet:
            click.echo(f"Analyzing function: {function_name}")
            click.echo(f"  File: {source_file}")
            click.echo(f"  Project: {effective_project_name}")

        try:
            async with MCPHttpClient(self.mcp_config) as mcp_client:
                # Ensure project is parsed (auto-parse if needed)
                success, effective_project_name = await self._ensure_project_parsed(
                    mcp_client=mcp_client,
                    project_path=project_path,
                    project_name=effective_project_name,
                    language="auto",
                    auto_parse=True,
                )

                if not success:
                    return None

                # Run analysis with AutoGen agents (uses custom model client)
                from fuzz_generator.agents.autogen_agents import AnalysisWorkflowRunner

                async with AnalysisWorkflowRunner(
                    settings=self.settings,
                    mcp_client=mcp_client,
                    project_name=effective_project_name,
                    custom_knowledge=custom_knowledge,
                    storage_path=self.work_dir,
                    project_path=project_path,
                ) as runner:
                    result = await runner.run_analysis(task, verbose=self.verbose)

                if result.success:
                    if not self.quiet:
                        click.echo(click.style("✓ Analysis completed", fg="green"))

                    # Save output
                    if output_path and result.xml_content:
                        self._save_output(result.xml_content, output_path)
                        # Also save to results directory
                        await self._save_to_results(task.task_id, result)

                    return result
                else:
                    click.echo(click.style("✗ Analysis failed", fg="red"))
                    for error in result.errors:
                        click.echo(f"  - {error}")
                    return result

        except MCPConnectionError as e:
            click.echo(click.style(f"✗ MCP connection error: {e}", fg="red"))
            return None
        except Exception as e:
            logger.exception("Analysis error")
            click.echo(click.style(f"✗ Error: {e}", fg="red"))
            return None

    async def analyze_batch(
        self,
        project_path: Path,
        task_file: Path,
        output_dir: Path | None = None,
        knowledge_file: Path | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Run batch analysis.

        Implements: fuzz-generator analyze -p ./src -t tasks.yaml

        Design Reference: docs/TECHNICAL_DESIGN.md Section 4.6 (Batch Task Format)

        Args:
            project_path: Path to source code project
            task_file: Path to task file (YAML/JSON)
            output_dir: Output directory for results
            knowledge_file: Path to custom knowledge file
            resume: Resume from last checkpoint

        Returns:
            Summary of batch execution
        """
        project_name = project_path.name

        # Parse task file
        parser = TaskParser(base_path=task_file.parent)
        batch = parser.parse(str(task_file))

        # Override project path if provided
        if project_path:
            batch.project_path = str(project_path)

        if not self.quiet:
            click.echo(f"Running batch analysis: {batch.batch_id}")
            click.echo(f"  Tasks: {len(batch.tasks)}")

        # Load custom knowledge
        custom_knowledge = ""
        if knowledge_file and knowledge_file.exists():
            custom_knowledge = knowledge_file.read_text()

        # Initialize state manager for resume
        state_manager = BatchStateManager(storage=self.storage)

        # Check for resume
        if resume:
            existing_state = await state_manager.load_state(batch.batch_id)
            if existing_state:
                batch = await state_manager.get_resumable_batch(batch.batch_id, batch)
                if not self.quiet:
                    click.echo(f"  Resuming from checkpoint: {len(batch.tasks)} tasks remaining")
        else:
            # Create initial state
            await state_manager.create_state(batch)

        try:
            async with MCPHttpClient(self.mcp_config) as mcp_client:
                # Ensure project is parsed (auto-parse if needed)
                success, project_name = await self._ensure_project_parsed(
                    mcp_client=mcp_client,
                    project_path=project_path,
                    project_name=project_name,
                    language="auto",
                    auto_parse=True,
                )

                if not success:
                    return {"error": "Failed to initialize project", "completed": 0, "failed": 0}

                # Create workflow runner
                from fuzz_generator.agents.autogen_agents import AnalysisWorkflowRunner

                async def run_task(task: AnalysisTask) -> TaskResult:
                    async with AnalysisWorkflowRunner(
                        settings=self.settings,
                        mcp_client=mcp_client,
                        project_name=project_name,
                        custom_knowledge=custom_knowledge,
                        storage_path=self.work_dir,
                    ) as runner:
                        return await runner.run_analysis(task, verbose=self.verbose)

                # Run tasks
                results = []
                completed = 0
                failed = 0

                for i, task in enumerate(batch.tasks):
                    if not self.quiet:
                        click.echo(f"\n  [{i + 1}/{len(batch.tasks)}] {task.function_name}")

                    # Mark as running
                    await state_manager.mark_running(batch.batch_id, task.task_id)

                    result = await run_task(task)
                    results.append(result)

                    if result.success:
                        completed += 1
                        await state_manager.mark_completed(
                            batch.batch_id, task.task_id, {"xml": result.xml_content}
                        )

                        # Save output
                        if output_dir and result.xml_content:
                            output_file = (
                                output_dir / f"{task.output_name or task.function_name}.xml"
                            )
                            self._save_output(result.xml_content, output_file)

                        # Save to results
                        await self._save_to_results(task.task_id, result)
                    else:
                        failed += 1
                        await state_manager.mark_failed(
                            batch.batch_id,
                            task.task_id,
                            result.errors[0] if result.errors else "Unknown error",
                        )

                        if self.settings.batch.fail_fast:
                            click.echo(
                                click.style("  Stopping due to fail_fast setting", fg="yellow")
                            )
                            break

                # Summary
                summary = {
                    "batch_id": batch.batch_id,
                    "total": len(batch.tasks),
                    "completed": completed,
                    "failed": failed,
                    "results": results,
                }

                if not self.quiet:
                    click.echo(f"\n{'=' * 40}")
                    click.echo(f"Batch complete: {completed}/{len(batch.tasks)} succeeded")
                    if failed > 0:
                        click.echo(click.style(f"  Failed: {failed}", fg="red"))

                return summary

        except MCPConnectionError as e:
            click.echo(click.style(f"✗ MCP connection error: {e}", fg="red"))
            return {"error": str(e)}
        except Exception as e:
            logger.exception("Batch analysis error")
            click.echo(click.style(f"✗ Error: {e}", fg="red"))
            return {"error": str(e)}

    async def _save_to_results(self, task_id: str, result: TaskResult) -> None:
        """Save result to results storage.

        Design Reference: docs/TECHNICAL_DESIGN.md Section 3.2.8 (Storage Structure)

        Args:
            task_id: Task identifier
            result: Task result
        """
        # Save to storage
        await self.storage.save(
            "results",
            task_id,
            {
                "task_id": task_id,
                "success": result.success,
                "xml_content": result.xml_content,
                "errors": result.errors,
                "warnings": result.warnings,
                "created_at": result.created_at.isoformat()
                if hasattr(result, "created_at")
                else None,
            },
        )

        # Also save XML to output directory
        if result.xml_content:
            output_dir = self.work_dir / "results" / task_id / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "datamodel.xml").write_text(result.xml_content)

    def _save_output(self, xml_content: str, output_path: Path) -> None:
        """Save XML output to file.

        Args:
            xml_content: XML content to save
            output_path: Output file path
        """
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate XML
        validator = XMLValidator()
        validation = validator.validate(xml_content)

        if not validation.is_valid:
            logger.warning(f"Generated XML has validation issues: {validation.errors}")

        # Write file
        output_path.write_text(xml_content)

        if not self.quiet:
            click.echo(f"  Output: {output_path}")


class ResultsViewer:
    """Viewer for analysis results.

    Implements: fuzz-generator results commands
    """

    def __init__(self, storage: JsonStorage, work_dir: Path):
        """Initialize results viewer.

        Args:
            storage: Storage backend
            work_dir: Working directory
        """
        self.storage = storage
        self.work_dir = work_dir

    async def list_results(self, format: str = "table") -> list[dict]:
        """List all analysis results.

        Args:
            format: Output format (table, json, yaml)

        Returns:
            List of result summaries
        """
        results = []

        # List from storage
        keys = await self.storage.list_keys("results")

        for key in keys:
            data = await self.storage.load("results", key)
            if data:
                results.append(
                    {
                        "task_id": key,
                        "success": data.get("success", False),
                        "created_at": data.get("created_at"),
                    }
                )

        return results

    async def get_result(self, task_id: str) -> dict | None:
        """Get a specific result.

        Args:
            task_id: Task ID

        Returns:
            Result data or None
        """
        return await self.storage.load("results", task_id)

    async def get_batch_results(self, batch_id: str) -> list[dict]:
        """Get all results for a batch.

        Args:
            batch_id: Batch ID

        Returns:
            List of results
        """
        state_manager = BatchStateManager(storage=self.storage)
        state = await state_manager.load_state(batch_id)

        if not state:
            return []

        results = []
        for task_id in state.completed:
            result = await self.get_result(task_id)
            if result:
                results.append(result)

        return results


class CacheCleaner:
    """Cleaner for cache and intermediate data.

    Implements: fuzz-generator clean commands
    """

    def __init__(self, storage: JsonStorage, work_dir: Path):
        """Initialize cache cleaner.

        Args:
            storage: Storage backend
            work_dir: Working directory
        """
        self.storage = storage
        self.work_dir = work_dir

    async def clear_all(self) -> int:
        """Clear all cached data.

        Returns:
            Number of items cleared
        """
        count = 0

        # Clear all categories
        for category in await self.storage.list_categories():
            cleared = await self.storage.clear_category(category)
            count += cleared

        return count

    async def clear_task(self, task_id: str) -> bool:
        """Clear data for a specific task.

        Args:
            task_id: Task ID

        Returns:
            True if cleared
        """
        # Delete from storage
        deleted = await self.storage.delete("results", task_id)

        # Delete task directory
        task_dir = self.work_dir / "results" / task_id
        if task_dir.exists():
            import shutil

            shutil.rmtree(task_dir)
            deleted = True

        return deleted

    async def clear_cache_only(self) -> int:
        """Clear only cache data (not results).

        Returns:
            Number of items cleared
        """
        count = 0

        # Clear cache categories as per docs/TECHNICAL_DESIGN.md Section 4.7.1
        for category in ["functions", "dataflow", "callgraph", "analysis"]:
            try:
                cleared = await self.storage.clear_category(category)
                count += cleared
            except Exception:
                pass

        return count
