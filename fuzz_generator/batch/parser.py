"""Task file parser for batch processing.

Supports parsing YAML and JSON task files into BatchTask objects.
"""

import json
import uuid
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from fuzz_generator.models import AnalysisTask, BatchTask


class TaskParseError(Exception):
    """Exception raised when task file parsing fails."""

    def __init__(self, message: str, file_path: str | None = None):
        self.file_path = file_path
        super().__init__(f"{message}" + (f" (file: {file_path})" if file_path else ""))


class TaskDefinition(BaseModel):
    """Single task definition from task file."""

    source_file: str = Field(description="Source file path (relative to project)")
    function_name: str = Field(description="Function name to analyze")
    output_name: str | None = Field(
        default=None,
        description="Custom name for generated DataModel",
    )
    priority: int = Field(
        default=0,
        description="Task priority (higher = more urgent)",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Task IDs this task depends on",
    )


class BatchDefinition(BaseModel):
    """Batch task file definition."""

    project_path: str = Field(description="Project source path")
    description: str | None = Field(
        default=None,
        description="Batch description",
    )
    tasks: list[TaskDefinition] = Field(
        description="List of tasks to execute",
    )
    config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration overrides for this batch",
    )


class TaskParser:
    """Parser for YAML and JSON task files."""

    def __init__(self, base_path: Path | str | None = None):
        """Initialize parser.

        Args:
            base_path: Base path for resolving relative paths in task files.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def parse(self, file_path: str | Path) -> BatchTask:
        """Parse a task file and return a BatchTask.

        Args:
            file_path: Path to the task file (YAML or JSON).

        Returns:
            BatchTask object containing all tasks.

        Raises:
            TaskParseError: If file parsing or validation fails.
            FileNotFoundError: If file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {path}")

        # Determine format and parse
        content = self._read_file(path)
        data = self._parse_content(content, path)

        # Validate and convert
        return self._create_batch_task(data, str(path))

    def parse_string(
        self,
        content: str,
        format_type: str = "yaml",
        file_path: str | None = None,
    ) -> BatchTask:
        """Parse task content from a string.

        Args:
            content: Task file content as string.
            format_type: Content format ("yaml" or "json").
            file_path: Optional file path for error messages.

        Returns:
            BatchTask object containing all tasks.

        Raises:
            TaskParseError: If parsing or validation fails.
        """
        data = self._parse_string(content, format_type, file_path)
        return self._create_batch_task(data, file_path)

    def _read_file(self, path: Path) -> str:
        """Read file content."""
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            raise TaskParseError(f"Failed to read file: {e}", str(path)) from None

    def _parse_content(self, content: str, path: Path) -> dict[str, Any]:
        """Parse file content based on file extension."""
        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            return self._parse_string(content, "yaml", str(path))
        elif suffix == ".json":
            return self._parse_string(content, "json", str(path))
        else:
            # Try YAML first, then JSON
            try:
                return self._parse_string(content, "yaml", str(path))
            except TaskParseError:
                return self._parse_string(content, "json", str(path))

    def _parse_string(
        self,
        content: str,
        format_type: str,
        file_path: str | None,
    ) -> dict[str, Any]:
        """Parse string content."""
        try:
            if format_type == "yaml":
                data = yaml.safe_load(content)
            elif format_type == "json":
                data = json.loads(content)
            else:
                raise TaskParseError(f"Unsupported format: {format_type}", file_path)

            if not isinstance(data, dict):
                raise TaskParseError("Task file must contain a mapping/object", file_path)

            return data
        except yaml.YAMLError as e:
            raise TaskParseError(f"Invalid YAML: {e}", file_path) from None
        except json.JSONDecodeError as e:
            raise TaskParseError(f"Invalid JSON: {e}", file_path) from None

    def _create_batch_task(
        self,
        data: dict[str, Any],
        file_path: str | None,
    ) -> BatchTask:
        """Create BatchTask from parsed data."""
        try:
            # Validate using Pydantic model
            batch_def = BatchDefinition(**data)
        except ValidationError as e:
            # Extract meaningful error message
            errors = []
            for err in e.errors():
                loc = ".".join(str(loc) for loc in err["loc"])
                errors.append(f"{loc}: {err['msg']}")
            raise TaskParseError(f"Validation failed: {'; '.join(errors)}", file_path) from None

        # Generate batch ID
        batch_id = self._generate_batch_id()

        # Resolve project path
        project_path = self._resolve_path(batch_def.project_path)

        # Convert task definitions to AnalysisTask objects
        tasks = self._create_tasks(batch_def.tasks, batch_id)

        return BatchTask(
            batch_id=batch_id,
            project_path=str(project_path),
            description=batch_def.description,
            tasks=tasks,
            config_overrides=batch_def.config_overrides,
        )

    def _create_tasks(
        self,
        task_defs: list[TaskDefinition],
        batch_id: str,
    ) -> list[AnalysisTask]:
        """Create AnalysisTask list from task definitions."""
        tasks = []
        for i, task_def in enumerate(task_defs):
            task_id = f"{batch_id}_task_{i:03d}"

            # Map depends_on to actual task IDs if they are indices
            depends_on = []
            for dep in task_def.depends_on:
                if dep.isdigit():
                    dep_idx = int(dep)
                    if 0 <= dep_idx < i:
                        depends_on.append(f"{batch_id}_task_{dep_idx:03d}")
                else:
                    depends_on.append(dep)

            task = AnalysisTask(
                task_id=task_id,
                source_file=task_def.source_file,
                function_name=task_def.function_name,
                output_name=task_def.output_name,
                priority=task_def.priority,
                depends_on=depends_on,
            )
            tasks.append(task)

        return tasks

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path (absolute or relative to base_path)."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self.base_path / path

    def _generate_batch_id(self) -> str:
        """Generate a unique batch ID."""
        return f"batch_{uuid.uuid4().hex[:8]}"

    def validate_batch(self, batch: BatchTask) -> list[str]:
        """Validate a BatchTask for common issues.

        Args:
            batch: BatchTask to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Check project path exists
        project_path = Path(batch.project_path)
        if not project_path.exists():
            errors.append(f"Project path does not exist: {project_path}")
        elif not project_path.is_dir():
            errors.append(f"Project path is not a directory: {project_path}")

        # Check for duplicate task IDs
        task_ids = [t.task_id for t in batch.tasks]
        if len(task_ids) != len(set(task_ids)):
            errors.append("Duplicate task IDs found")

        # Check dependencies
        for task in batch.tasks:
            for dep in task.depends_on:
                if dep not in task_ids:
                    errors.append(f"Task {task.task_id} depends on non-existent task: {dep}")

        # Check for circular dependencies
        if self._has_circular_dependencies(batch.tasks):
            errors.append("Circular dependencies detected")

        return errors

    def _has_circular_dependencies(self, tasks: list[AnalysisTask]) -> bool:
        """Check for circular dependencies using DFS."""
        task_map = {t.task_id: t for t in tasks}

        def dfs(task_id: str, visited: set[str], path: set[str]) -> bool:
            if task_id in path:
                return True
            if task_id in visited:
                return False

            visited.add(task_id)
            path.add(task_id)

            task = task_map.get(task_id)
            if task:
                for dep in task.depends_on:
                    if dfs(dep, visited, path):
                        return True

            path.remove(task_id)
            return False

        visited: set[str] = set()
        for task in tasks:
            if dfs(task.task_id, visited, set()):
                return True

        return False
