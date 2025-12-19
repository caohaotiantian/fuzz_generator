"""Input validators for CLI commands."""

from pathlib import Path

import click
import yaml


def validate_project_path(
    ctx: click.Context,
    param: click.Parameter,
    value: Path | None,
) -> Path | None:
    """Validate project path exists and contains source files.

    Args:
        ctx: Click context
        param: Click parameter
        value: Path value to validate

    Returns:
        Validated path or None

    Raises:
        click.BadParameter: If validation fails
    """
    if value is None:
        return None

    if not value.exists():
        raise click.BadParameter(f"Project path does not exist: {value}")

    if not value.is_dir():
        raise click.BadParameter(f"Project path is not a directory: {value}")

    # Check for common source file extensions
    source_extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".java"}
    has_sources = any(
        f.suffix.lower() in source_extensions for f in value.rglob("*") if f.is_file()
    )

    if not has_sources:
        click.echo(
            click.style(
                f"Warning: No common source files found in {value}",
                fg="yellow",
            ),
            err=True,
        )

    return value


def validate_config_file(
    ctx: click.Context,
    param: click.Parameter,
    value: Path | None,
) -> Path | None:
    """Validate configuration file.

    Args:
        ctx: Click context
        param: Click parameter
        value: Path value to validate

    Returns:
        Validated path or None

    Raises:
        click.BadParameter: If validation fails
    """
    if value is None:
        return None

    if not value.exists():
        raise click.BadParameter(f"Config file does not exist: {value}")

    if value.suffix.lower() not in {".yaml", ".yml"}:
        raise click.BadParameter(f"Config file must be YAML format: {value}")

    # Try to parse the config file
    try:
        with open(value, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise click.BadParameter(f"Config file is empty: {value}")
    except yaml.YAMLError as e:
        raise click.BadParameter(f"Invalid YAML in config file: {e}") from None

    return value


def validate_task_file(
    ctx: click.Context,
    param: click.Parameter,
    value: Path | None,
) -> Path | None:
    """Validate task file format.

    Args:
        ctx: Click context
        param: Click parameter
        value: Path value to validate

    Returns:
        Validated path or None

    Raises:
        click.BadParameter: If validation fails
    """
    if value is None:
        return None

    if not value.exists():
        raise click.BadParameter(f"Task file does not exist: {value}")

    suffix = value.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise click.BadParameter(f"Task file must be YAML or JSON format: {value}")

    # Try to parse and validate structure
    try:
        with open(value, encoding="utf-8") as f:
            if suffix == ".json":
                import json

                tasks = json.load(f)
            else:
                tasks = yaml.safe_load(f)

        if tasks is None:
            raise click.BadParameter(f"Task file is empty: {value}")

        # Check for required structure
        if not isinstance(tasks, dict):
            raise click.BadParameter(f"Task file must be a dictionary with 'tasks' key: {value}")

        if "tasks" not in tasks:
            raise click.BadParameter(f"Task file must contain 'tasks' key: {value}")

        if not isinstance(tasks["tasks"], list):
            raise click.BadParameter(f"'tasks' must be a list: {value}")

        # Validate each task
        for i, task in enumerate(tasks["tasks"]):
            if not isinstance(task, dict):
                raise click.BadParameter(f"Task {i} must be a dictionary: {value}")

            if "function_name" not in task:
                raise click.BadParameter(
                    f"Task {i} missing required field 'function_name': {value}"
                )

            if "source_file" not in task:
                raise click.BadParameter(f"Task {i} missing required field 'source_file': {value}")

    except (yaml.YAMLError, Exception) as e:
        if isinstance(e, click.BadParameter):
            raise
        raise click.BadParameter(f"Failed to parse task file: {e}") from None

    return value


def validate_output_path(
    ctx: click.Context,
    param: click.Parameter,
    value: Path | None,
) -> Path | None:
    """Validate output path.

    Args:
        ctx: Click context
        param: Click parameter
        value: Path value to validate

    Returns:
        Validated path or None

    Raises:
        click.BadParameter: If validation fails
    """
    if value is None:
        return None

    # If path has .xml extension, treat as file
    if value.suffix.lower() == ".xml":
        # Ensure parent directory exists or can be created
        parent = value.parent
        if parent and not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise click.BadParameter(f"Cannot create output directory: {e}") from None
    else:
        # Treat as directory
        if not value.exists():
            try:
                value.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise click.BadParameter(f"Cannot create output directory: {e}") from None

    return value


def validate_function_name(name: str) -> bool:
    """Validate function name format.

    Args:
        name: Function name to validate

    Returns:
        True if valid
    """
    if not name:
        return False

    # Basic C/C++ function name validation
    # Should start with letter or underscore
    if not (name[0].isalpha() or name[0] == "_"):
        return False

    # Rest should be alphanumeric or underscore
    return all(c.isalnum() or c == "_" for c in name)


def validate_source_file(
    project_path: Path,
    source_file: str,
) -> Path:
    """Validate source file exists within project.

    Args:
        project_path: Project root path
        source_file: Source file relative path

    Returns:
        Absolute path to source file

    Raises:
        click.BadParameter: If file doesn't exist
    """
    full_path = project_path / source_file

    if not full_path.exists():
        raise click.BadParameter(f"Source file not found: {source_file} (in {project_path})")

    if not full_path.is_file():
        raise click.BadParameter(f"Source path is not a file: {source_file}")

    return full_path
