"""Common validators for fuzz_generator."""

import re
from pathlib import Path
from typing import Any

from fuzz_generator.utils.logger import get_logger

logger = get_logger(__name__)


def validate_function_name(name: str) -> bool:
    """Validate a function name.

    Args:
        name: Function name to validate

    Returns:
        True if valid
    """
    if not name:
        return False

    # C/C++ identifier rules: starts with letter or underscore,
    # contains letters, digits, or underscores
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    return bool(re.match(pattern, name))


def validate_source_file(file_path: str, project_path: Path) -> bool:
    """Validate a source file path.

    Args:
        file_path: Source file path (relative to project)
        project_path: Project root path

    Returns:
        True if valid
    """
    if not file_path:
        return False

    # Check file exists
    full_path = project_path / file_path
    if not full_path.exists():
        logger.warning(f"Source file not found: {full_path}")
        return False

    # Check extension
    valid_extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"}
    if full_path.suffix.lower() not in valid_extensions:
        logger.warning(f"Unsupported source file type: {full_path.suffix}")
        return False

    return True


def validate_project_structure(project_path: Path) -> dict[str, Any]:
    """Validate project structure and gather info.

    Args:
        project_path: Project root path

    Returns:
        Dictionary with validation results and info
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {
            "c_files": 0,
            "cpp_files": 0,
            "header_files": 0,
            "total_files": 0,
        },
    }

    if not project_path.exists():
        result["valid"] = False
        result["errors"].append(f"Project path does not exist: {project_path}")
        return result

    if not project_path.is_dir():
        result["valid"] = False
        result["errors"].append(f"Project path is not a directory: {project_path}")
        return result

    # Count files
    c_extensions = {".c"}
    cpp_extensions = {".cpp", ".cc", ".cxx"}
    header_extensions = {".h", ".hpp", ".hxx"}

    for path in project_path.rglob("*"):
        if path.is_file():
            result["info"]["total_files"] += 1
            suffix = path.suffix.lower()

            if suffix in c_extensions:
                result["info"]["c_files"] += 1
            elif suffix in cpp_extensions:
                result["info"]["cpp_files"] += 1
            elif suffix in header_extensions:
                result["info"]["header_files"] += 1

    # Validate there are source files
    source_count = result["info"]["c_files"] + result["info"]["cpp_files"]
    if source_count == 0:
        result["valid"] = False
        result["errors"].append("No C/C++ source files found in project")

    return result


def validate_xml_output_path(output_path: Path) -> tuple[bool, str | None]:
    """Validate an XML output path.

    Args:
        output_path: Output file path

    Returns:
        Tuple of (is_valid, error_message)
    """
    # If it's a directory, that's fine
    if output_path.is_dir():
        return True, None

    # Check if parent directory exists or can be created
    parent = output_path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
            return True, None
        except Exception as e:
            return False, f"Cannot create output directory: {e}"

    # Check extension for file output
    if output_path.suffix.lower() not in {".xml", ""}:
        return False, "Output file should have .xml extension"

    return True, None


def validate_batch_task(task: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a batch task definition.

    Args:
        task: Task dictionary

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Required fields
    if "source_file" not in task:
        errors.append("Missing required field: source_file")
    if "function_name" not in task:
        errors.append("Missing required field: function_name")

    # Validate function name format
    if "function_name" in task and not validate_function_name(task["function_name"]):
        errors.append(f"Invalid function name: {task['function_name']}")

    # Validate output_name if provided
    if "output_name" in task and task["output_name"]:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", task["output_name"]):
            errors.append(f"Invalid output_name: {task['output_name']}")

    # Validate priority if provided
    if "priority" in task:
        if not isinstance(task["priority"], int) or task["priority"] < 0:
            errors.append("Priority must be a non-negative integer")

    # Validate depends_on if provided
    if "depends_on" in task:
        if not isinstance(task["depends_on"], list):
            errors.append("depends_on must be a list")
        else:
            for dep in task["depends_on"]:
                if not isinstance(dep, str):
                    errors.append(f"Invalid dependency: {dep}")

    return len(errors) == 0, errors


def sanitize_datamodel_name(name: str) -> str:
    """Sanitize a name for use as DataModel name.

    Args:
        name: Raw name

    Returns:
        Sanitized name valid for XML
    """
    if not name:
        return "UnnamedModel"

    # Remove invalid characters
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # Ensure starts with letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized

    # Capitalize first letter (convention)
    if sanitized:
        sanitized = sanitized[0].upper() + sanitized[1:]

    return sanitized or "UnnamedModel"
