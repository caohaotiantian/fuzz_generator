"""Tests for batch task parser."""

import json
from pathlib import Path

import pytest

from fuzz_generator.batch.parser import TaskParseError, TaskParser
from fuzz_generator.models import BatchTask


class TestTaskParser:
    """Tests for TaskParser class."""

    def test_parser_creation(self):
        """Test parser can be created."""
        parser = TaskParser()
        assert parser.base_path == Path.cwd()

    def test_parser_with_base_path(self, tmp_path: Path):
        """Test parser with custom base path."""
        parser = TaskParser(base_path=tmp_path)
        assert parser.base_path == tmp_path

    def test_parse_yaml(self, tmp_path: Path):
        """Test parsing YAML task file."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            """
project_path: "/path/to/source"
tasks:
  - source_file: "main.c"
    function_name: "process"
    output_name: "ProcessModel"
  - source_file: "handler.c"
    function_name: "handle"
        """
        )

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        assert batch.project_path == "/path/to/source"
        assert len(batch.tasks) == 2
        assert batch.tasks[0].source_file == "main.c"
        assert batch.tasks[0].function_name == "process"
        assert batch.tasks[0].output_name == "ProcessModel"
        assert batch.tasks[1].source_file == "handler.c"
        assert batch.tasks[1].function_name == "handle"
        assert batch.tasks[1].output_name is None

    def test_parse_json(self, tmp_path: Path):
        """Test parsing JSON task file."""
        task_file = tmp_path / "tasks.json"
        task_file.write_text(
            json.dumps(
                {
                    "project_path": "/path/to/source",
                    "tasks": [
                        {"source_file": "main.c", "function_name": "process"},
                    ],
                }
            )
        )

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        assert len(batch.tasks) == 1
        assert batch.tasks[0].function_name == "process"

    def test_parse_yml_extension(self, tmp_path: Path):
        """Test parsing .yml file."""
        task_file = tmp_path / "tasks.yml"
        task_file.write_text(
            """
project_path: "/path/to/source"
tasks:
  - source_file: "main.c"
    function_name: "process"
        """
        )

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        assert len(batch.tasks) == 1

    def test_parse_invalid_yaml(self, tmp_path: Path):
        """Test parsing invalid YAML file."""
        task_file = tmp_path / "invalid.yaml"
        task_file.write_text("invalid: yaml: content:")

        parser = TaskParser()
        with pytest.raises(TaskParseError, match="Invalid YAML"):
            parser.parse(str(task_file))

    def test_parse_invalid_json(self, tmp_path: Path):
        """Test parsing invalid JSON file."""
        task_file = tmp_path / "invalid.json"
        task_file.write_text("{ invalid json }")

        parser = TaskParser()
        with pytest.raises(TaskParseError, match="Invalid JSON"):
            parser.parse(str(task_file))

    def test_parse_missing_file(self, tmp_path: Path):
        """Test parsing non-existent file."""
        parser = TaskParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(str(tmp_path / "nonexistent.yaml"))

    def test_validate_required_fields(self, tmp_path: Path):
        """Test validation of required fields."""
        # Missing function_name
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            """
project_path: "/path/to/source"
tasks:
  - source_file: "main.c"
        """
        )

        parser = TaskParser()
        with pytest.raises(TaskParseError, match="function_name"):
            parser.parse(str(task_file))

    def test_validate_missing_project_path(self, tmp_path: Path):
        """Test validation when project_path is missing."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            """
tasks:
  - source_file: "main.c"
    function_name: "process"
        """
        )

        parser = TaskParser()
        with pytest.raises(TaskParseError, match="project_path"):
            parser.parse(str(task_file))

    def test_validate_missing_tasks(self, tmp_path: Path):
        """Test validation when tasks is missing."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            """
project_path: "/path/to/source"
        """
        )

        parser = TaskParser()
        with pytest.raises(TaskParseError, match="tasks"):
            parser.parse(str(task_file))

    def test_parse_with_description(self, tmp_path: Path):
        """Test parsing with description field."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            """
project_path: "/path/to/source"
description: "Test batch"
tasks:
  - source_file: "main.c"
    function_name: "process"
        """
        )

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        assert batch.description == "Test batch"

    def test_parse_with_config_overrides(self, tmp_path: Path):
        """Test parsing with config_overrides."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            """
project_path: "/path/to/source"
config_overrides:
  llm:
    temperature: 0.5
tasks:
  - source_file: "main.c"
    function_name: "process"
        """
        )

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        assert batch.config_overrides == {"llm": {"temperature": 0.5}}

    def test_parse_with_priority(self, tmp_path: Path):
        """Test parsing with task priority."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            """
project_path: "/path/to/source"
tasks:
  - source_file: "main.c"
    function_name: "process"
    priority: 10
        """
        )

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        assert batch.tasks[0].priority == 10

    def test_parse_with_dependencies(self, tmp_path: Path):
        """Test parsing with task dependencies."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            """
project_path: "/path/to/source"
tasks:
  - source_file: "main.c"
    function_name: "process"
  - source_file: "handler.c"
    function_name: "handle"
    depends_on:
      - "0"
        """
        )

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        # Task 1 should depend on task 0
        assert len(batch.tasks[1].depends_on) == 1
        assert batch.tasks[0].task_id in batch.tasks[1].depends_on

    def test_parse_string_yaml(self):
        """Test parsing YAML string."""
        content = """
project_path: "/path/to/source"
tasks:
  - source_file: "main.c"
    function_name: "process"
        """

        parser = TaskParser()
        batch = parser.parse_string(content, format_type="yaml")

        assert batch.project_path == "/path/to/source"
        assert len(batch.tasks) == 1

    def test_parse_string_json(self):
        """Test parsing JSON string."""
        content = json.dumps(
            {
                "project_path": "/path/to/source",
                "tasks": [
                    {"source_file": "main.c", "function_name": "process"},
                ],
            }
        )

        parser = TaskParser()
        batch = parser.parse_string(content, format_type="json")

        assert len(batch.tasks) == 1

    def test_parse_string_invalid_format(self):
        """Test parsing with invalid format type."""
        parser = TaskParser()
        with pytest.raises(TaskParseError, match="Unsupported format"):
            parser.parse_string("{}", format_type="xml")

    def test_validate_batch_project_exists(self, tmp_path: Path):
        """Test validate_batch checks project path exists."""
        batch = BatchTask(
            batch_id="test",
            project_path=str(tmp_path / "nonexistent"),
            tasks=[],
        )

        parser = TaskParser()
        errors = parser.validate_batch(batch)

        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_validate_batch_project_is_directory(self, tmp_path: Path):
        """Test validate_batch checks project path is directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        batch = BatchTask(
            batch_id="test",
            project_path=str(file_path),
            tasks=[],
        )

        parser = TaskParser()
        errors = parser.validate_batch(batch)

        assert len(errors) == 1
        assert "not a directory" in errors[0]

    def test_validate_batch_no_errors(self, tmp_path: Path):
        """Test validate_batch returns no errors for valid batch."""
        batch = BatchTask(
            batch_id="test",
            project_path=str(tmp_path),
            tasks=[],
        )

        parser = TaskParser()
        errors = parser.validate_batch(batch)

        assert len(errors) == 0

    def test_relative_path_resolution(self, tmp_path: Path):
        """Test relative path resolution with base_path."""
        # Create a project subdirectory
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            """
project_path: "project"
tasks:
  - source_file: "main.c"
    function_name: "process"
        """
        )

        parser = TaskParser(base_path=tmp_path)
        batch = parser.parse(str(task_file))

        # Project path should be resolved to absolute
        assert batch.project_path == str(tmp_path / "project")

    def test_absolute_path_preserved(self, tmp_path: Path):
        """Test absolute paths are preserved."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            f"""
project_path: "{tmp_path}"
tasks:
  - source_file: "main.c"
    function_name: "process"
        """
        )

        parser = TaskParser(base_path=Path("/other/path"))
        batch = parser.parse(str(task_file))

        assert batch.project_path == str(tmp_path)

    def test_batch_id_generation(self, tmp_path: Path):
        """Test batch ID is generated."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            """
project_path: "/path/to/source"
tasks:
  - source_file: "main.c"
    function_name: "process"
        """
        )

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        assert batch.batch_id.startswith("batch_")
        assert len(batch.batch_id) > len("batch_")

    def test_task_id_generation(self, tmp_path: Path):
        """Test task IDs are generated."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            """
project_path: "/path/to/source"
tasks:
  - source_file: "main.c"
    function_name: "func1"
  - source_file: "main.c"
    function_name: "func2"
        """
        )

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        assert batch.tasks[0].task_id.endswith("_task_000")
        assert batch.tasks[1].task_id.endswith("_task_001")

    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        from fuzz_generator.models import AnalysisTask

        tasks = [
            AnalysisTask(
                task_id="task_1",
                source_file="a.c",
                function_name="func_a",
                depends_on=["task_2"],
            ),
            AnalysisTask(
                task_id="task_2",
                source_file="b.c",
                function_name="func_b",
                depends_on=["task_1"],
            ),
        ]

        batch = BatchTask(
            batch_id="test",
            project_path="/path",
            tasks=tasks,
        )

        parser = TaskParser()
        errors = parser.validate_batch(batch)

        assert any("Circular dependencies" in e for e in errors)

    def test_missing_dependency_detection(self):
        """Test missing dependency detection."""
        from fuzz_generator.models import AnalysisTask

        tasks = [
            AnalysisTask(
                task_id="task_1",
                source_file="a.c",
                function_name="func_a",
                depends_on=["nonexistent"],
            ),
        ]

        batch = BatchTask(
            batch_id="test",
            project_path="/path",
            tasks=tasks,
        )

        parser = TaskParser()
        errors = parser.validate_batch(batch)

        assert any("non-existent task" in e for e in errors)

    def test_parse_non_dict_content(self, tmp_path: Path):
        """Test parsing file with non-dict content."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text("- item1\n- item2")

        parser = TaskParser()
        with pytest.raises(TaskParseError, match="must contain a mapping"):
            parser.parse(str(task_file))

    def test_parse_empty_tasks_list(self, tmp_path: Path):
        """Test parsing with empty tasks list."""
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            """
project_path: "/path/to/source"
tasks: []
        """
        )

        parser = TaskParser()
        batch = parser.parse(str(task_file))

        assert len(batch.tasks) == 0
