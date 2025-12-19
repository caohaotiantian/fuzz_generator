"""Pytest configuration and shared fixtures."""

import asyncio
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a sample C project for testing."""
    project_dir = tmp_path / "sample_project"
    project_dir.mkdir()

    # Create a sample C file
    (project_dir / "main.c").write_text("""
#include <stdio.h>
#include <string.h>

int process_request(char* buffer, int length) {
    char local_buf[256];
    if (length > 0 && length < 256) {
        strncpy(local_buf, buffer, length);
        printf("Processing: %s\\n", local_buf);
        return 0;
    }
    return -1;
}

int main(int argc, char* argv[]) {
    char buf[100] = "test";
    return process_request(buf, strlen(buf));
}
""")

    # Create a header file
    (project_dir / "handler.h").write_text("""
#ifndef HANDLER_H
#define HANDLER_H

int process_request(char* buffer, int length);

#endif
""")

    return project_dir


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Create a sample configuration file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
version: "1.0"

llm:
  base_url: "http://localhost:11434/v1"
  model: "test-model"
  temperature: 0.5

mcp_server:
  url: "http://localhost:8000/mcp"
  timeout: 30
""")
    return config_file


@pytest.fixture
def sample_tasks(tmp_path: Path) -> Path:
    """Create a sample tasks file."""
    tasks_file = tmp_path / "tasks.yaml"
    tasks_file.write_text("""
project_path: "/path/to/source"
description: "Sample batch tasks"

tasks:
  - source_file: "main.c"
    function_name: "process_request"
    output_name: "RequestModel"
  - source_file: "main.c"
    function_name: "main"
""")
    return tasks_file


@pytest.fixture
def invalid_tasks(tmp_path: Path) -> Path:
    """Create an invalid tasks file."""
    tasks_file = tmp_path / "invalid_tasks.yaml"
    tasks_file.write_text("""
tasks:
  - source_file: "main.c"
    # missing function_name
""")
    return tasks_file
