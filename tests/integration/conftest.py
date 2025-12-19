"""Integration test fixtures and configuration."""

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires external services)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running",
    )


@pytest.fixture
def sample_c_project(tmp_path: Path) -> Path:
    """Create a sample C project for testing."""
    project_dir = tmp_path / "sample_project"
    project_dir.mkdir()

    # Create handler.c
    handler_c = project_dir / "handler.c"
    handler_c.write_text("""#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Process a request buffer
int process_request(char* buffer, int length) {
    char local_buf[256];

    if (buffer == NULL) {
        return -1;
    }

    if (length > 0 && length < 256) {
        strncpy(local_buf, buffer, length);
        local_buf[length] = '\\0';
        printf("Processing: %s\\n", local_buf);
        return 0;
    }

    return -1;
}

// Parse a header line
int parse_header(const char* line, char* name, char* value) {
    if (line == NULL || name == NULL || value == NULL) {
        return -1;
    }

    const char* colon = strchr(line, ':');
    if (colon == NULL) {
        return -1;
    }

    size_t name_len = colon - line;
    strncpy(name, line, name_len);
    name[name_len] = '\\0';

    // Skip colon and whitespace
    const char* val_start = colon + 1;
    while (*val_start == ' ' || *val_start == '\\t') {
        val_start++;
    }

    strcpy(value, val_start);
    return 0;
}

// Handle a connection
void handle_connection(int socket_fd, char* buffer, size_t buf_size) {
    int bytes_read = 0;

    // Read data from socket (simulated)
    if (buf_size > 0) {
        bytes_read = buf_size;
    }

    if (bytes_read > 0) {
        process_request(buffer, bytes_read);
    }
}
""")

    # Create utils.h
    utils_h = project_dir / "utils.h"
    utils_h.write_text("""#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

// String utilities
size_t safe_strlen(const char* str, size_t max_len);
int safe_strcmp(const char* s1, const char* s2);

// Buffer utilities
void* safe_memcpy(void* dest, const void* src, size_t n);

#endif // UTILS_H
""")

    # Create utils.c
    utils_c = project_dir / "utils.c"
    utils_c.write_text("""#include "utils.h"
#include <string.h>

size_t safe_strlen(const char* str, size_t max_len) {
    if (str == NULL) {
        return 0;
    }

    size_t len = 0;
    while (len < max_len && str[len] != '\\0') {
        len++;
    }
    return len;
}

int safe_strcmp(const char* s1, const char* s2) {
    if (s1 == NULL && s2 == NULL) {
        return 0;
    }
    if (s1 == NULL) {
        return -1;
    }
    if (s2 == NULL) {
        return 1;
    }
    return strcmp(s1, s2);
}

void* safe_memcpy(void* dest, const void* src, size_t n) {
    if (dest == NULL || src == NULL || n == 0) {
        return dest;
    }
    return memcpy(dest, src, n);
}
""")

    return project_dir


@pytest.fixture
def sample_task_file(tmp_path: Path, sample_c_project: Path) -> Path:
    """Create a sample task file."""
    task_file = tmp_path / "tasks.yaml"
    task_file.write_text(f'''project_path: "{sample_c_project}"
description: "Sample analysis tasks"
tasks:
  - source_file: "handler.c"
    function_name: "process_request"
    output_name: "ProcessRequestModel"
  - source_file: "handler.c"
    function_name: "parse_header"
    output_name: "ParseHeaderModel"
''')
    return task_file


@pytest.fixture
def mcp_server_available() -> bool:
    """Check if MCP server is available."""
    import httpx

    mcp_url = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/mcp")

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.post(
                mcp_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 1,
                },
            )
            return response.status_code == 200
    except Exception:
        return False


@pytest.fixture
def require_mcp_server(mcp_server_available: bool):
    """Skip test if MCP server is not available."""
    if not mcp_server_available:
        pytest.skip("MCP server not available")
