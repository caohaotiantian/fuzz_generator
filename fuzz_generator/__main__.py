"""Entry point for running fuzz_generator as a module."""

import os

# Set NO_PROXY to bypass proxy for local services
# This is needed for LM Studio and Joern MCP Server running locally
if "NO_PROXY" not in os.environ:
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
elif "localhost" not in os.environ["NO_PROXY"]:
    os.environ["NO_PROXY"] += ",localhost,127.0.0.1"

from fuzz_generator.cli import cli

if __name__ == "__main__":
    cli()
