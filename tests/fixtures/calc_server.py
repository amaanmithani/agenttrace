"""A tiny MCP server for tests: deterministic arithmetic plus a non-deterministic clock."""

import sys
import time

import anyio
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("calc")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def clock() -> str:
    """Current time in nanoseconds (differs every call)."""
    return str(time.time_ns())


@mcp.tool()
def fail() -> str:
    """Always raises."""
    raise ValueError("backend down")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        anyio.run(lambda: mcp.run_streamable_http_async(port=int(sys.argv[2])))
    else:
        mcp.run("stdio")
