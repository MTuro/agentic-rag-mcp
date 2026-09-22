"""First MCP server exposing the existing read-only file listing tool."""

from typing import Annotated

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from tools.filesystem import LIST_FILES_TOOL, list_files


mcp = MCPServer("agentic-rag-mcp")


@mcp.tool(
    name=LIST_FILES_TOOL.name,
    description=LIST_FILES_TOOL.description,
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def mcp_list_files(
    path: Annotated[
        str,
        Field(description="Project-relative directory path. Use '.' for the project root."),
    ],
) -> list[str]:
    """Expose the existing project-confined file listing through MCP."""
    return list_files(path)


if __name__ == "__main__":
    mcp.run(transport="stdio")
