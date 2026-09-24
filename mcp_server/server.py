"""MCP server exposing the project's existing read-only tools."""

from typing import Annotated

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from tools.filesystem import (
    LIST_FILES_TOOL,
    READ_FILE_TOOL,
    list_files,
    read_file,
)
from tools.git import GIT_DIFF_TOOL, GIT_STATUS_TOOL, git_diff, git_status
from tools.rag import SEARCH_DOCS_TOOL, search_docs
from tools.search import SEARCH_CODE_TOOL, search_code


mcp = MCPServer("agentic-rag-mcp")


@mcp.tool(
    name=LIST_FILES_TOOL.name,
    description=LIST_FILES_TOOL.description,
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def mcp_list_files(
    path: Annotated[
        str,
        Field(
            description=LIST_FILES_TOOL.argument_schema["properties"]["path"]["description"],
            strict=True,
        ),
    ],
) -> list[str]:
    """Expose the existing project-confined file listing through MCP."""
    return list_files(path)


@mcp.tool(
    name=READ_FILE_TOOL.name,
    description=READ_FILE_TOOL.description,
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def mcp_read_file(
    path: Annotated[
        str,
        Field(
            description=READ_FILE_TOOL.argument_schema["properties"]["path"]["description"],
            strict=True,
        ),
    ],
) -> str:
    """Expose the existing project-confined file reader through MCP."""
    return read_file(path)


@mcp.tool(
    name=SEARCH_CODE_TOOL.name,
    description=SEARCH_CODE_TOOL.description,
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def mcp_search_code(
    query: Annotated[
        str,
        Field(
            description=SEARCH_CODE_TOOL.argument_schema["properties"]["query"]["description"],
            strict=True,
        ),
    ],
) -> list[dict[str, str | int]]:
    """Expose the existing project-confined code search through MCP."""
    return search_code(query)


@mcp.tool(
    name=GIT_STATUS_TOOL.name,
    description=GIT_STATUS_TOOL.description,
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def mcp_git_status() -> str:
    """Expose the existing read-only Git status inspection through MCP."""
    return git_status()


@mcp.tool(
    name=GIT_DIFF_TOOL.name,
    description=GIT_DIFF_TOOL.description,
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def mcp_git_diff(
    staged: Annotated[
        bool,
        Field(
            description=GIT_DIFF_TOOL.argument_schema["properties"]["staged"]["description"],
            strict=True,
        ),
    ] = False,
) -> str:
    """Expose the existing staged or unstaged Git diff through MCP."""
    return git_diff(staged=staged)


@mcp.tool(
    name=SEARCH_DOCS_TOOL.name,
    description=SEARCH_DOCS_TOOL.description,
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def mcp_search_docs(
    query: Annotated[
        str,
        Field(
            description=SEARCH_DOCS_TOOL.argument_schema["properties"]["query"]["description"],
            strict=True,
        ),
    ],
) -> str:
    """Expose the existing lazy grounded-document search through MCP."""
    return search_docs(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
