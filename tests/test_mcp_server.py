"""Tests for the first MCP server and its list_files tool."""

import sys

import pytest
from mcp import Client, StdioServerParameters

from mcp_server import server


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_adapter_reuses_existing_list_files(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_list_files(path: str) -> list[str]:
        calls.append(path)
        return ["README.md"]

    monkeypatch.setattr(server, "list_files", fake_list_files)

    assert server.mcp_list_files("docs") == ["README.md"]
    assert calls == ["docs"]


@pytest.mark.anyio
async def test_server_discovers_only_list_files_with_safe_schema() -> None:
    async with Client(server.mcp, raise_exceptions=True) as client:
        result = await client.list_tools()

    assert len(result.tools) == 1
    tool = result.tools[0]
    assert tool.name == "list_files"
    assert tool.description == server.LIST_FILES_TOOL.description
    assert tool.input_schema["required"] == ["path"]
    assert tool.input_schema["properties"]["path"]["type"] == "string"
    assert set(tool.input_schema["properties"]) == {"path"}
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.open_world_hint is False


@pytest.mark.anyio
async def test_server_calls_existing_list_files_and_reports_tool_errors() -> None:
    async with Client(server.mcp, raise_exceptions=True) as client:
        success = await client.call_tool("list_files", {"path": "."})
        missing = await client.call_tool("list_files", {"path": "missing-directory"})
        traversal = await client.call_tool("list_files", {"path": ".."})
        invalid = await client.call_tool("list_files", {})

    assert success.is_error is False
    assert success.structured_content is not None
    assert "README.md" in success.structured_content["result"]
    assert missing.is_error is True
    assert traversal.is_error is True
    assert invalid.is_error is True


@pytest.mark.anyio
async def test_stdio_startup_discovers_and_calls_list_files() -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
    )

    async with Client(parameters) as client:
        tools = await client.list_tools()
        result = await client.call_tool("list_files", {"path": "."})

    assert [tool.name for tool in tools.tools] == ["list_files"]
    assert result.is_error is False
    assert result.structured_content is not None
    assert "README.md" in result.structured_content["result"]
