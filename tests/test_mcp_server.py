"""Tests for the MCP adapters over the existing read-only tools."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
from mcp import Client, StdioServerParameters

from mcp_server import server


TOOL_NAMES = [
    "list_files",
    "read_file",
    "search_code",
    "git_status",
    "git_diff",
    "search_docs",
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_adapters_delegate_once_and_return_results_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    search_results = [{"filename": "app.py", "line_number": 2, "line": "needle"}]

    monkeypatch.setattr(
        server,
        "list_files",
        lambda path: calls.append(("list_files", path)) or ["README.md"],
    )
    monkeypatch.setattr(
        server,
        "read_file",
        lambda path: calls.append(("read_file", path)) or "contents",
    )
    monkeypatch.setattr(
        server,
        "search_code",
        lambda query: calls.append(("search_code", query)) or search_results,
    )
    monkeypatch.setattr(
        server,
        "git_status",
        lambda: calls.append(("git_status", None)) or "Working tree clean.",
    )
    monkeypatch.setattr(
        server,
        "git_diff",
        lambda *, staged=False: calls.append(("git_diff", staged)) or "diff",
    )
    monkeypatch.setattr(
        server,
        "search_docs",
        lambda query: calls.append(("search_docs", query)) or "grounded answer",
    )

    assert server.mcp_list_files("docs") == ["README.md"]
    assert server.mcp_read_file("README.md") == "contents"
    assert server.mcp_search_code("needle") is search_results
    assert server.mcp_git_status() == "Working tree clean."
    assert server.mcp_git_diff(staged=True) == "diff"
    assert server.mcp_search_docs("What is RAG?") == "grounded answer"
    assert calls == [
        ("list_files", "docs"),
        ("read_file", "README.md"),
        ("search_code", "needle"),
        ("git_status", None),
        ("git_diff", True),
        ("search_docs", "What is RAG?"),
    ]


@pytest.mark.anyio
async def test_server_discovers_exact_tool_catalog_and_schemas() -> None:
    constants = {
        "list_files": server.LIST_FILES_TOOL,
        "read_file": server.READ_FILE_TOOL,
        "search_code": server.SEARCH_CODE_TOOL,
        "git_status": server.GIT_STATUS_TOOL,
        "git_diff": server.GIT_DIFF_TOOL,
        "search_docs": server.SEARCH_DOCS_TOOL,
    }

    async with Client(server.mcp, raise_exceptions=True) as client:
        result = await client.list_tools()

    assert [tool.name for tool in result.tools] == TOOL_NAMES
    for tool in result.tools:
        assert tool.description == constants[tool.name].description
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.open_world_hint is False

    schemas = {tool.name: tool.input_schema for tool in result.tools}
    assert schemas["list_files"]["required"] == ["path"]
    assert schemas["read_file"]["required"] == ["path"]
    assert schemas["search_code"]["required"] == ["query"]
    assert schemas["search_docs"]["required"] == ["query"]
    assert schemas["git_status"].get("required", []) == []
    assert schemas["git_diff"].get("required", []) == []
    assert schemas["git_diff"]["properties"]["staged"]["default"] is False
    assert schemas["git_diff"]["properties"]["staged"]["type"] == "boolean"
    assert {name: set(schema["properties"]) for name, schema in schemas.items()} == {
        "list_files": {"path"},
        "read_file": {"path"},
        "search_code": {"query"},
        "git_status": set(),
        "git_diff": {"staged"},
        "search_docs": {"query"},
    }
    assert all("project_root" not in schema["properties"] for schema in schemas.values())


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("list_files", {}),
        ("list_files", {"path": 123}),
        ("list_files", {"path": ".", "extra": True}),
        ("read_file", {}),
        ("read_file", {"path": None}),
        ("read_file", {"path": "README.md", "extra": True}),
        ("search_code", {}),
        ("search_code", {"query": 123}),
        ("search_code", {"query": "mcp", "extra": True}),
        ("search_docs", {}),
        ("search_docs", {"query": 123}),
        ("search_docs", {"query": "valid documentation query", "extra": True}),
        ("git_status", {"unexpected": True}),
        ("git_diff", {"staged": "yes"}),
        ("git_diff", {"extra": True}),
    ],
)
async def test_server_rejects_invalid_arguments(
    name: str, arguments: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    rag_calls = []
    monkeypatch.setattr(server, "search_docs", lambda query: rag_calls.append(query) or "answer")
    async with Client(server.mcp, raise_exceptions=True) as client:
        result = await client.call_tool(name, arguments)

    assert result.is_error is True
    assert rag_calls == []


@pytest.mark.anyio
async def test_underlying_failures_become_mcp_tool_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_git_status() -> str:
        raise RuntimeError("Git unavailable")

    def fail_search_docs(query: str) -> str:
        raise RuntimeError(f"RAG unavailable for {query}")

    monkeypatch.setattr(server, "git_status", fail_git_status)
    monkeypatch.setattr(server, "search_docs", fail_search_docs)

    async with Client(server.mcp, raise_exceptions=True) as client:
        missing = await client.call_tool("read_file", {"path": "missing-file"})
        traversal = await client.call_tool("list_files", {"path": ".."})
        git_failure = await client.call_tool("git_status", {})
        rag_failure = await client.call_tool("search_docs", {"query": "RAG?"})

    assert missing.is_error is True
    assert traversal.is_error is True
    assert git_failure.is_error is True
    assert rag_failure.is_error is True


def test_import_and_discovery_do_not_initialize_rag_dependencies() -> None:
    script = """
import asyncio
import rag.answer
import rag.embeddings
import rag.vector_store

def fail(*args, **kwargs):
    raise AssertionError("RAG dependency initialized")

rag.answer.chat = fail
rag.embeddings.load_embedding_model = fail
rag.vector_store.open_collection = fail

from mcp import Client
from mcp_server.server import mcp

async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert len(tools.tools) == 6

asyncio.run(main())
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


@pytest.mark.parametrize(
    "imports",
    [
        "import mcp_server.server; import tools; import rag.vector_store; import rag.answer",
        "import rag.answer; import rag.vector_store; import tools; import mcp_server.server",
    ],
)
def test_representative_import_orders_have_no_cycles(imports: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", imports],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


@pytest.mark.anyio
async def test_stdio_startup_discovers_and_calls_dependency_free_tools() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": Path.cwd().as_posix(),
        }
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        env=environment,
    )

    async with Client(parameters) as client:
        tools = await client.list_tools()
        results = [
            await client.call_tool("list_files", {"path": "."}),
            await client.call_tool("read_file", {"path": "README.md"}),
            await client.call_tool("search_code", {"query": "search_docs"}),
            await client.call_tool("git_status", {}),
            await client.call_tool("git_diff", {}),
            await client.call_tool("git_diff", {"staged": True}),
        ]

    assert [tool.name for tool in tools.tools] == TOOL_NAMES
    assert all(result.is_error is False for result in results)
    assert results[0].structured_content is not None
    assert "README.md" in results[0].structured_content["result"]
    assert results[1].structured_content is not None
    assert "# agentic-rag-mcp" in results[1].structured_content["result"]
    assert results[2].structured_content is not None
    assert results[2].structured_content["result"]
