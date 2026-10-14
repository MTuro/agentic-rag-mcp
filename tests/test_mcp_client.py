"""Standalone client contracts, including real stdio lifecycle checks."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
from mcp import StdioServerParameters
from mcp.shared.exceptions import MCPError
import mcp.client.stdio as stdio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_construction_is_lazy_and_uses_current_interpreter(monkeypatch):
    from mcp_client.client import connect

    def forbidden(*args, **kwargs):
        raise AssertionError("construction started a process")

    monkeypatch.setattr(stdio, "_create_platform_compatible_process", forbidden)
    client = connect()
    assert client.server.command == sys.executable
    assert client.server.args == ["-m", "mcp_server.server"]
    assert Path(client.server.cwd) == Path(__file__).resolve().parents[1]
    with pytest.raises(RuntimeError):
        _ = client.session


@pytest.mark.anyio
@pytest.mark.parametrize("fail_in_body", [False, True])
async def test_real_stdio_discovery_invocation_errors_and_cleanup(monkeypatch, fail_in_body):
    from mcp_client.client import connect

    processes = []
    spawn = stdio._create_platform_compatible_process

    async def record_process(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(stdio, "_create_platform_compatible_process", record_process)
    # Two independent sessions exercise repeated startup and shutdown.
    for _ in range(2):
        try:
            async with connect() as client:
                assert client.protocol_version
                discovery = await client.list_tools()
                assert {tool.name for tool in discovery.tools} == {
                    "list_files", "read_file", "search_code", "git_status", "git_diff", "search_docs"
                }
                assert all(tool.description and tool.input_schema for tool in discovery.tools)
                listing_tool = next(t for t in discovery.tools if t.name == "list_files")
                assert listing_tool.input_schema["required"] == ["path"]
                assert listing_tool.input_schema["properties"]["path"]["type"] == "string"
                result = await client.call_tool(listing_tool.name, {"path": "."})
                assert not result.is_error
                assert "README.md" in result.structured_content["result"]
                search = await client.call_tool("search_code", {"query": "mcp_list_files"})
                assert not search.is_error
                assert any(match["filename"] == "mcp_server/server.py" for match in search.structured_content["result"])
                for name, arguments in [
                    ("list_files", {"path": ".."}),
                    ("list_files", {"path": 123}),
                    ("list_files", {}),
                    ("git_status", {"unexpected": True}),
                    ("unknown_tool", {}),
                ]:
                    error = await client.call_tool(name, arguments)
                    assert error.is_error
                    assert error.content
                if fail_in_body:
                    raise ValueError("caller failed")
        except* ValueError as group:
            def leaves(exc):
                if isinstance(exc, BaseExceptionGroup):
                    return [leaf for child in exc.exceptions for leaf in leaves(child)]
                return [exc]

            assert fail_in_body
            assert [str(exc) for exc in leaves(group)] == ["caller failed"]
        assert processes[-1].returncode is not None
        if not fail_in_body:
            with pytest.raises(RuntimeError):
                _ = client.session
    assert len(processes) == 2


@pytest.mark.anyio
async def test_initialization_failure_reaps_child(monkeypatch):
    from mcp_client.client import connect

    processes = []
    spawn = stdio._create_platform_compatible_process

    async def record_process(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(stdio, "_create_platform_compatible_process", record_process)
    parameters = StdioServerParameters(command=sys.executable, args=["-c", "pass"])
    caught = False
    try:
        async with connect(parameters):
            pytest.fail("initialization unexpectedly succeeded")
    except* MCPError:
        caught = True
    assert caught
    assert len(processes) == 1
    assert processes[0].returncode is not None


@pytest.mark.anyio
async def test_start_failure_is_fatal():
    from mcp_client.client import connect

    caught = False
    try:
        async with connect(StdioServerParameters(command="missing-mcp-executable-12345")):
            pytest.fail("startup unexpectedly succeeded")
    except* FileNotFoundError:
        caught = True
    assert caught


def test_import_has_no_application_or_external_side_effects():
    script = """
import sys
def guard(event, args):
    if event in {'subprocess.Popen', 'socket.connect'}:
        raise AssertionError(event)
sys.addaudithook(guard)
import mcp_client
import mcp_client.client
assert not any(name in sys.modules for name in (
    'sentence_transformers', 'ollama', 'chromadb', 'rag.answer', 'mcp_server.server'
))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.anyio
@pytest.mark.parametrize("failure", [None, "discovery", "exit"])
async def test_custom_catalog_and_protocol_failures_reap_child(monkeypatch, failure):
    from mcp_client.client import connect

    processes = []
    spawn = stdio._create_platform_compatible_process

    async def record_process(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(stdio, "_create_platform_compatible_process", record_process)
    script = '''
import os
from mcp.server import MCPServer
server = MCPServer("independent-review")
@server.tool(description="Echo the exact caller text.")
def review_echo(text: str) -> str:
    if text == "exit":
        os._exit(17)
    return text
'''
    if failure == "discovery":
        script += '''
async def fail_discovery():
    raise RuntimeError("discovery deliberately unavailable")
server.list_tools = fail_discovery
'''
    script += '\nserver.run(transport="stdio")\n'
    parameters = StdioServerParameters(command=sys.executable, args=["-c", script])

    async def exercise():
        async with connect(parameters) as client:
            tools = (await client.list_tools()).tools
            assert len(tools) == 1
            tool = tools[0]
            assert tool.name == "review_echo"
            assert tool.description == "Echo the exact caller text."
            assert tool.input_schema["required"] == ["text"]
            assert tool.input_schema["properties"]["text"]["type"] == "string"
            text = "exit" if failure == "exit" else "forwarded unchanged"
            result = await client.call_tool(tool.name, {"text": text})
            assert not result.is_error
            assert result.structured_content["result"] == text

    if failure:
        caught = False
        try:
            await exercise()
        except* MCPError:
            caught = True
        assert caught, "protocol failure was not surfaced"
    else:
        await exercise()
    assert len(processes) == 1
    assert processes[0].returncode is not None
    if failure == "exit":
        assert processes[0].returncode == 17


@pytest.mark.parametrize(("arguments", "code"), [
    (["--list-tools"], 0),
    (["list_files", '{"path":"."}'], 0),
    (["list_files", '{"path":".."}'], 1),
    (["unknown_tool", '{}'], 1),
    (["list_files", '[]'], 2),
    (["list_files", '{broken'], 2),
])
def test_cli(arguments, code):
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "mcp_client.client", *arguments],
        capture_output=True, timeout=60,
    )
    assert result.returncode == code, result.stderr
    if code != 2:
        payload = json.loads(result.stdout)
        if arguments == ["--list-tools"]:
            assert any(t["name"] == "list_files" and t["inputSchema"] for t in payload["tools"])
        else:
            assert payload["isError"] is (code == 1)
