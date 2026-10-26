"""MCP-backed loop checks; model decisions are deterministic, never real inference."""

from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

import agent.agent as agent


def action(name="remote_echo", **arguments):
    return json.dumps({"type": "action", "tool": name, "arguments": arguments})


FINAL = '{"type":"final","answer":"Done"}'


def scripted_model(monkeypatch, responses):
    responses = iter(responses)
    messages = []

    def chat(history):
        messages.append(deepcopy(history))
        return next(responses)

    monkeypatch.setattr(agent, "chat", chat)
    return messages


class FakeClient:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.discovery_calls = 0
        self.failure = None
        self.result = CallToolResult(content=[TextContent(type="text", text="echo evidence")])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def list_tools(self):
        self.discovery_calls += 1
        return SimpleNamespace(tools=[SimpleNamespace(
            name="remote_echo", description="A server-only tool.",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        )])

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.failure:
            raise self.failure
        return self.result


@pytest.fixture
def remote(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(agent, "connect", lambda: client)
    return client


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_dynamic_catalog_and_multiple_calls(monkeypatch, remote, caplog):
    caplog.set_level("INFO", logger="agent.agent")
    messages = scripted_model(monkeypatch, [action(text="one"), action(text="two"), FINAL])
    assert agent.run_agent("Inspect") == "Done"
    assert remote.calls == [("remote_echo", {"text": "one"}), ("remote_echo", {"text": "two"})]
    assert remote.discovery_calls == 1 and remote.closed
    prompt = messages[0][0]["content"]
    assert '"name": "remote_echo"' in prompt
    assert '"description": "A server-only tool."' in prompt
    assert '"type": "string"' in prompt
    assert "word_count" not in prompt and "search_docs" not in prompt
    assert "Tool remote_echo returned: echo evidence" in messages[1][-1]["content"]
    assert caplog.messages == [
        '[tool call] remote_echo {"text": "one"}',
        '[observation] Tool remote_echo returned: echo evidence',
        '[tool call] remote_echo {"text": "two"}',
        '[observation] Tool remote_echo returned: echo evidence',
    ]


@pytest.mark.anyio
async def test_async_direct_answer_only_discovers(monkeypatch, remote, caplog):
    caplog.set_level("INFO", logger="agent.agent")
    scripted_model(monkeypatch, [FINAL])
    assert await agent.run_agent_async("Hello") == "Done"
    assert remote.calls == [] and remote.closed
    assert caplog.messages == []


def test_error_can_retry_but_successful_duplicate_is_blocked(monkeypatch, remote, caplog):
    caplog.set_level("INFO", logger="agent.agent")
    remote.result = CallToolResult(is_error=True, content=[TextContent(type="text", text="Invalid arguments")])
    original = remote.call_tool

    async def fail_once(name, arguments):
        result = await original(name, arguments)
        remote.result = CallToolResult(content=[], structured_content={"result": "recovered"})
        return result

    monkeypatch.setattr(remote, "call_tool", fail_once)
    messages = scripted_model(monkeypatch, [action(text="x"), action(text="x"), action(text="x"), FINAL])
    assert agent.run_agent("Retry") == "Done"
    assert len(remote.calls) == 2
    assert "Error: Invalid arguments" in messages[1][-1]["content"]
    assert '"result": "recovered"' in messages[2][-1]["content"]
    assert "already made" in messages[3][-1]["content"]
    assert sum(message.startswith("[tool call]") for message in caplog.messages) == 2
    assert "[observation] Error: Invalid arguments" in caplog.messages
    assert "already made" in caplog.messages[-1]


def test_large_result_is_limited(monkeypatch, remote, caplog):
    caplog.set_level("INFO", logger="agent.agent")
    remote.result = CallToolResult(content=[], structured_content={"result": "x" * 16000})
    messages = scripted_model(monkeypatch, [action(text="x"), FINAL])
    assert agent.run_agent("Inspect") == "Done"
    observation = messages[1][-1]["content"].removeprefix("Observation: ").removesuffix(". Decide what to do next.")
    assert len(observation) == agent.MAX_OBSERVATION_LENGTH
    assert observation.endswith(agent.TRUNCATION_MARKER)
    assert caplog.messages[-1] == f"[observation] {observation}"


def test_cli_shows_actual_tool_evidence_on_stderr():
    script = '''
import builtins
import json
import agent.agent as agent
import main
builtins.input = lambda prompt: "Read the MCP client"
decisions = iter([
    {"type": "action", "tool": "read_file", "arguments": {"path": "mcp_client/client.py"}},
    {"type": "final", "answer": "The client uses sys.executable."},
])
agent.chat = lambda messages: json.dumps(next(decisions))
main.main()
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "The client uses sys.executable.\n"
    assert '[tool call] read_file {"path": "mcp_client/client.py"}' in result.stderr
    assert "[observation] Tool read_file returned:" in result.stderr
    assert "command=sys.executable" in result.stderr


def test_invalid_decision_is_observed_without_invocation(monkeypatch, remote):
    messages = scripted_model(monkeypatch, ["not JSON", FINAL])
    assert agent.run_agent("Inspect") == "Done"
    assert remote.calls == []
    assert "invalid JSON" in messages[1][-1]["content"]


def test_default_path_never_executes_local_registry(monkeypatch, remote):
    def forbidden(*args):
        pytest.fail("local tool execution")

    monkeypatch.setattr(agent.ToolRegistry, "execute", forbidden)
    scripted_model(monkeypatch, [action(text="x"), FINAL])
    assert agent.run_agent("Inspect") == "Done"


def test_direct_answer_does_not_import_rag_or_start_tools():
    script = '''
import sys
import agent.agent as agent
agent.chat = lambda messages: '{"type":"final","answer":"Hello"}'
assert agent.run_agent("Hello") == "Hello"
assert not any(name in sys.modules for name in (
    'rag.answer', 'chromadb', 'sentence_transformers', 'mcp_server.server'
))
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("stage", ["startup", "discovery", "transport", "model", "max_steps"])
def test_fatal_failures_close_session(monkeypatch, remote, stage):
    scripted_model(monkeypatch, [action(text="x")])
    failure = ConnectionError(stage)

    async def fail(*args):
        raise failure

    if stage == "startup":
        monkeypatch.setattr(FakeClient, "__aenter__", fail)
    elif stage == "discovery":
        monkeypatch.setattr(remote, "list_tools", fail)
    elif stage == "transport":
        remote.failure = failure
    elif stage == "model":
        def fail_chat(messages):
            raise failure
        monkeypatch.setattr(agent, "chat", fail_chat)

    with pytest.raises(RuntimeError if stage == "max_steps" else ConnectionError) as caught:
        agent.run_agent("Inspect", max_steps=1)
    if stage != "max_steps":
        assert caught.value is failure
    assert remote.closed is (stage != "startup")


def test_real_stdio_search_and_read(monkeypatch):
    messages = scripted_model(monkeypatch, [
        action("search_code", query="def connect("),
        action("read_file", path="mcp_client/client.py"), FINAL,
    ])
    assert agent.run_agent("Inspect the MCP client") == "Done"
    assert "mcp_client/client.py" in messages[1][-1]["content"]
    assert "StdioServerParameters" in messages[2][-1]["content"]


@pytest.mark.anyio
async def test_docs_versus_implementation_through_real_mcp(monkeypatch):
    from uuid import uuid4
    import chromadb
    from chromadb.config import Settings
    from mcp_server import server
    import rag.answer as answer
    import rag.embeddings as embeddings
    import rag.vector_store as store
    from rag.chunking import Chunk
    from rag.embeddings import EmbeddedChunk

    collection = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False)).create_collection(
        name=f"test_{uuid4().hex}", embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    store.upsert_chunks(collection, [EmbeddedChunk(
        Chunk("The MCP client starts the server using the current interpreter.", {"source": "documents/client.md"}),
        [1.0, 0.0],
    )], ids=["client::0"])

    class Encoder:
        def encode(self, sentences, **kwargs):
            return [[1.0, 0.0] for _ in sentences]

    monkeypatch.setattr(embeddings, "load_embedding_model", Encoder)
    monkeypatch.setattr(store, "open_collection", lambda: collection)
    grounded = []

    def grounded_chat(messages):
        grounded.append(messages)
        return "The documentation requires the current interpreter."

    monkeypatch.setattr(answer, "chat", grounded_chat)
    monkeypatch.setattr(agent, "connect", lambda: Client(server.mcp))
    messages = scripted_model(monkeypatch, [
        action("search_docs", query="How is the server started?"),
        action("search_code", query="command=sys.executable"),
        action("read_file", path="mcp_client/client.py"),
        '{"type":"final","answer":"The implementation matches the documentation: sys.executable."}',
    ])
    result = await agent.run_agent_async("Check the documentation and determine whether the implementation matches it.")
    assert "matches" in result
    assert "[Source: documents/client.md]" in grounded[0][1]["content"]
    assert "documentation requires" in messages[1][-1]["content"]
    assert "command=sys.executable" in messages[2][-1]["content"]
    assert "StdioServerParameters" in messages[3][-1]["content"]
