"""HTTP contract tests through LangGraph with deterministic Qwen decisions and real MCP stdio."""

import json

import httpx
import pytest

import agent.agent as agent
from app import app


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


async def test_chat_returns_direct_answer(client, monkeypatch):
    def model(messages):
        assert messages[-1] == {"role": "user", "content": "  Hello  "}
        return '{"type":"final","answer":"Hello!"}'

    monkeypatch.setattr(agent, "chat", model)
    response = await client.post("/chat", json={"message": "  Hello  "})

    assert response.status_code == 200
    assert response.json() == {"answer": "Hello!", "tools_used": []}


async def test_chat_reports_successful_tools_in_first_use_order(client, monkeypatch):
    decisions = iter([
        {"type": "action", "tool": "list_files", "arguments": {"path": "README.md"}},
        {"type": "action", "tool": "read_file", "arguments": {"path": "README.md"}},
        {"type": "action", "tool": "search_code", "arguments": {"query": "def run_agent"}},
        {"type": "action", "tool": "read_file", "arguments": {"path": "main.py"}},
        {"type": "final", "answer": "Inspected the agent entry points."},
    ])
    observations = []

    def model(messages):
        observations.append(messages[-1]["content"])
        return json.dumps(next(decisions))

    monkeypatch.setattr(agent, "chat", model)
    response = await client.post("/chat", json={"message": "Inspect the agent entry points."})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Inspected the agent entry points.",
        "tools_used": ["read_file", "search_code"],
    }
    assert "Error:" in observations[1]
    assert "educational software engineering assistant" in observations[2]
    assert "agent/agent.py" in observations[3]
    assert "run_agent_graph(prompt)" in observations[4]


@pytest.mark.parametrize("body", [
    b"{}",
    b'{"message":null}',
    b'{"message":123}',
    b'{"message":""}',
    b'{"message":" \\t\\n "}',
    b'{"message":',
])
async def test_chat_rejects_invalid_request_without_running_agent(client, monkeypatch, body):
    async def unexpected_run(*args):
        pytest.fail("Invalid requests must not run the agent.")

    monkeypatch.setattr("api.routes.run_agent_graph", unexpected_run)
    response = await client.post("/chat", content=body, headers={"Content-Type": "application/json"})

    assert response.status_code == 422


async def test_chat_returns_generic_error_when_agent_fails(client, monkeypatch):
    def model(messages):
        raise ConnectionError("Private Ollama connection details")

    monkeypatch.setattr(agent, "chat", model)
    response = await client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
