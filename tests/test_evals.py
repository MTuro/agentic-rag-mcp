"""Script model decisions while exercising the evaluation runner and MCP tools."""

import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from mcp import Client
import pytest

import agent.agent as agent
from agent.trace import FinalDecision, RejectedDecision, StepStarted, ToolAttempt, ToolObservation
from evals import __main__ as cli
from evals.runner import CASES, run_case
from tools.registry import Tool, ToolRegistry


CASES_BY_ID = {case.id: case for case in CASES}
DOCS_QUERY = "How does ChromaDB rank retrieved chunks?"
DOCS_ANSWER = "ChromaDB uses **cosine distance**. Lower distances indicate closer matches."
DOCS_EXCERPT = "ChromaDB searches using cosine distance. Smaller distance means a closer match."


def action(name, **arguments):
    return json.dumps({"type": "action", "tool": name, "arguments": arguments})


def final(answer):
    return json.dumps({"type": "final", "answer": answer})


def script(monkeypatch, *responses):
    remaining = iter(responses)

    def chat(messages):
        response = next(remaining)
        if isinstance(response, Exception):
            raise response
        return response(messages) if callable(response) else response

    monkeypatch.setattr(agent, "chat", chat)


def evaluate(case_id):
    return asyncio.run(run_case(CASES_BY_ID[case_id]))


@pytest.mark.parametrize(("answer", "passed"), [("4", True), ("14", False)])
def test_direct_answer_and_cli_exit_code(monkeypatch, capsys, answer, passed):
    monkeypatch.setenv("OLLAMA_MODEL", "evaluation-model")
    script(monkeypatch, final(answer))
    assert cli.main(["--case", "direct"]) == int(not passed)
    output = capsys.readouterr().out
    assert "Ollama model: evaluation-model" in output
    assert "direct (1 steps used)" in output
    assert "Terminal error:" not in output
    assert "Human review:" in output


def test_retry_duplicate_unexpected_attempt_and_trace_order(monkeypatch):
    search = action("search_code", query="def retrieve_chunks(")
    script(
        monkeypatch, action("search_code"), search, search, action("list_files", path="."),
        final("`rag/retriever.py`"),
    )
    report = evaluate("code")

    assert not report.passed
    assert report.unexpected_attempts == 1
    assert report.excess_attempts == 2
    assert [(event.step, type(event)) for event in report.trace] == [
        (1, StepStarted), (1, ToolAttempt), (1, ToolObservation),
        (2, StepStarted), (2, ToolAttempt), (2, ToolObservation),
        (3, StepStarted), (3, ToolAttempt), (3, RejectedDecision),
        (4, StepStarted), (4, ToolAttempt), (4, ToolObservation),
        (5, StepStarted), (5, FinalDecision),
    ]
    first_attempt, first_result = report.trace[1:3]
    assert first_attempt.arguments == {}
    assert not first_result.succeeded and first_result.observation.startswith("Error:")
    second_result = report.trace[5]
    assert second_result.succeeded and "rag/retriever.py" in second_result.observation
    assert "already made" in report.trace[8].observation


@pytest.mark.parametrize(("answer", "passed"), [
    ("`rag/retriever.py` defines it.", True),
    ("See rag/retriever.py.bak.", False),
])
def test_code_answer_and_optional_read(monkeypatch, answer, passed):
    script(
        monkeypatch,
        action("search_code", query="def retrieve_chunks("),
        action("read_file", path="rag/retriever.py"),
        final(answer),
    )
    report = evaluate("code")
    assert report.passed == passed
    assert report.unexpected_attempts == report.excess_attempts == 0


def test_attempt_survives_transport_failure(monkeypatch):
    def connect():
        async def list_tools():
            return SimpleNamespace(tools=[])

        async def call_tool(name, arguments):
            assert trace[-1] == ToolAttempt(1, "search_code", {"query": "example"})
            raise RuntimeError("transport interrupted")

        @asynccontextmanager
        async def connection():
            yield SimpleNamespace(list_tools=list_tools, call_tool=call_tool)

        return connection()

    trace = []
    monkeypatch.setattr(agent, "connect", connect)
    script(monkeypatch, action("search_code", query="example"))
    with pytest.raises(RuntimeError, match="transport interrupted"):
        asyncio.run(agent.run_agent_with_result("Inspect", trace=trace.append))
    assert trace == [StepStarted(1), ToolAttempt(1, "search_code", {"query": "example"})]


def test_trace_observation_matches_the_bounded_model_evidence(monkeypatch):
    registry = ToolRegistry()
    registry.register(Tool("large", "Return a large result.", {"type": "object"}, lambda: "x" * 9000))
    trace = []

    def answer_after_observation(messages):
        event = trace[-2]
        assert trace[-1] == StepStarted(2)
        assert isinstance(event, ToolObservation) and event.succeeded
        assert len(event.observation) == agent.MAX_OBSERVATION_LENGTH
        assert event.observation.endswith(agent.TRUNCATION_MARKER)
        assert messages[-1]["content"] == f"Observation: {event.observation}. Decide what to do next."
        return final("Done.")

    script(monkeypatch, action("large"), answer_after_observation)
    result = asyncio.run(agent.run_agent_with_result("Inspect", tool_registry=registry, trace=trace.append))
    assert result.answer == "Done."


@pytest.mark.parametrize(("decision", "error_type", "passed"), [
    (action("search_code", query="def retrieve_chunks("), agent.StepLimitError, True),
    (RuntimeError("Agent stopped after reaching the maximum of 1 steps."), RuntimeError, False),
    (final("rag/retriever.py"), type(None), False),
])
def test_step_limit_requires_successful_call_and_typed_error(monkeypatch, decision, error_type, passed):
    script(monkeypatch, decision)
    report = evaluate("step-limit")
    assert type(report.error) is error_type
    assert report.passed == passed


def test_suite_continues_after_runtime_error(monkeypatch, capsys):
    script(monkeypatch, ConnectionError("Ollama unavailable"), action("search_code", query="def retrieve_chunks("))
    assert asyncio.run(cli._run([CASES_BY_ID["direct"], CASES_BY_ID["step-limit"]])) == 1
    output = capsys.readouterr().out
    assert "Terminal error: ConnectionError: Ollama unavailable" in output
    assert "[PASS] step-limit (1 steps used)" in output
    assert "1/2 cases passed" in output


@pytest.mark.parametrize(("status", "answer", "passed"), [
    ("Working tree clean.", "Working tree clean.", True),
    (" M README.md\n?? new.txt", "```text\n M README.md\n?? new.txt\n```", True),
    (" M README.md\n?? new.txt", "```\nM  README.md\n?? new.txt\n```", False),
])
def test_git_compares_status_columns_with_observation(monkeypatch, status, answer, passed):
    from mcp_server import server

    monkeypatch.setattr(server, "git_status", lambda: status)
    monkeypatch.setattr(agent, "connect", lambda: Client(server.mcp))
    script(monkeypatch, action("git_status"), final(answer))
    assert evaluate("git").passed == passed


@pytest.mark.parametrize(("source", "text", "answer", "hit", "passed"), [
    ("documents/README.md", DOCS_EXCERPT, DOCS_ANSWER, 2, True),
    ("documents/another.md", DOCS_EXCERPT, DOCS_ANSWER, None, False),
    ("documents/README.md", "ChromaDB uses cosine distance.", DOCS_ANSWER, None, False),
    (
        "documents/README.md", DOCS_EXCERPT,
        "ChromaDB uses cosine distance. Smaller distances indicate less similar matches.", 2, False,
    ),
    (
        "documents/README.md", DOCS_EXCERPT,
        "Cosine distance is not used. Smaller distances indicate closer matches.", 2, False,
    ),
])
def test_docs_grades_answer_and_replays_actual_query_at_top_three(
    monkeypatch, capsys, source, text, answer, hit, passed,
):
    from mcp_server import server
    import rag.answer as rag_answer
    from rag.chunking import Chunk
    import rag.embeddings as embeddings
    import rag.vector_store as store

    collection = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False)).create_collection(
        name=f"eval_{uuid4().hex}", embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    rows = [
        ("nearest", "The closest stored example has unrelated text.", "documents/extra.md", [1.0, 0.0]),
        ("target", text, source, [1.0, 1.0]),
        ("third", "ChromaDB stores text and metadata.", "documents/extra.md", [0.0, 1.0]),
        ("fourth", DOCS_EXCERPT, "documents/README.md", [-1.0, 0.0]),
    ]
    store.upsert_chunks(collection, [
        embeddings.EmbeddedChunk(Chunk(content, {"source": path}), vector)
        for _, content, path, vector in rows
    ], ids=[row[0] for row in rows])
    queries = []

    class Encoder:
        def encode(self, sentences, **kwargs):
            queries.extend(sentences)
            return [[1.0, 0.0] for _ in sentences]

    def grounded_chat(messages):
        assert text in messages[1]["content"]
        assert DOCS_QUERY in messages[1]["content"]
        return "Grounded tool answer."

    monkeypatch.setattr(embeddings, "load_embedding_model", Encoder)
    monkeypatch.setattr(store, "open_collection", lambda: collection)
    monkeypatch.setattr(rag_answer, "chat", grounded_chat)
    monkeypatch.setattr(agent, "connect", lambda: Client(server.mcp))

    def answer_after_observation(messages):
        assert "Grounded tool answer." in messages[-1]["content"]
        return final(answer)

    script(monkeypatch, action("search_docs", query=DOCS_QUERY), answer_after_observation)
    assert cli.main(["--case", "docs"]) == int(not passed)
    output = capsys.readouterr().out
    assert queries == [DOCS_QUERY, DOCS_QUERY]
    assert f"Top-3 retrieval replay for {DOCS_QUERY!r}" in output
    assert f"Rank 2: {source}" in output and repr(text) in output
    assert ("[supports both facts]" in output) == (hit is not None)
    assert "Rank 3:" in output and "Rank 4:" not in output
