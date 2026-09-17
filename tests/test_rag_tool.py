"""RAG tool and agent integration without a running Ollama server."""

from collections.abc import Iterator
from uuid import uuid4

import chromadb
from chromadb.config import Settings
import pytest

import agent.agent as agent_module
import rag.answer as answer_module
import rag.embeddings as embeddings_module
import rag.vector_store as vector_store_module
import tools.rag as rag_tool
from rag.answer import NO_EVIDENCE
from rag.chunking import Chunk
from rag.embeddings import EmbeddedChunk, EmbeddingError
from rag.vector_store import upsert_chunks


def test_tool_schema_and_valid_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object()
    collection = object()
    calls = []
    monkeypatch.setattr(embeddings_module, "load_embedding_model", lambda: model)
    monkeypatch.setattr(vector_store_module, "open_collection", lambda: collection)

    def fake_answer(question, received_collection, *, model):
        calls.append((question, received_collection, model))
        return "Grounded answer."

    monkeypatch.setattr(answer_module, "answer_question", fake_answer)
    definition = rag_tool.SEARCH_DOCS_TOOL.model_definition()

    assert definition["name"] == "search_docs"
    assert definition["arguments"]["required"] == ["query"]
    assert definition["arguments"]["properties"]["query"]["type"] == "string"
    assert definition["arguments"]["additionalProperties"] is False
    assert rag_tool.SEARCH_DOCS_TOOL.implementation(query="What is RAG?") == "Grounded answer."
    assert calls == [("What is RAG?", collection, model)]


@pytest.mark.parametrize("query,error", [(None, TypeError), ("", ValueError), (" \n", ValueError)])
def test_invalid_query_does_not_load_dependencies(monkeypatch, query, error) -> None:
    monkeypatch.setattr(embeddings_module, "load_embedding_model", lambda: pytest.fail("model loaded"))
    with pytest.raises(error, match="query"):
        rag_tool.search_docs(query)


def test_no_evidence_is_returned_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings_module, "load_embedding_model", lambda: object())
    monkeypatch.setattr(vector_store_module, "open_collection", lambda: object())
    monkeypatch.setattr(answer_module, "answer_question", lambda *args, **kwargs: NO_EVIDENCE)
    assert rag_tool.search_docs("Unknown?") == NO_EVIDENCE


def test_dependency_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = EmbeddingError("Model unavailable")

    def fail():
        raise failure

    monkeypatch.setattr(embeddings_module, "load_embedding_model", fail)
    monkeypatch.setattr(vector_store_module, "open_collection", lambda: pytest.fail("collection opened"))
    with pytest.raises(EmbeddingError) as error:
        rag_tool.search_docs("Question?")
    assert error.value is failure


def test_agent_advertises_selects_and_observes_search_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    responses: Iterator[str] = iter([
        '{"type": "action", "tool": "search_docs", "arguments": {"query": "What is RAG?"}}',
        '{"type": "final", "answer": "RAG retrieves document context."}',
    ])
    calls = []

    def fake_chat(messages):
        calls.append([message.copy() for message in messages])
        return next(responses)

    monkeypatch.setattr(agent_module, "chat", fake_chat)
    monkeypatch.setattr(embeddings_module, "load_embedding_model", lambda: object())
    monkeypatch.setattr(vector_store_module, "open_collection", lambda: object())
    monkeypatch.setattr(answer_module, "answer_question", lambda *args, **kwargs: "Grounded answer.")

    assert agent_module.run_agent("What do the docs say about RAG?") == "RAG retrieves document context."
    assert '"name": "search_docs"' in calls[0][0]["content"]
    assert calls[1][-1]["content"] == (
        "Observation: Tool search_docs returned: Grounded answer.. Decide what to do next."
    )


def test_direct_final_does_not_load_rag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_module, "chat", lambda messages: '{"type": "final", "answer": "Hello."}'
    )
    monkeypatch.setattr(embeddings_module, "load_embedding_model", lambda: pytest.fail("RAG loaded"))
    assert agent_module.run_agent("Hello") == "Hello."


def test_agent_to_ephemeral_chroma_to_grounded_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    collection = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False)
    ).create_collection(
        name=f"test_{uuid4().hex}",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    upsert_chunks(
        collection,
        [EmbeddedChunk(Chunk("RAG retrieves Markdown chunks.", {"source": "documents/guide.md"}), [1.0, 0.0])],
        ids=["documents/guide.md::0"],
    )

    class FakeEncoder:
        def encode(self, sentences, **kwargs):
            return [[1.0, 0.0] for _ in sentences]

    monkeypatch.setattr(embeddings_module, "load_embedding_model", FakeEncoder)
    monkeypatch.setattr(vector_store_module, "open_collection", lambda: collection)
    grounded_calls = []

    def fake_grounded_chat(messages):
        grounded_calls.append(messages)
        return "RAG retrieves Markdown chunks."

    monkeypatch.setattr(answer_module, "chat", fake_grounded_chat)
    responses = iter([
        '{"type": "action", "tool": "search_docs", "arguments": {"query": "What does RAG retrieve?"}}',
        '{"type": "final", "answer": "RAG retrieves Markdown chunks."}',
    ])
    agent_calls = []

    def fake_agent_chat(messages):
        agent_calls.append([message.copy() for message in messages])
        return next(responses)

    monkeypatch.setattr(agent_module, "chat", fake_agent_chat)
    assert agent_module.run_agent("What does the documentation say RAG retrieves?") == (
        "RAG retrieves Markdown chunks."
    )
    assert "[Source: documents/guide.md]" in grounded_calls[0][1]["content"]
    assert "RAG retrieves Markdown chunks." in grounded_calls[0][1]["content"]
    assert "Tool search_docs returned: RAG retrieves Markdown chunks." in agent_calls[1][-1]["content"]
