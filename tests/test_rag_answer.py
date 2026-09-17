"""Grounded answering without a running Ollama server."""

import pytest

import rag.answer as answer
from rag.chunking import Chunk
from rag.context import build_context
from rag.vector_store import SearchResult


def _result(source: str, content: str) -> SearchResult:
    return SearchResult("chunk", Chunk(content, {"source": source}), 0.1)


def test_context_preserves_order_sources_and_text() -> None:
    results = [
        _result("documents/first.md", "First fact."),
        _result("documents/second.md", "Second fact."),
    ]
    assert build_context(results) == (
        "[Source: documents/first.md]\nFirst fact.\n\n"
        "[Source: documents/second.md]\nSecond fact."
    )
    assert build_context([]) == ""


def test_answer_retrieves_builds_prompt_calls_qwen_and_returns_text(monkeypatch) -> None:
    collection = object()
    model = object()
    calls = {}

    def fake_retrieve(question, received_collection, *, model, top_k):
        calls["retrieval"] = (question, received_collection, model, top_k)
        return [_result("documents/guide.md", "Retrieval finds relevant chunks.")]

    def fake_chat(messages):
        calls["messages"] = messages
        return "It finds relevant chunks."

    monkeypatch.setattr(answer, "retrieve_chunks", fake_retrieve)
    monkeypatch.setattr(answer, "chat", fake_chat)

    result = answer.answer_question("How does retrieval work?", collection, model=model, top_k=2)

    assert result == "It finds relevant chunks."
    assert calls["retrieval"] == ("How does retrieval work?", collection, model, 2)
    messages = calls["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "using only the supplied context" in messages[0]["content"]
    assert "Do not invent" in messages[0]["content"]
    assert "does not contain enough information" in messages[0]["content"]
    assert "[Source: documents/guide.md]\nRetrieval finds relevant chunks." in messages[1]["content"]
    assert "Question: How does retrieval work?" in messages[1]["content"]


def test_empty_evidence_returns_clear_answer_without_model_call(monkeypatch) -> None:
    monkeypatch.setattr(answer, "retrieve_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(answer, "chat", lambda messages: pytest.fail("Qwen must not be called"))

    assert answer.answer_question("Unknown?", object(), model=object()) == answer.NO_EVIDENCE


@pytest.mark.parametrize("question,error", [(None, TypeError), (7, TypeError), ("", ValueError), (" \n", ValueError)])
def test_invalid_question_is_rejected_before_retrieval(monkeypatch, question, error) -> None:
    monkeypatch.setattr(answer, "retrieve_chunks", lambda *args, **kwargs: pytest.fail("retrieval must not run"))

    with pytest.raises(error, match="question"):
        answer.answer_question(question, object(), model=object())


def test_cli_prints_answer(monkeypatch, capsys) -> None:
    model = object()
    collection = object()
    calls = {}
    monkeypatch.setattr(answer, "load_embedding_model", lambda: model)
    monkeypatch.setattr(answer, "open_collection", lambda: collection)

    def fake_answer(question, received_collection, *, model, top_k):
        calls["args"] = (question, received_collection, model, top_k)
        return "Grounded answer."

    monkeypatch.setattr(answer, "answer_question", fake_answer)
    answer.main(["How does retrieval work?", "--top-k", "2"])

    assert calls["args"] == ("How does retrieval work?", collection, model, 2)
    assert capsys.readouterr().out == "Grounded answer.\n"
