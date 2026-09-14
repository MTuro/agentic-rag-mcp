"""Tests for the thin standalone RAG command-line adapters."""

from pathlib import Path

from rag.chunking import Chunk
import rag.ingest as ingest_cli
import rag.query as query_cli
from rag.vector_store import SearchResult


def test_ingest_cli_passes_arguments_and_prints_count(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    model = object()
    collection = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(ingest_cli, "load_embedding_model", lambda: model)
    monkeypatch.setattr(ingest_cli, "open_collection", lambda: collection)

    def fake_ingest(received_collection, **kwargs):
        captured.update(collection=received_collection, **kwargs)
        return ["a", "b"]

    monkeypatch.setattr(ingest_cli, "ingest_documents", fake_ingest)
    ingest_cli.main(["manuals", "--chunk-size", "120", "--chunk-overlap", "20"])

    assert captured == {
        "collection": collection,
        "model": model,
        "path": "manuals",
        "chunk_size": 120,
        "chunk_overlap": 20,
    }
    assert capsys.readouterr().out == "Indexed chunks: 2\n"


def test_query_cli_prints_ranked_evidence(monkeypatch, capsys) -> None:
    model = object()
    collection = object()
    monkeypatch.setattr(query_cli, "load_embedding_model", lambda: model)
    monkeypatch.setattr(query_cli, "open_collection", lambda: collection)
    monkeypatch.setattr(
        query_cli,
        "retrieve_chunks",
        lambda *args, **kwargs: [
            SearchResult(
                "documents/guide.md::0",
                Chunk("retrieved text", {"source": "documents/guide.md", "chunk_index": 0}),
                0.125,
            )
        ],
    )

    query_cli.main(["How does retrieval work?", "--top-k", "1"])

    output = capsys.readouterr().out
    assert "[1] source=documents/guide.md distance=0.125000" in output
    assert "chunk_index" in output
    assert "retrieved text" in output


def test_query_cli_prints_clear_empty_result(monkeypatch, capsys) -> None:
    monkeypatch.setattr(query_cli, "load_embedding_model", object)
    monkeypatch.setattr(query_cli, "open_collection", object)
    monkeypatch.setattr(query_cli, "retrieve_chunks", lambda *args, **kwargs: [])

    query_cli.main(["unknown topic"])

    assert capsys.readouterr().out == "No matching chunks found.\n"
