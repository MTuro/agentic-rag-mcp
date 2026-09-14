"""Tests for embedding questions and retrieving stored evidence."""

from typing import Any
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
import pytest

from rag.chunking import Chunk
from rag.embeddings import EmbeddedChunk, EmbeddingError
from rag.ingest import ingest_documents
from rag.retriever import retrieve_chunks
from rag.vector_store import upsert_chunks


class QueryEncoder:
    def __init__(self, vector: list[float] | None = None, error: Exception | None = None) -> None:
        self.vector = [1.0, 0.0] if vector is None else vector
        self.error = error
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append((sentences, kwargs))
        if self.error is not None:
            raise self.error
        return [self.vector]


@pytest.fixture
def collection() -> Collection:
    return chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False)
    ).create_collection(
        name=f"test_{uuid4().hex}",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def _populate(collection: Collection) -> None:
    upsert_chunks(
        collection,
        [
            EmbeddedChunk(Chunk("documentation", {"source": "documents/docs.md"}), [1.0, 0.0]),
            EmbeddedChunk(Chunk("bicycle", {"source": "documents/bike.md"}), [0.0, 1.0]),
        ],
        ids=["docs", "bike"],
    )


def test_question_is_embedded_once_and_known_vector_ranks_evidence(
    collection: Collection,
) -> None:
    _populate(collection)
    encoder = QueryEncoder()

    results = retrieve_chunks("How are documents loaded?", collection, model=encoder)

    assert encoder.calls == [
        (
            ["How are documents loaded?"],
            {
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            },
        )
    ]
    assert [result.chunk_id for result in results] == ["docs", "bike"]
    assert results[0].chunk.content == "documentation"
    assert results[0].chunk.metadata == {"source": "documents/docs.md"}
    assert results[0].distance == pytest.approx(0.0)


def test_top_k_and_empty_collection(collection: Collection) -> None:
    encoder = QueryEncoder()
    assert retrieve_chunks("question", collection, model=encoder, top_k=20) == []
    _populate(collection)
    assert len(retrieve_chunks("question", collection, model=encoder, top_k=1)) == 1
    assert len(retrieve_chunks("question", collection, model=encoder, top_k=20)) == 2


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_is_rejected_before_embedding(
    collection: Collection, question: str
) -> None:
    encoder = QueryEncoder(error=AssertionError("must not encode"))
    with pytest.raises(ValueError, match="question"):
        retrieve_chunks(question, collection, model=encoder)
    assert encoder.calls == []


@pytest.mark.parametrize("question", [None, 7, True])
def test_non_string_question_is_rejected(
    collection: Collection, question: object
) -> None:
    with pytest.raises(TypeError, match="question"):
        retrieve_chunks(question, collection, model=QueryEncoder())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("top_k", "error"),
    [(0, ValueError), (-1, ValueError), (1.5, TypeError), ("2", TypeError), (True, TypeError)],
)
def test_invalid_top_k_is_rejected_before_embedding(
    collection: Collection, top_k: object, error: type[Exception]
) -> None:
    encoder = QueryEncoder(error=AssertionError("must not encode"))
    with pytest.raises(error, match="top_k"):
        retrieve_chunks("question", collection, model=encoder, top_k=top_k)  # type: ignore[arg-type]
    assert encoder.calls == []


def test_embedding_failure_preserves_cause(collection: Collection) -> None:
    failure = RuntimeError("inference failed")
    with pytest.raises(EmbeddingError) as error:
        retrieve_chunks("question", collection, model=QueryEncoder(error=failure))
    assert error.value.__cause__ is failure


def test_automatic_text_embedding_remains_disabled(collection: Collection) -> None:
    assert collection.configuration["embedding_function"] is None
    with pytest.raises(ValueError, match="embedding function"):
        collection.query(query_texts=["question"])


def test_deterministic_end_to_end_ingestion_and_retrieval(
    tmp_path, collection: Collection
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "architecture.md").write_text(
        "Markdown documents are loaded and divided into chunks.", encoding="utf-8"
    )
    (documents / "cycling.md").write_text(
        "A bicycle has two wheels and pedals.", encoding="utf-8"
    )

    class KeywordEncoder:
        def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
            assert kwargs["normalize_embeddings"] is True
            return [
                [1.0, 0.0] if "document" in text.casefold() else [0.0, 1.0]
                for text in sentences
            ]

    model = KeywordEncoder()
    ids = ingest_documents(collection, model=model, project_root=tmp_path)
    results = retrieve_chunks(
        "What happens to documents before retrieval?",
        collection,
        model=model,
        top_k=1,
    )

    assert ids == ["documents/architecture.md::0", "documents/cycling.md::0"]
    assert results[0].chunk_id == "documents/architecture.md::0"
    assert results[0].chunk.metadata == {
        "source": "documents/architecture.md",
        "filename": "architecture.md",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 54,
    }
    ingest_documents(collection, model=model, project_root=tmp_path)
    assert collection.count() == 2
