"""Tests for the Milestone 10 ingestion orchestration."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
import pytest

from rag.embeddings import EmbeddingError
from rag.ingest import ingest_documents


class FakeEncoder:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append((sentences, kwargs))
        if self.error is not None:
            raise self.error
        return [[1.0, float(index + 1)] for index, _ in enumerate(sentences)]


@pytest.fixture
def collection() -> Collection:
    return chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False)
    ).create_collection(
        name=f"test_{uuid4().hex}",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def test_ingestion_composes_real_loader_chunker_embedding_and_store(
    tmp_path: Path, collection: Collection
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("abcdefghij", encoding="utf-8")
    encoder = FakeEncoder()

    ids = ingest_documents(
        collection,
        model=encoder,
        project_root=tmp_path,
        chunk_size=6,
        chunk_overlap=2,
    )

    assert ids == ["documents/guide.md::0", "documents/guide.md::1"]
    assert encoder.calls[0][0] == ["abcdef", "efghij"]
    assert encoder.calls[0][1] == {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    saved = collection.get(ids=ids, include=["documents", "metadatas"])
    assert saved["documents"] == ["abcdef", "efghij"]
    assert saved["metadatas"] == [
        {
            "source": "documents/guide.md",
            "filename": "guide.md",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 6,
        },
        {
            "source": "documents/guide.md",
            "filename": "guide.md",
            "chunk_index": 1,
            "start_char": 4,
            "end_char": 10,
        },
    ]


def test_reingestion_uses_stable_ids_and_replaces_existing_records(
    tmp_path: Path, collection: Collection
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    source = documents / "guide.md"
    source.write_text("first", encoding="utf-8")
    encoder = FakeEncoder()

    first_ids = ingest_documents(collection, model=encoder, project_root=tmp_path)
    source.write_text("changed", encoding="utf-8")
    second_ids = ingest_documents(collection, model=encoder, project_root=tmp_path)

    assert first_ids == second_ids == ["documents/guide.md::0"]
    assert collection.count() == 1
    assert collection.get(ids=second_ids)["documents"] == ["changed"]


def test_obsolete_tail_records_are_not_deleted(
    tmp_path: Path, collection: Collection
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    source = documents / "guide.md"
    source.write_text("abcdefghij", encoding="utf-8")
    encoder = FakeEncoder()
    ingest_documents(
        collection, model=encoder, project_root=tmp_path, chunk_size=6, chunk_overlap=2
    )
    source.write_text("short", encoding="utf-8")

    ids = ingest_documents(
        collection, model=encoder, project_root=tmp_path, chunk_size=6, chunk_overlap=2
    )

    assert ids == ["documents/guide.md::0"]
    assert collection.count() == 2


@pytest.mark.parametrize("content", [None, ""])
def test_no_markdown_or_empty_markdown_is_a_no_op(
    tmp_path: Path, collection: Collection, content: str | None
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    if content is not None:
        (documents / "empty.md").write_text(content, encoding="utf-8")
    encoder = FakeEncoder(error=AssertionError("encoder must not be called"))

    assert ingest_documents(collection, model=encoder, project_root=tmp_path) == []
    assert encoder.calls == []
    assert collection.count() == 0


def test_loader_and_chunk_validation_errors_propagate(
    tmp_path: Path, collection: Collection
) -> None:
    encoder = FakeEncoder()
    with pytest.raises(FileNotFoundError):
        ingest_documents(collection, model=encoder, project_root=tmp_path)
    with pytest.raises(ValueError, match="traversal"):
        ingest_documents(collection, model=encoder, project_root=tmp_path, path="../docs")
    (tmp_path / "documents").mkdir()
    with pytest.raises(ValueError, match="chunk_overlap"):
        ingest_documents(
            collection,
            model=encoder,
            project_root=tmp_path,
            chunk_size=10,
            chunk_overlap=10,
        )


def test_embedding_failure_preserves_cause(
    tmp_path: Path, collection: Collection
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text("content", encoding="utf-8")
    failure = RuntimeError("inference failed")

    with pytest.raises(EmbeddingError) as error:
        ingest_documents(
            collection,
            model=FakeEncoder(error=failure),
            project_root=tmp_path,
        )

    assert error.value.__cause__ is failure
    assert collection.count() == 0
