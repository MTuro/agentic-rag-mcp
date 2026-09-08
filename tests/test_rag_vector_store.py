"""Deterministic tests using real local Chroma, without model downloads."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
import pytest

from rag.chunking import Chunk
from rag.embeddings import EmbeddedChunk, embed_chunks
import rag.vector_store as store
from rag.vector_store import SearchResult, VectorStoreError, open_collection, query_chunks, upsert_chunks


@pytest.fixture
def collection() -> Collection:
    return chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False)
    ).create_collection(
        name=f"test_{uuid4().hex}",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def embedded(text: str = "Markdown text", vector: list[float] | None = None) -> EmbeddedChunk:
    return EmbeddedChunk(
        Chunk(text, {"source": "documents/example.md", "chunk_index": 0}),
        [1.0, 0.0] if vector is None else vector,
    )


def test_round_trip_and_inputs_unchanged(collection: Collection) -> None:
    item = embedded()
    before = deepcopy(item)
    ids = ["chunk-1"]
    upsert_chunks(collection, iter([item]), ids=ids)
    saved = collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
    assert saved["ids"] == ids
    assert saved["documents"] == [item.chunk.content]
    assert saved["metadatas"] == [item.chunk.metadata]
    assert saved["embeddings"][0] == pytest.approx(item.vector)
    assert item == before
    assert ids == ["chunk-1"]
    result = query_chunks(collection, [1.0, 0.0])[0]
    assert result == SearchResult("chunk-1", item.chunk, pytest.approx(0.0))
    with pytest.raises(FrozenInstanceError):
        result.distance = 99


def test_top_k_orders_by_cosine_distance(collection: Collection) -> None:
    items = [embedded("far", [0.0, 1.0]), embedded("near", [0.8, 0.6]), embedded("exact")]
    upsert_chunks(collection, items, ids=["far", "near", "exact"])
    result = query_chunks(collection, [1.0, 0.0], top_k=2)
    assert [hit.chunk_id for hit in result] == ["exact", "near"]
    assert [hit.distance for hit in result] == pytest.approx([0.0, 0.2])
    assert len(query_chunks(collection, [1.0, 0.0], top_k=20)) == 3


def test_upsert_replaces_text_metadata_and_vector_without_increasing_count(collection: Collection) -> None:
    old = EmbeddedChunk(Chunk("old", {"obsolete": 1, "version": 1}), [1.0, 0.0])
    new = EmbeddedChunk(Chunk("new", {"version": 2}), [0.0, 1.0])
    original = deepcopy(new)
    upsert_chunks(collection, [old], ids=["same"])
    upsert_chunks(collection, [new], ids=["same"])
    assert collection.count() == 1
    saved = collection.get(ids=["same"], include=["documents", "metadatas", "embeddings"])
    assert saved["documents"] == ["new"]
    assert saved["metadatas"] == [{"version": 2}]
    assert saved["embeddings"][0] == pytest.approx([0.0, 1.0])
    assert query_chunks(collection, [0.0, 1.0])[0].distance == pytest.approx(0.0)
    assert new == original


def test_duplicate_batch_ids_do_not_write(collection: Collection) -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        upsert_chunks(collection, [embedded(), embedded()], ids=["same", "same"])
    assert collection.count() == 0


@pytest.mark.parametrize(("items", "ids"), [([], ["one"]), ([embedded()], []), ([embedded()], ["a", "b"])])
def test_mismatched_lengths_do_not_write(collection: Collection, items: list, ids: list) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        upsert_chunks(collection, items, ids=ids)
    assert collection.count() == 0


def test_empty_write_and_empty_query(collection: Collection) -> None:
    upsert_chunks(collection, [], ids=[])
    assert collection.count() == 0
    assert query_chunks(collection, [1.0, 0.0]) == []


@pytest.mark.parametrize(("value", "error"), [(0, ValueError), (-1, ValueError), (1.5, TypeError), ("2", TypeError), (True, TypeError), (False, TypeError)])
def test_invalid_top_k_on_empty_collection(collection: Collection, value: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="top_k"):
        query_chunks(collection, [1.0, 0.0], top_k=value)


@pytest.mark.parametrize(("vector", "error"), [
    ([], ValueError), ([0.0, 0.0], ValueError), (["x", 1.0], TypeError),
    ([True, 1.0], TypeError), ([math.nan, 1.0], ValueError),
    ([math.inf, 1.0], ValueError), ([-math.inf, 1.0], ValueError),
    (None, TypeError), ("12", TypeError),
])
def test_invalid_vectors_rejected_before_write_and_on_empty_query(collection: Collection, vector: object, error: type[Exception]) -> None:
    item = EmbeddedChunk(Chunk("text", {"source": "test"}), vector)
    with pytest.raises(error):
        upsert_chunks(collection, [embedded(), item], ids=["valid", "invalid"])
    assert collection.count() == 0
    with pytest.raises(error):
        query_chunks(collection, vector)


def test_inconsistent_batch_dimensions_do_not_write(collection: Collection) -> None:
    with pytest.raises(ValueError, match="consistent dimensions"):
        upsert_chunks(collection, [embedded(), embedded(vector=[1.0])], ids=["a", "b"])
    assert collection.count() == 0


def test_incompatible_stored_dimension_is_reported(collection: Collection) -> None:
    upsert_chunks(collection, [embedded()], ids=["a"])
    with pytest.raises(VectorStoreError, match="query.*[Dd]imension") as error:
        query_chunks(collection, [1.0])
    assert error.value.__cause__ is not None
    with pytest.raises(VectorStoreError, match="upsert.*[Dd]imension"):
        upsert_chunks(collection, [embedded(vector=[1.0])], ids=["b"])
    assert collection.count() == 1


@pytest.mark.parametrize(("chunk_id", "error"), [("", ValueError), (1, TypeError), (True, TypeError), (None, TypeError)])
def test_invalid_ids_do_not_write(collection: Collection, chunk_id: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        upsert_chunks(collection, [embedded()], ids=[chunk_id])
    assert collection.count() == 0


@pytest.mark.parametrize("ids", ["a", None, 1])
def test_invalid_ids_container(collection: Collection, ids: object) -> None:
    with pytest.raises(TypeError):
        upsert_chunks(collection, [embedded()], ids=ids)


@pytest.mark.parametrize(("item", "error"), [
    ("not embedded", TypeError), (Chunk("text", {"x": 1}), TypeError),
    (EmbeddedChunk("not chunk", [1.0, 0.0]), TypeError),
    (EmbeddedChunk(Chunk("", {"x": 1}), [1.0, 0.0]), ValueError),
    (EmbeddedChunk(Chunk(7, {"x": 1}), [1.0, 0.0]), TypeError),
    (EmbeddedChunk(Chunk("text", {}), [1.0, 0.0]), ValueError),
    (EmbeddedChunk(Chunk("text", None), [1.0, 0.0]), TypeError),
    (EmbeddedChunk(Chunk("text", {1: "x"}), [1.0, 0.0]), TypeError),
    (EmbeddedChunk(Chunk("text", {"x": True}), [1.0, 0.0]), TypeError),
    (EmbeddedChunk(Chunk("text", {"x": 1.5}), [1.0, 0.0]), TypeError),
    (EmbeddedChunk(Chunk("text", {"x": None}), [1.0, 0.0]), TypeError),
    (EmbeddedChunk(Chunk("text", {"x": []}), [1.0, 0.0]), TypeError),
])
def test_whole_batch_validation_keeps_existing_records_unchanged(collection: Collection, item: object, error: type[Exception]) -> None:
    upsert_chunks(collection, [embedded("original")], ids=["existing"])
    with pytest.raises(error):
        upsert_chunks(collection, [embedded("replacement"), item], ids=["existing", "bad"])
    assert collection.count() == 1
    assert collection.get(ids=["existing"])["documents"] == ["original"]


def test_open_collection_and_isolation(tmp_path: Path) -> None:
    first = open_collection(project_root=tmp_path)
    second = open_collection(project_root=tmp_path, name="second_collection")
    upsert_chunks(first, [embedded()], ids=["a"])
    assert first.configuration["hnsw"]["space"] == "cosine"
    assert first.configuration["embedding_function"] is None
    assert (tmp_path / "chroma_db").is_dir()
    assert second.count() == 0
    assert open_collection(project_root=tmp_path).count() == 1
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"), settings=Settings(anonymized_telemetry=False))
    assert client.get_settings().anonymized_telemetry is False


def test_incompatible_collection_is_not_changed(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"), settings=Settings(anonymized_telemetry=False))
    existing = client.create_collection("document_chunks", embedding_function=None, configuration={"hnsw": {"space": "l2"}})
    existing.upsert(ids=["original"], embeddings=[[1.0, 0.0]], documents=["original"])
    with pytest.raises(ValueError, match="cosine"):
        open_collection(project_root=tmp_path)
    assert existing.configuration["hnsw"]["space"] == "l2"
    assert existing.get()["documents"] == ["original"]


def test_invalid_database_path(tmp_path: Path) -> None:
    (tmp_path / "chroma_db").write_text("not a directory")
    with pytest.raises(ValueError, match="database path"):
        open_collection(project_root=tmp_path)
    assert (tmp_path / "chroma_db").read_text() == "not a directory"


def test_invalid_root_and_name(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Path"):
        open_collection(project_root=str(tmp_path))
    with pytest.raises(ValueError, match="existing directory"):
        open_collection(project_root=tmp_path / "missing")
    with pytest.raises(TypeError, match="name"):
        open_collection(project_root=tmp_path, name=7)
    with pytest.raises(ValueError, match="empty"):
        open_collection(project_root=tmp_path, name="")


def test_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "chroma_db").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the project root"):
        open_collection(project_root=root)
    assert list(outside.iterdir()) == []


def test_storage_access_failure_preserves_cause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failure = PermissionError("storage denied")
    def fail(**kwargs: object) -> None:
        raise failure
    monkeypatch.setattr(store.chromadb, "PersistentClient", fail)
    with pytest.raises(VectorStoreError, match="open Chroma") as error:
        open_collection(project_root=tmp_path)
    assert error.value.__cause__ is failure


@pytest.mark.parametrize("operation", ["upsert", "get", "query", "count"])
def test_operational_failure_preserves_cause(collection: Collection, monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    upsert_chunks(collection, [embedded()], ids=["a"])
    failure = RuntimeError("database unavailable")
    def fail(*args: object, **kwargs: object) -> None:
        raise failure
    monkeypatch.setattr(Collection, operation, fail)
    expected_operation = "upsert" if operation in ("upsert", "get") else "query"
    with pytest.raises(VectorStoreError, match=expected_operation) as error:
        if expected_operation == "upsert":
            upsert_chunks(collection, [embedded()], ids=["a"])
        else:
            query_chunks(collection, [1.0, 0.0])
    assert error.value.__cause__ is failure


def test_explicit_vectors_without_embedding_function(collection: Collection) -> None:
    assert collection.configuration["embedding_function"] is None
    upsert_chunks(collection, [embedded()], ids=["a"])
    assert query_chunks(collection, [1.0, 0.0])[0].chunk_id == "a"
    # Without a configured encoder, a text-only query cannot silently embed.
    with pytest.raises(ValueError, match="embedding function"):
        collection.query(query_texts=["text"])


def test_milestone_8_output_integrates_with_real_chroma(collection: Collection) -> None:
    class FakeEncoder:
        def encode(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
            assert sentences == ["documentation", "bicycle"]
            assert kwargs["normalize_embeddings"] is True
            return [[1.0, 0.0], [0.0, 1.0]]
    chunks = [Chunk("documentation", {"source": "docs.md", "start_char": 0}), Chunk("bicycle", {"source": "bike.md", "start_char": 0})]
    items = embed_chunks(chunks, model=FakeEncoder())
    upsert_chunks(collection, items, ids=["docs", "bike"])
    result = query_chunks(collection, items[0].vector, top_k=1)[0]
    assert result.chunk_id == "docs"
    assert result.chunk == chunks[0]
    assert result.distance == pytest.approx(0.0)


def test_persistence_across_separate_processes(tmp_path: Path) -> None:
    # Both processes use fixed test code and the same interpreter as pytest.
    # The reader does not call upsert or an embedding model.
    writer = """
from pathlib import Path
import sys
from rag.chunking import Chunk
from rag.embeddings import EmbeddedChunk
from rag.vector_store import open_collection, upsert_chunks
c = open_collection(project_root=Path(sys.argv[1]))
upsert_chunks(c, [EmbeddedChunk(Chunk('persisted text', {'source': 'docs.md', 'index': 3}), [1., 0.])], ids=['persisted'])
"""
    reader = """
from dataclasses import asdict
from pathlib import Path
import json
import sys
from rag.vector_store import open_collection, query_chunks
c = open_collection(project_root=Path(sys.argv[1]))
assert c.count() == 1
print(json.dumps([asdict(hit) for hit in query_chunks(c, [1., 0.])]))
"""
    cwd = Path(__file__).resolve().parents[1]
    written = subprocess.run([sys.executable, "-c", writer, str(tmp_path)], cwd=cwd, capture_output=True, text=True, timeout=90)
    assert written.returncode == 0, written.stderr
    read = subprocess.run([sys.executable, "-c", reader, str(tmp_path)], cwd=cwd, capture_output=True, text=True, timeout=90)
    assert read.returncode == 0, read.stderr
    hits = json.loads(read.stdout)
    assert hits == [{"chunk_id": "persisted", "chunk": {"content": "persisted text", "metadata": {"source": "docs.md", "index": 3}}, "distance": pytest.approx(0.0)}]
