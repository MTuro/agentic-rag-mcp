"""Persist supplied chunk embeddings and search them with local ChromaDB."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from config import PROJECT_ROOT
from rag.chunking import Chunk
from rag.embeddings import (
    EmbeddedChunk,
    cosine_similarity,
    embed_chunks,
    load_embedding_model,
)
from tools.filesystem import _resolve_project_path


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk; smaller cosine distance means a closer vector."""

    chunk_id: str
    chunk: Chunk
    distance: float


class VectorStoreError(RuntimeError):
    """Report a Chroma operation failure while preserving its cause."""


def open_collection(
    *,
    project_root: Path = PROJECT_ROOT,
    name: str = "document_chunks",
) -> Collection:
    """Open a local cosine collection without automatic embedding or telemetry."""
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be a Path.")
    if not isinstance(name, str):
        raise TypeError("Collection name must be a string.")
    if not name:
        raise ValueError("Collection name must not be empty.")
    directory = _resolve_project_path("chroma_db", project_root=project_root)
    if not project_root.is_dir():
        raise ValueError("project_root must be an existing directory.")
    if directory.exists() and not directory.is_dir():
        raise ValueError("Chroma database path must be a directory.")

    try:
        client = chromadb.PersistentClient(
            path=str(directory), settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_or_create_collection(
            name=name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
        configuration = collection.configuration
    except Exception as exc:
        raise VectorStoreError(f"Could not open Chroma collection {name!r}: {exc}") from exc

    if (configuration.get("hnsw") or {}).get("space") != "cosine":
        raise ValueError("Existing collection must use cosine distance; it was not changed.")
    return collection


def _validate_vector(vector: Sequence[float]) -> None:
    """Reuse Milestone 8 validation without generating or changing a vector."""
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise TypeError("Vector must be a numeric sequence.")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in vector):
        raise TypeError("Vector values must be numeric, not booleans.")
    cosine_similarity(vector, vector)


def upsert_chunks(
    collection: Collection,
    embedded_chunks: Iterable[EmbeddedChunk],
    *,
    ids: Sequence[str],
) -> None:
    """Validate the whole batch, then insert or replace caller-identified chunks."""
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise TypeError("ids must be a sequence of strings.")
    items = list(embedded_chunks)
    chunk_ids = list(ids)
    if len(items) != len(chunk_ids):
        raise ValueError("IDs and embedded chunks must have equal lengths.")

    seen: set[str] = set()
    dimension: int | None = None
    for chunk_id, item in zip(chunk_ids, items, strict=True):
        if not isinstance(chunk_id, str):
            raise TypeError("Each chunk ID must be a string.")
        if not chunk_id:
            raise ValueError("Chunk IDs must not be empty.")
        if chunk_id in seen:
            raise ValueError("Duplicate chunk IDs in one batch are not allowed.")
        seen.add(chunk_id)
        if not isinstance(item, EmbeddedChunk):
            raise TypeError("Each item must be an EmbeddedChunk.")
        if not isinstance(item.chunk, Chunk):
            raise TypeError("EmbeddedChunk.chunk must be a Chunk.")
        if not isinstance(item.chunk.content, str):
            raise TypeError("Chunk text must be a string.")
        if not item.chunk.content:
            raise ValueError("Chunk text must not be empty.")
        metadata = item.chunk.metadata
        if not isinstance(metadata, dict):
            raise TypeError("Chunk metadata must be a dictionary.")
        if not metadata:
            raise ValueError("Chunk metadata must not be empty.")
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise TypeError("Metadata keys must be strings.")
            if type(value) not in (str, int):
                raise TypeError("Metadata values must be strings or integers, not booleans.")
        _validate_vector(item.vector)
        if dimension is not None and len(item.vector) != dimension:
            raise ValueError("Embedding vectors must have consistent dimensions.")
        dimension = len(item.vector)

    if not items:
        return

    try:
        # Chroma merges metadata on upsert. Null out omitted old keys so our
        # explicit replacement contract also holds when a key disappears.
        previous = collection.get(ids=chunk_ids, include=["metadatas"])
        previous_metadata = dict(zip(previous["ids"], previous["metadatas"], strict=True))
        metadatas = []
        for chunk_id, item in zip(chunk_ids, items, strict=True):
            metadata = dict(item.chunk.metadata)
            removed_keys = (previous_metadata.get(chunk_id) or {}).keys() - metadata.keys()
            metadatas.append({**dict.fromkeys(removed_keys), **metadata})
        collection.upsert(
            ids=chunk_ids,
            documents=[item.chunk.content for item in items],
            metadatas=metadatas,
            embeddings=[list(item.vector) for item in items],
        )
    except Exception as exc:
        raise VectorStoreError(f"Could not upsert Chroma chunks: {exc}") from exc


def query_chunks(
    collection: Collection,
    query_vector: Sequence[float],
    *,
    top_k: int = 3,
) -> list[SearchResult]:
    """Retrieve up to K chunks using a supplied vector, preserving Chroma order."""
    if type(top_k) is not int:
        raise TypeError("top_k must be an integer, not a boolean.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    _validate_vector(query_vector)

    try:
        count = collection.count()
        if count == 0:
            return []
        response = collection.query(
            query_embeddings=[list(query_vector)],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(
                chunk_id=chunk_id,
                chunk=Chunk(content=text, metadata=dict(metadata)),
                distance=float(distance),
            )
            for chunk_id, text, metadata, distance in zip(
                response["ids"][0],
                response["documents"][0],
                response["metadatas"][0],
                response["distances"][0],
                strict=True,
            )
        ]
    except Exception as exc:
        raise VectorStoreError(f"Could not query Chroma chunks: {exc}") from exc


def main() -> None:
    """Demonstrate fixed MiniLM examples, persistent records, and vector search."""
    chunks = [
        Chunk("Python loads Markdown documents.", {"example": 1}),
        Chunk("The application reads Markdown files.", {"example": 2}),
        Chunk("A bicycle has two wheels.", {"example": 3}),
    ]
    embedded = embed_chunks(chunks, model=load_embedding_model())
    collection = open_collection(name="milestone_9_demo")
    upsert_chunks(collection, embedded, ids=["markdown", "related", "bicycle"])

    # Reopen without rewriting. A separate-process test verifies persistence too.
    reopened = open_collection(name="milestone_9_demo")
    results = query_chunks(reopened, embedded[0].vector, top_k=3)
    if [result.chunk_id for result in results] != ["markdown", "related", "bicycle"]:
        raise VectorStoreError("The demonstration did not retrieve the expected neighbors.")
    if not results[1].distance < results[2].distance:
        raise VectorStoreError("The related sentence was not closer than the bicycle.")
    print(f"Stored records: {reopened.count()}; vector dimensions: {len(embedded[0].vector)}")
    print(f"Database: {PROJECT_ROOT / 'chroma_db'}")
    for result in results:
        print(f"{result.chunk_id}: distance={result.distance:.4f}; {result.chunk.content}")


if __name__ == "__main__":
    main()
