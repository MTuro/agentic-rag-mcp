"""Orchestrate Markdown loading, chunking, embedding, and storage."""

import argparse
from pathlib import Path
from collections.abc import Sequence

from chromadb.api.models.Collection import Collection

from config import PROJECT_ROOT
from rag.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_documents
from rag.embeddings import EmbeddingEncoder, embed_chunks, load_embedding_model
from rag.loader import load_markdown_documents
from rag.vector_store import open_collection, upsert_chunks


def _chunk_id(source: str, chunk_index: int) -> str:
    """Build the stable, human-readable identity of one source chunk."""
    return f"{source}::{chunk_index}"


def ingest_documents(
    collection: Collection,
    *,
    model: EmbeddingEncoder,
    path: str = "documents",
    project_root: Path = PROJECT_ROOT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Load and index Markdown chunks, returning IDs written by this run."""
    documents = load_markdown_documents(path, project_root=project_root)
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embedded_chunks = embed_chunks(chunks, model=model)
    ids = [
        _chunk_id(str(chunk.metadata["source"]), int(chunk.metadata["chunk_index"]))
        for chunk in chunks
    ]
    upsert_chunks(collection, embedded_chunks, ids=ids)
    return ids


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index project Markdown in ChromaDB.")
    parser.add_argument("path", nargs="?", default="documents")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run standalone ingestion against the default persistent collection."""
    args = _build_parser().parse_args(argv)
    model = load_embedding_model()
    collection = open_collection()
    ids = ingest_documents(
        collection,
        model=model,
        path=args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Indexed chunks: {len(ids)}")


if __name__ == "__main__":
    main()
