"""Split loaded documents into deterministic overlapping text chunks."""

from collections.abc import Iterable
from dataclasses import dataclass

from rag.loader import Document


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


@dataclass(frozen=True)
class Chunk:
    """One text window and metadata tracing it back to its source."""

    content: str
    metadata: dict[str, str | int]


def _validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    """Require integer values that always advance the chunking window."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if type(chunk_overlap) is not int or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[str, int, int]]:
    """Return character windows as content, inclusive start, and exclusive end."""
    _validate_chunk_parameters(chunk_size, chunk_overlap)
    if not text:
        return []

    chunks: list[tuple[str, int, int]] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((text[start:end], start, end))
        if end == len(text):
            break
        start = end - chunk_overlap

    return chunks


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk documents while preserving source metadata and character offsets."""
    _validate_chunk_parameters(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []

    for document in documents:
        for chunk_index, (content, start, end) in enumerate(
            chunk_text(
                document.content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ):
            chunks.append(
                Chunk(
                    content=content,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "end_char": end,
                    },
                )
            )

    return chunks
