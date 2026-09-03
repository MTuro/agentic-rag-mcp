"""Tests for deterministic document chunking with overlap and metadata."""

import pytest

from rag.chunking import Chunk, chunk_documents, chunk_text
from rag.loader import Document


def test_chunk_text_returns_one_chunk_for_short_text() -> None:
    assert chunk_text("short", chunk_size=10, chunk_overlap=2) == [
        ("short", 0, 5)
    ]


def test_chunk_text_returns_one_chunk_for_exact_size() -> None:
    assert chunk_text("12345", chunk_size=5, chunk_overlap=2) == [
        ("12345", 0, 5)
    ]


def test_chunk_text_returns_exact_overlapping_windows() -> None:
    assert chunk_text("abcdefghijk", chunk_size=5, chunk_overlap=2) == [
        ("abcde", 0, 5),
        ("defgh", 3, 8),
        ("ghijk", 6, 11),
    ]


def test_chunk_text_supports_zero_overlap() -> None:
    assert chunk_text("abcdefgh", chunk_size=3, chunk_overlap=0) == [
        ("abc", 0, 3),
        ("def", 3, 6),
        ("gh", 6, 8),
    ]


def test_chunk_text_returns_no_chunks_for_empty_text() -> None:
    assert chunk_text("", chunk_size=10, chunk_overlap=2) == []


def test_chunk_documents_preserves_order_metadata_and_offsets() -> None:
    first_metadata = {"source": "documents/first.md", "filename": "first.md"}
    documents = [
        Document(content="abcdef", metadata=first_metadata),
        Document(
            content="xyz",
            metadata={"source": "documents/second.md", "filename": "second.md"},
        ),
    ]

    chunks = chunk_documents(documents, chunk_size=4, chunk_overlap=1)

    assert chunks == [
        Chunk(
            content="abcd",
            metadata={
                **first_metadata,
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 4,
            },
        ),
        Chunk(
            content="def",
            metadata={
                **first_metadata,
                "chunk_index": 1,
                "start_char": 3,
                "end_char": 6,
            },
        ),
        Chunk(
            content="xyz",
            metadata={
                "source": "documents/second.md",
                "filename": "second.md",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 3,
            },
        ),
    ]
    assert first_metadata == {
        "source": "documents/first.md",
        "filename": "first.md",
    }


def test_chunk_documents_skips_empty_documents() -> None:
    document = Document(
        content="",
        metadata={"source": "documents/empty.md", "filename": "empty.md"},
    )

    assert chunk_documents([document], chunk_size=10, chunk_overlap=2) == []


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5, "10"])
def test_chunk_text_rejects_invalid_chunk_size(chunk_size: object) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("text", chunk_size=chunk_size, chunk_overlap=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("chunk_overlap", [-1, True, 1.5, "1"])
def test_chunk_text_rejects_invalid_chunk_overlap(chunk_overlap: object) -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("text", chunk_size=10, chunk_overlap=chunk_overlap)  # type: ignore[arg-type]


@pytest.mark.parametrize("chunk_overlap", [5, 6])
def test_chunk_text_rejects_overlap_not_smaller_than_size(chunk_overlap: int) -> None:
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        chunk_text("text", chunk_size=5, chunk_overlap=chunk_overlap)


def test_chunk_documents_validates_parameters_for_empty_input() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_documents([], chunk_size=0)
