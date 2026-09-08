"""Tests for local chunk embeddings and semantic vector comparison."""

import math
from typing import Any

import numpy as np
import pytest

import rag.embeddings as embeddings_module
from rag.chunking import Chunk
from rag.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddedChunk,
    EmbeddingError,
    cosine_similarity,
    embed_chunks,
    load_embedding_model,
)


class FakeEncoder:
    """Record batch calls and return a configured fake model result."""

    def __init__(self, result: object = None, *, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def encode(self, sentences: list[str], **kwargs: Any) -> object:
        self.calls.append((sentences, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def _chunk(content: str, index: int) -> Chunk:
    return Chunk(
        content=content,
        metadata={
            "source": "documents/example.md",
            "filename": "example.md",
            "chunk_index": index,
            "start_char": index * 10,
            "end_char": index * 10 + len(content),
        },
    )


def test_default_embedding_model_is_exact_required_model() -> None:
    assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_load_embedding_model_uses_fixed_model_without_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    expected_model = object()

    def fake_sentence_transformer(model_name: str, **kwargs: object) -> object:
        calls.append((model_name, kwargs))
        return expected_model

    monkeypatch.setattr(
        embeddings_module, "SentenceTransformer", fake_sentence_transformer
    )

    assert load_embedding_model() is expected_model
    assert calls == [(DEFAULT_EMBEDDING_MODEL, {"trust_remote_code": False})]


def test_load_embedding_model_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_to_load(*args: object, **kwargs: object) -> object:
        raise OSError("model cache unavailable")

    monkeypatch.setattr(embeddings_module, "SentenceTransformer", fail_to_load)

    with pytest.raises(EmbeddingError, match="Could not load embedding model") as error:
        load_embedding_model()

    assert isinstance(error.value.__cause__, OSError)


def test_embed_chunks_batches_content_and_preserves_chunks_and_order() -> None:
    first = _chunk("first text", 0)
    second = _chunk("second text", 1)
    original_first_metadata = first.metadata.copy()
    original_second_metadata = second.metadata.copy()
    encoder = FakeEncoder(np.array([[1, 2], [3.5, 4]], dtype=np.float32))

    result = embed_chunks([first, second], model=encoder)

    assert encoder.calls == [
        (
            ["first text", "second text"],
            {
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            },
        )
    ]
    assert result == [
        EmbeddedChunk(chunk=first, vector=[1.0, 2.0]),
        EmbeddedChunk(chunk=second, vector=[3.5, 4.0]),
    ]
    assert result[0].chunk is first
    assert result[1].chunk is second
    assert first.content == "first text"
    assert second.content == "second text"
    assert first.metadata == original_first_metadata
    assert second.metadata == original_second_metadata
    assert all(type(value) is float for item in result for value in item.vector)


def test_embed_chunks_returns_empty_without_calling_encoder() -> None:
    encoder = FakeEncoder(error=AssertionError("encode must not be called"))

    assert embed_chunks([], model=encoder) == []
    assert encoder.calls == []


def test_embed_chunks_rejects_non_chunk_input() -> None:
    encoder = FakeEncoder([[1.0]])

    with pytest.raises(TypeError, match=r"chunks\[1\] must be a Chunk"):
        embed_chunks([_chunk("valid", 0), "invalid"], model=encoder)  # type: ignore[list-item]

    assert encoder.calls == []


def test_embed_chunks_reports_encoding_failure() -> None:
    encoder = FakeEncoder(error=RuntimeError("inference failed"))

    with pytest.raises(EmbeddingError, match="Could not encode") as error:
        embed_chunks([_chunk("text", 0)], model=encoder)

    assert isinstance(error.value.__cause__, RuntimeError)


def test_embed_chunks_rejects_incorrect_row_count() -> None:
    encoder = FakeEncoder([[1.0, 2.0]])

    with pytest.raises(EmbeddingError, match="unexpected number of vectors"):
        embed_chunks([_chunk("first", 0), _chunk("second", 1)], model=encoder)


def test_embed_chunks_rejects_empty_vectors() -> None:
    with pytest.raises(EmbeddingError, match="must be non-empty"):
        embed_chunks([_chunk("text", 0)], model=FakeEncoder([[]]))


def test_embed_chunks_rejects_inconsistent_dimensions() -> None:
    encoder = FakeEncoder([[1.0, 2.0], [3.0]])

    with pytest.raises(EmbeddingError, match="inconsistent dimensions"):
        embed_chunks([_chunk("first", 0), _chunk("second", 1)], model=encoder)


@pytest.mark.parametrize("invalid_value", ["1", True, math.inf, -math.inf, math.nan])
def test_embed_chunks_rejects_invalid_numeric_values(invalid_value: object) -> None:
    with pytest.raises(EmbeddingError, match="non-numeric|non-finite"):
        embed_chunks(
            [_chunk("text", 0)], model=FakeEncoder([[1.0, invalid_value]])
        )


def test_cosine_similarity_for_identical_vectors() -> None:
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_for_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_for_opposite_vectors() -> None:
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_cosine_similarity_rejects_unequal_dimensions() -> None:
    with pytest.raises(ValueError, match="equal dimensions"):
        cosine_similarity([1.0], [1.0, 2.0])


@pytest.mark.parametrize(("left", "right"), [([], []), ([], [1.0])])
def test_cosine_similarity_rejects_empty_vectors(
    left: list[float], right: list[float]
) -> None:
    with pytest.raises(ValueError, match="empty|equal dimensions"):
        cosine_similarity(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [([0.0, 0.0], [1.0, 1.0]), ([1.0, 1.0], [0.0, 0.0])],
)
def test_cosine_similarity_rejects_zero_magnitude_vectors(
    left: list[float], right: list[float]
) -> None:
    with pytest.raises(ValueError, match="zero-magnitude"):
        cosine_similarity(left, right)


@pytest.mark.parametrize("invalid_value", [math.inf, -math.inf, math.nan])
def test_cosine_similarity_rejects_non_finite_values(invalid_value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity([1.0, invalid_value], [1.0, 2.0])
