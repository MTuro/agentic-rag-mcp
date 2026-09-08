"""Convert text chunks into local semantic vector representations."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Protocol

from sentence_transformers import SentenceTransformer

from rag.chunking import Chunk


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(RuntimeError):
    """Report a failure while loading or running the embedding model."""


@dataclass(frozen=True)
class EmbeddedChunk:
    """An existing chunk paired with its plain-Python embedding vector."""

    chunk: Chunk
    vector: list[float]


class _EmbeddingEncoder(Protocol):
    """The part of SentenceTransformer used by this module."""

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> object: ...


def load_embedding_model() -> SentenceTransformer:
    """Load the fixed local embedding model without trusting remote code."""
    try:
        return SentenceTransformer(
            DEFAULT_EMBEDDING_MODEL,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load embedding model {DEFAULT_EMBEDDING_MODEL!r}."
        ) from exc


def _validate_encoded_matrix(encoded: object, expected_rows: int) -> list[list[float]]:
    """Convert a model matrix to validated, equal-sized Python float lists."""
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()

    if not isinstance(encoded, list):
        raise EmbeddingError("The embedding model returned a non-matrix result.")
    if len(encoded) != expected_rows:
        raise EmbeddingError(
            "The embedding model returned an unexpected number of vectors: "
            f"expected {expected_rows}, received {len(encoded)}."
        )

    vectors: list[list[float]] = []
    expected_dimension: int | None = None

    for row_index, row in enumerate(encoded):
        if not isinstance(row, (list, tuple)) or not row:
            raise EmbeddingError(
                f"Embedding vector at row {row_index} must be non-empty."
            )

        vector: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise EmbeddingError(
                    f"Embedding vector at row {row_index} contains a non-numeric value."
                )
            converted = float(value)
            if not math.isfinite(converted):
                raise EmbeddingError(
                    f"Embedding vector at row {row_index} contains a non-finite value."
                )
            vector.append(converted)

        if expected_dimension is None:
            expected_dimension = len(vector)
        elif len(vector) != expected_dimension:
            raise EmbeddingError("Embedding vectors have inconsistent dimensions.")

        vectors.append(vector)

    return vectors


def embed_chunks(
    chunks: Iterable[Chunk], *, model: _EmbeddingEncoder
) -> list[EmbeddedChunk]:
    """Encode chunks in one batch while preserving their identity and order."""
    materialized_chunks = list(chunks)
    for index, chunk in enumerate(materialized_chunks):
        if not isinstance(chunk, Chunk):
            raise TypeError(f"chunks[{index}] must be a Chunk instance.")

    if not materialized_chunks:
        return []

    try:
        encoded = model.encode(
            [chunk.content for chunk in materialized_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise EmbeddingError("Could not encode text chunks.") from exc

    vectors = _validate_encoded_matrix(encoded, len(materialized_chunks))
    return [
        EmbeddedChunk(chunk=chunk, vector=vector)
        for chunk, vector in zip(materialized_chunks, vectors, strict=True)
    ]


def _validated_similarity_vector(
    vector: Sequence[float], *, name: str
) -> list[float]:
    """Return finite Python floats suitable for cosine similarity."""
    if not vector:
        raise ValueError(f"{name} must not be empty.")

    validated: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain only numeric values.")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must contain only finite values.")
        validated.append(converted)
    return validated


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Measure the cosine of the angle between two equal-sized vectors."""
    if len(left) != len(right):
        raise ValueError("Vectors must have equal dimensions.")

    left_values = _validated_similarity_vector(left, name="left")
    right_values = _validated_similarity_vector(right, name="right")
    left_magnitude = math.sqrt(math.fsum(value * value for value in left_values))
    right_magnitude = math.sqrt(math.fsum(value * value for value in right_values))

    if left_magnitude == 0.0 or right_magnitude == 0.0:
        raise ValueError("Cosine similarity is undefined for a zero-magnitude vector.")

    dot_product = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    similarity = dot_product / (left_magnitude * right_magnitude)
    return max(-1.0, min(1.0, similarity))


def main() -> None:
    """Demonstrate that related sentences receive more similar embeddings."""
    chunks = [
        Chunk(content="Python loads Markdown documents.", metadata={"example": 1}),
        Chunk(content="The application reads Markdown files.", metadata={"example": 2}),
        Chunk(content="A bicycle has two wheels.", metadata={"example": 3}),
    ]
    embedded_chunks = embed_chunks(chunks, model=load_embedding_model())
    dimensions = [len(embedded.vector) for embedded in embedded_chunks]
    related_similarity = cosine_similarity(
        embedded_chunks[0].vector, embedded_chunks[1].vector
    )
    unrelated_similarity = cosine_similarity(
        embedded_chunks[0].vector, embedded_chunks[2].vector
    )

    if len(embedded_chunks) != 3:
        raise EmbeddingError("The demonstration did not generate three vectors.")
    if dimensions != [384, 384, 384]:
        raise EmbeddingError(f"Unexpected demonstration dimensions: {dimensions}.")
    if related_similarity <= unrelated_similarity:
        raise EmbeddingError(
            "The related sentence pair was not more similar than the unrelated pair."
        )

    print(f"Generated vectors: {len(embedded_chunks)}")
    print(f"Vector dimensions: {dimensions}")
    print(f"Related similarity: {related_similarity:.4f}")
    print(f"Unrelated similarity: {unrelated_similarity:.4f}")


if __name__ == "__main__":
    main()
