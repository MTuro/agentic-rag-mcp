"""Embed a text question and retrieve its closest stored chunks."""

from chromadb.api.models.Collection import Collection

from rag.chunking import Chunk
from rag.embeddings import EmbeddingEncoder, embed_chunks
from rag.vector_store import SearchResult, query_chunks


def retrieve_chunks(
    question: str,
    collection: Collection,
    *,
    model: EmbeddingEncoder,
    top_k: int = 3,
) -> list[SearchResult]:
    """Return Top-K evidence chunks for a nonempty natural-language question."""
    if not isinstance(question, str):
        raise TypeError("question must be a string.")
    if not question.strip():
        raise ValueError("question must not be empty or whitespace.")
    if type(top_k) is not int:
        raise TypeError("top_k must be an integer, not a boolean.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    query_chunk = Chunk(content=question, metadata={"kind": "query"})
    query_vector = embed_chunks([query_chunk], model=model)[0].vector
    return query_chunks(collection, query_vector, top_k=top_k)
