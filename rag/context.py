"""Format retrieved evidence for a grounded language-model prompt."""

from collections.abc import Iterable

from rag.vector_store import SearchResult


def build_context(results: Iterable[SearchResult]) -> str:
    """Preserve retrieval order and label every chunk with its source."""
    return "\n\n".join(
        f"[Source: {result.chunk.metadata.get('source', 'unknown')}]\n"
        f"{result.chunk.content}"
        for result in results
    )
