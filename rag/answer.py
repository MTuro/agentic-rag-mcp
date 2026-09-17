"""Answer a question from retrieved Markdown evidence using local Qwen."""

import argparse
from collections.abc import Sequence

from chromadb.api.models.Collection import Collection

from llm.ollama_client import chat
from rag.context import build_context
from rag.embeddings import EmbeddingEncoder, load_embedding_model
from rag.retriever import retrieve_chunks
from rag.vector_store import open_collection


NO_EVIDENCE = "The available context does not contain enough information to answer."
SYSTEM_PROMPT = (
    "Answer the user's question using only the supplied context. "
    "Do not invent information that the context does not support. "
    "If the context does not contain enough information, say so."
)


def answer_question(
    question: str,
    collection: Collection,
    *,
    model: EmbeddingEncoder,
    top_k: int = 3,
) -> str:
    """Retrieve evidence once, then ask the local model for a grounded answer."""
    if not isinstance(question, str):
        raise TypeError("question must be a string.")
    if not question.strip():
        raise ValueError("question must not be empty or whitespace.")

    results = retrieve_chunks(question, collection, model=model, top_k=top_k)
    if not results:
        return NO_EVIDENCE

    context = build_context(results)
    return chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Answer from the default persistent collection."""
    parser = argparse.ArgumentParser(description="Answer from indexed Markdown with local Qwen.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    model = load_embedding_model()
    collection = open_collection()
    print(answer_question(args.question, collection, model=model, top_k=args.top_k))


if __name__ == "__main__":
    main()
