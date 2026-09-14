"""Command-line interface for standalone document retrieval."""

import argparse
from collections.abc import Sequence

from rag.embeddings import load_embedding_model
from rag.retriever import retrieve_chunks
from rag.vector_store import open_collection


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve evidence from indexed Markdown.")
    parser.add_argument("question", help="Question to embed and search for")
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Retrieve from the default persistent collection and print evidence."""
    args = _build_parser().parse_args(argv)
    model = load_embedding_model()
    collection = open_collection()
    results = retrieve_chunks(
        args.question,
        collection,
        model=model,
        top_k=args.top_k,
    )

    if not results:
        print("No matching chunks found.")
        return

    for rank, result in enumerate(results, start=1):
        source = result.chunk.metadata.get("source", "unknown")
        print(f"[{rank}] source={source} distance={result.distance:.6f}")
        print(f"metadata={result.chunk.metadata}")
        print(result.chunk.content)
        if rank != len(results):
            print()


if __name__ == "__main__":
    main()
