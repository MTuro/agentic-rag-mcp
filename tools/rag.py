"""Expose the standalone grounded RAG answer as a read-only agent tool."""

from tools.registry import Tool


def search_docs(query: str) -> str:
    """Answer a documentation question from the indexed project documents."""
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    if not query.strip():
        raise ValueError("query must not be empty or whitespace.")

    from rag.answer import answer_question
    from rag.embeddings import load_embedding_model
    from rag.vector_store import open_collection

    model = load_embedding_model()
    collection = open_collection()
    return answer_question(query, collection, model=model)


SEARCH_DOCS_TOOL = Tool(
    name="search_docs",
    description=(
        "Answer a question about indexed project documentation using retrieved "
        "context. Use search_code for exact source-code text. If documentation "
        "is insufficient, report that instead of inventing an answer."
    ),
    argument_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question to answer from indexed documentation.",
                "minLength": 1,
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    implementation=search_docs,
)
