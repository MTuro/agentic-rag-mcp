# Local RAG Documents

This project implements retrieval-augmented generation (RAG) locally. Retrieval
finds relevant passages in indexed documentation; generation uses those passages
as evidence for an answer from Qwen through Ollama.

## Document ingestion

`rag.ingest` loads Markdown files recursively from `documents/` and splits them
into overlapping chunks (500 characters with 50 characters of overlap by default).
The sentence-transformers model `sentence-transformers/all-MiniLM-L6-v2` converts
each chunk into an embedding. ChromaDB persists the chunk text, embedding, stable
ID, and metadata such as source path and character offsets in `chroma_db/`.

Run from the repository root on Linux, including fish:

```sh
./.venv/bin/python -m rag.ingest documents
```

This command indexes Markdown under `documents/`; it does not index the root
`README.md`. Re-run ingestion after editing documentation to update matching
chunk IDs. Ingestion upserts records; it does not delete obsolete chunks when
files disappear or become shorter.

## Retrieval

`rag.retriever.retrieve_chunks` embeds the question with the same MiniLM model
used for ingestion. ChromaDB searches for nearby vectors using cosine distance
and returns the Top-K chunks (three by default), including text, source metadata,
and distance. Smaller distance means a closer match. Retrieval finds evidence;
it does not itself generate an answer or guarantee that the evidence is enough.

Inspect retrieved passages without asking Qwen to generate an answer:

```sh
./.venv/bin/python -m rag.query "How does retrieval work?"
```

## Grounded answer generation

`rag.answer.answer_question` reuses the retriever and labels retrieved passages
with their source paths. The prompt separates the evidence from the question
and instructs Qwen to answer only from that evidence. If retrieval returns no
chunks, the function returns an explicit insufficient-context message without
calling Ollama. With retrieved chunks, Qwen is instructed to report insufficient
evidence when needed; model compliance is not guaranteed.

```sh
./.venv/bin/python -m rag.answer "How does retrieval work?"
```

MiniLM creates vectors for semantic search; Qwen generates conversational text.
They are separate models with separate jobs. Inference and vector storage run
locally without paid APIs. Loading MiniLM can contact Hugging Face to download
or check model files, so normal execution is not necessarily offline.

## Agent and MCP integration

The agent discovers tools through MCP and dynamically chooses whether to call
`search_docs(query)`. Its MCP wrapper delegates to the existing Python tool,
which reuses the standalone grounded-answer path. The generated answer returns
as a tool observation before the agent decides its next action or final answer.
Retrieval is not automatically performed for every user request.

```text
Question -> agent -> MCP search_docs -> retriever -> MiniLM -> ChromaDB
                                           |
                                           v
                               retrieved evidence -> Ollama/Qwen
                                                          |
                                                          v
                                  agent <- tool observation
```

For exact source-code text, use `search_code`; `search_docs` searches indexed
documentation by semantic similarity. Documentation describes intended behavior;
inspecting source files and Git state requires the corresponding read-only tools.

Start the agent with Ollama running and `qwen3:8b` available:

```sh
./.venv/bin/python main.py
```

The MCP server starts automatically. Ask what the documentation says about
retrieval and check the terminal for a `search_docs` tool call and its observation.
