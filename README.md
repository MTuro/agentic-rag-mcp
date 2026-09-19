# agentic-rag-mcp

An educational, fully local agentic software engineering assistant exploring
RAG, tool calling, and Model Context Protocol (MCP).

## Milestone 10: complete standalone RAG

Milestones 7–9 implemented document loading, chunking, local embeddings, and
ChromaDB separately. Milestone 10 connects them into two explicit workflows:

```text
INGESTION
Markdown -> Document -> Chunk -> MiniLM embedding -> ChromaDB

RETRIEVAL
question -> MiniLM embedding -> Chroma similarity search -> Top-K chunks
```

`rag.ingest.ingest_documents` recursively loads UTF-8 Markdown below the
configured project root, creates overlapping character chunks, embeds them in
one batch, and upserts them into a supplied Chroma collection. Each chunk uses a
stable, readable ID such as `documents/guide.md::0`. Repeating the same
ingestion replaces matching records instead of growing the collection.

`rag.retriever.retrieve_chunks` validates and embeds one question with the same
embedding interface, then sends its explicit vector to Chroma. It returns
`SearchResult` values containing the ID, text, source metadata, offsets, and
cosine distance. Smaller distance means a closer result; Top-K limits the number
of evidence chunks returned.

This milestone retrieves context only. It does not ask Qwen to generate an
answer, expose RAG as an agent tool, or change agent behavior. Those concerns are
kept outside the standalone pipeline so loading, indexing, and retrieval remain
deterministic and independently testable.

### Run locally on Windows

Use the repository-local interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m rag.ingest
.\.venv\Scripts\python.exe -m rag.query "What happens to Markdown documents before retrieval?"
```

Both commands use the persistent `document_chunks` collection under the
Git-ignored `chroma_db/` directory. Ingestion defaults to `documents/`, a chunk
size of 500 characters, and an overlap of 50. Optional CLI arguments are:

```powershell
.\.venv\Scripts\python.exe -m rag.ingest documents --chunk-size 500 --chunk-overlap 50
.\.venv\Scripts\python.exe -m rag.query "How does retrieval work?" --top-k 3
```

The MiniLM model is `sentence-transformers/all-MiniLM-L6-v2`. Its first load may
need to download model files; subsequent inference can use the local cache. No
Ollama server, hosted embedding API, paid API, or cloud vector database is used.

### Runtime trace

For the question `What happens to Markdown documents before retrieval?`:

1. `rag.ingest` opens the project-local Chroma collection and loads MiniLM once.
2. `load_markdown_documents` reads files such as `documents/README.md`, attaching
   `source` and `filename` metadata.
3. `chunk_documents` adds `chunk_index`, `start_char`, and `end_char`.
4. `embed_chunks` requests normalized vectors for all chunks in one model call.
5. `upsert_chunks` stores the text, metadata, explicit vector, and stable ID.
6. `rag.query` embeds the question with MiniLM in the same vector space.
7. `query_chunks` asks Chroma for the nearest records and returns Top-K evidence.
8. The CLI prints ranked source, distance, metadata, and chunk text. It does not
   generate a conversational answer.

Ingestion is intentionally upsert-only. If a file disappears, becomes shorter,
or chunking parameters change, obsolete IDs can remain. Full collection
synchronization and deletion are outside this milestone.

### Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_ingest.py tests/test_rag_retriever.py tests/test_rag_query.py -v
.\.venv\Scripts\python.exe -m pytest tests/test_rag_loader.py tests/test_rag_chunking.py tests/test_rag_embeddings.py tests/test_rag_vector_store.py tests/test_rag_ingest.py tests/test_rag_retriever.py tests/test_rag_query.py -v
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import rag.ingest; import rag.retriever; import rag.query"
.\.venv\Scripts\python.exe -m compileall -q rag tests
```

Automated RAG tests use deterministic fake encoders and real temporary or
ephemeral Chroma collections. They do not download MiniLM, contact Ollama, or
write to the repository's persistent `chroma_db/` directory.

## Milestone 11: grounded answers

`rag.answer` adds one explicit path after Milestone 10 retrieval:

```text
question -> retrieve_chunks -> build_context -> Qwen via Ollama -> answer
```

`build_context` labels each retrieved chunk with its `documents/...` source in
retrieval order. The prompt asks Qwen to use only that context, avoid unsupported
claims, and say when evidence is insufficient. With no retrieved chunks, the
command returns an insufficient-context message without calling Qwen. This
standalone command does not change the agent or add tool calling.

Index documents first, then run Ollama locally and ask a question:

```powershell
.\.venv\Scripts\python.exe -m rag.ingest
.\.venv\Scripts\python.exe -m rag.answer "How does retrieval work?"
```

The answer command uses the same local MiniLM model, persistent Chroma
collection, and `OLLAMA_MODEL` setting (default `qwen3:8b`) as the existing
components. Its unit tests mock the Ollama call and need no running server.

## Milestone 12: RAG as an agent tool

The agent can now choose `search_docs(query)` for questions about indexed
documentation. That tool calls the same standalone `rag.answer.answer_question`
path; its grounded answer becomes an observation for the agent's next decision.
Retrieval runs only when the model selects the tool. An empty collection returns
an insufficient-context observation. Index documents before asking the agent:

```powershell
.\.venv\Scripts\python.exe -m rag.ingest
"What does the documentation say about retrieval?" | .\.venv\Scripts\python.exe main.py
```

The agent loop and grounded answer each call local Ollama when `search_docs` is
selected. The tool does not expose ChromaDB or embedding details to the agent.

## Milestone 13: tool-calling reliability

The manual loop now returns malformed decisions, unknown tools, invalid
arguments, and tool failures to the model as error observations so it can
correct its next decision. Ollama connection failures remain fatal because the
agent cannot repair unavailable inference infrastructure.

Successful calls are identified by tool name and canonically serialized
arguments. An exact repeat is blocked, while different arguments and retries
after failures remain valid. Tool observations are capped at 8,000 characters
with a truncation marker, and `max_steps` must be a positive integer. The system
prompt tells the model to answer as soon as its evidence is sufficient; the
five-step limit remains the deterministic backstop.
