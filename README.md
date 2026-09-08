# agentic-rag-mcp
An agentic software engineering assistant exploring RAG, tool calling and Model Context Protocol (MCP).

## Milestone 9: standalone ChromaDB vector storage

Milestone 8 produces `EmbeddedChunk` objects in memory. Milestone 9 stores their
IDs, text, metadata, and vectors in a local Chroma collection, then retrieves
up to K nearest chunks using an already-created query vector.

```text
EmbeddedChunk objects + explicit IDs
        -> upsert_chunks -> persistent Chroma collection -> local disk

query vector + K
        -> query_chunks -> Chroma similarity search -> list[SearchResult]
```

`rag.vector_store` contains ordinary Python functions:

- `open_collection(project_root=PROJECT_ROOT, name="document_chunks")` opens a
  cosine collection under `PROJECT_ROOT / "chroma_db"`. The existing path
  containment check rejects symlink escapes. The project root must exist.
- `upsert_chunks(collection, embedded_chunks, ids=...)` validates the entire
  batch before writing. Reusing an ID replaces its text, metadata, and vector;
  duplicate IDs in one batch are rejected. Because Chroma merges metadata,
  omitted old keys are explicitly cleared in the upsert. No record is deleted.
- `query_chunks(collection, query_vector, top_k=3)` returns `SearchResult`
  objects containing `chunk_id`, the original chunk data, and cosine distance.
  It preserves Chroma's result order. An empty collection returns an empty list.

A **collection** groups searchable records. The **embedding** is used for
similarity search, while stored **text** is readable evidence and **metadata**
identifies its source and offsets. Cosine distance is `1 - cosine_similarity`:
smaller is closer. **Top-K** is a result-count limit, not a relevance threshold.
Chroma's index performs approximate nearest-neighbor search; tied results have
no promised ordering.

A persistent client writes to disk so another process can reopen the collection.
An ephemeral client keeps records in memory and is useful for isolated tests.
The client/path setup and validation are infrastructure; storing known vectors,
retrieving neighbors, and reading them in a fresh process demonstrate the concept.

Automatic embedding is disabled with `embedding_function=None`; both write and
query calls supply vectors explicitly. Telemetry is disabled. No cloud database,
paid API, or Qwen inference is involved in storage or search. Metadata paths are
data only and are never opened by the store.

The supported metadata values are strings and integers, excluding booleans.
Vectors must be finite, nonempty, nonzero, and dimensionally compatible. Invalid
types/values raise `TypeError`/`ValueError`; Chroma failures raise
`VectorStoreError` with the original cause. Incompatible collection distances
are rejected without migration or reset. Prevalidation does not promise rollback
for database failures. Upsert uses a read followed by a write to replace metadata;
concurrent writers to the same IDs are not coordinated in this learning milestone.
Upsert does not remove obsolete chunks when a source document changes.

The agent, ToolRegistry, existing tools, and MAX_STEPS are unchanged. This is not
a RAG tool. General document ingestion/question retrieval belongs to Milestone 10;
agent access to RAG belongs to Milestone 11.

### Install and run on Windows

Use the repository-local interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m rag.vector_store
```

The demonstration uses the existing local MiniLM model. Its first load may need
to download model files; subsequent inference can use the local cache.

1. Create three fixed chunks: Markdown loading, reading Markdown files, bicycles.
2. `embed_chunks` converts them into MiniLM vectors.
3. Store the vectors and explicit IDs in the persistent `milestone_9_demo` collection.
4. Reopen that collection and use the first chunk's vector as the query.
5. Print Top-K results with distances, checking that the related sentence is
   closer than the bicycle sentence. Exact floating-point values are not fixed.

The demo reuses its three IDs on subsequent runs. Data stays in the already
Git-ignored `chroma_db/` directory. Automated tests use temporary roots or
ephemeral collections, never this project's database.

### Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_vector_store.py tests/test_rag_embeddings.py -v
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import chromadb; import rag.vector_store"
.\.venv\Scripts\python.exe -m compileall -q rag tests/test_rag_vector_store.py
.\.venv\Scripts\python.exe -m pytest tests/test_rag_vector_store.py -k persistence -v
.\.venv\Scripts\python.exe -m rag.vector_store
```

Tests use synthetic vectors and a fake encoder for deterministic Milestone 8
integration, without downloading MiniLM or calling Qwen. Persistence is tested
with separate writer and reader processes using the interpreter running pytest.

For real agent regression, run `.\.venv\Scripts\python.exe main.py` and enter:

> Read rag/vector_store.py and explain how it stores embeddings and retrieves
> Top-K chunks. Base your answer on the code; do not claim to have queried the
> database.

Qwen should select existing repository-inspection tools and explain the code.
This checks code inspection, not Qwen access to Chroma. Run the command again with
`Say hello in one short sentence.` as a direct-answer control. Judge behavior,
not exact wording, and report unavailable integrations separately from passing tests.
