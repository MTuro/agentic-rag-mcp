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
