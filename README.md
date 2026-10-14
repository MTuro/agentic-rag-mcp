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

## Milestone 14: first MCP server

The first MCP server exposes only the existing read-only `list_files` function:

```text
MCP host -> MCP client -> stdio -> MCP server -> list_files -> result
```

The host is the application the user interacts with. Its MCP client discovers
and invokes capabilities offered by this repository's MCP server. A tool is a
model-invoked function; resources are application-loaded context, and prompts
are user-selected message templates. This milestone registers one tool and no
resources or prompts.

`mcp_server.server` uses the official Python MCP SDK. The SDK discovers the
tool's name, description, annotations, and input schema from its registration
and typed function signature. Its thin adapter accepts only a project-relative
`path` and delegates to `tools.filesystem.list_files`, so path confinement,
validation, sorting, and error behavior remain in the normal Python module.

Start the local stdio server with:

```powershell
.\.venv\Scripts\python.exe -m mcp_server.server
```

The command waits silently for a host to send MCP messages through stdin.
Stdout carries the protocol and must not be used for application logging. To
inspect the server interactively, install Node.js so `npx` is available, then
run:

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\mcp.exe dev mcp_server/server.py:mcp
```

The Inspector should discover exactly `list_files`. Calling it with `{"path":
"."}` returns a sorted project-relative listing; missing directories and `..`
return tool errors without allowing access outside the configured project root.

Automated tests use both the SDK's in-memory client and a real stdio subprocess:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mcp_server.py tests/test_filesystem.py -v
```

## Milestone 15: existing tools through MCP

The stdio server now advertises the six existing read-only capabilities:

```text
list_files(path)       read_file(path)       search_code(query)
git_status()           git_diff(staged=False) search_docs(query)
```

Each MCP adapter reuses the corresponding tool name and description, exposes
only its public arguments, and delegates directly to the existing Python
function. The MCP SDK generates the input schema from the adapter's type hints,
validates calls, and returns implementation exceptions as tool errors. All six
tools are marked read-only and closed-world.

`search_docs` remains lazy: importing the server or discovering its tools does
not load MiniLM, open ChromaDB, contact Ollama, or perform retrieval. Those
dependencies are used only when a client invokes `search_docs`.

```text
MCP Client
   |
   v
discover search_code
   |
   v
invoke search_code(query)
   |
   v
MCP adapter
   |
   v
tools.search.search_code
   |
   v
result
   |
   v
MCP response
```

Run the protocol server with:

```powershell
.\.venv\Scripts\python.exe -m mcp_server.server
```

The process writes MCP messages to stdout, so normal application output must
not be printed there. The main agent still calls its local `ToolRegistry`; MCP
does not replace agent tool access yet. The MCP client is Milestone 16, and
agent-to-MCP integration is Milestone 17.

## Milestone 16: standalone MCP client

The MCP **server** exposes the existing Python tools. The standalone MCP
**client** starts that server, initializes the protocol session, discovers its
tools, and sends explicit caller-selected invocations. The main application
agent still uses its local Python `ToolRegistry`; it does not use MCP yet.

```text
Standalone CLI / Python caller
        |
        v
MCP Client -> stdio -> MCP Server -> existing Python tools
```

`mcp_client.client.connect()` constructs a lazy official SDK `Client` without
starting a process. Entering `async with connect()` starts
`sys.executable -m mcp_server.server` in the repository root and performs the MCP
initialization handshake. Launch with the repository-local interpreter so the
child uses the same virtual environment. No server import, model loading, or
Ollama connection occurs when importing the client.

`list_tools()` returns the server's names, descriptions, input schemas, and
other metadata unchanged. The client contains no copied tool catalog or second
registry. `call_tool(name, arguments)` forwards caller input to MCP; the server
validates it and delegates to the existing Python implementation. A small server
middleware check rejects undeclared argument keys before the SDK can silently
discard them, using the server's existing schemas.

### Run locally on Windows

Run from the repository root. These commands work in Windows PowerShell 5.1;
the backslashes preserve JSON quotes through its native-command argument parser:

```powershell
.\.venv\Scripts\python.exe -m mcp_client.client --list-tools
.\.venv\Scripts\python.exe -m mcp_client.client list_files '{\"path\":\".\"}'
.\.venv\Scripts\python.exe -m mcp_client.client search_code '{\"query\":\"mcp_list_files\"}'
.\.venv\Scripts\python.exe -m mcp_client.client list_files '{\"path\":\"..\"}'
```

In shells that preserve single-quoted arguments directly, such as PowerShell 7
with standard native argument passing, use ordinary JSON: `'{"path":"."}'`.

The CLI always discovers tools first and prints SDK results as JSON. A successful
invocation exits with status 0. Tool failures, including unknown tools, invalid
arguments, and forbidden paths, retain `isError: true` and exit with status 1.
Malformed CLI JSON or a non-object argument exits with status 2 before starting
the server. Startup, initialization, discovery, and transport failures remain
fatal exceptions, distinct from tool results. SDK task groups may wrap exceptions
in an `ExceptionGroup`.

The Python interface uses the same SDK result objects (`input_schema` and
`is_error` are their Python field names):

```python
import asyncio
from mcp_client.client import connect

async def main():
    async with connect() as client:
        tools = await client.list_tools()
        for tool in tools.tools:
            print(tool.name, tool.description, tool.input_schema)
        result = await client.call_tool("list_files", {"path": "."})
        print(result.is_error, result.structured_content)

asyncio.run(main())
```

### Runtime and subprocess lifecycle

```text
Standalone MCP Client
        |
        v
start stdio server -> initialize -> list_tools -> receive tool schemas
        |
        v
call_tool("list_files", {"path":"."})
        |
        v
MCP Server -> tools.filesystem.list_files -> MCP result -> Client
        |
        v
exit async context -> close session and stdio -> reap server subprocess
```

For this example, the server confines `.` to the configured project root,
returns entries including `README.md`, and the client receives the structured
result. The SDK owns session and process cleanup on success and failure: it
closes stdin, waits for exit, and terminates the process tree if necessary.
Create a fresh `connect()` context for each session. Stdout carries MCP between
the child and client; server diagnostics go to stderr.

Verification includes the actual stdio server, repeated sessions, forbidden
paths, malformed/unknown tool calls, failed startup/initialization/discovery,
abrupt server exit, and recorded child-process exit codes. A separate temporary
server with a different catalog verifies dynamic discovery and schema/argument
preservation. Automated checks invoke no real Ollama or `search_docs` inference.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mcp_client.py tests/test_mcp_server.py -v
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m pip check
```
