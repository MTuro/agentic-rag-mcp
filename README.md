# agentic-rag-mcp

An educational software engineering assistant for learning local LLM inference,
manual agent loops, retrieval-augmented generation (RAG), and Model Context
Protocol (MCP). The application uses local models and requires no paid APIs.

## Current capabilities

The CLI uses Qwen through Ollama and dynamically discovers six read-only tools
through MCP: `list_files`, `read_file`, `search_code`, `git_status`, `git_diff`,
and `search_docs`. The model decides which tools to call and when to answer.
Tool calls and observations appear in the terminal so their evidence is visible.

Documentation search uses MiniLM embeddings, local ChromaDB storage, and a
separate Ollama call to generate an answer from retrieved passages. Retrieval
runs when the model selects `search_docs`; ordinary requests do not automatically
trigger it. The retrieval and grounded-answer commands also work independently.

The agent accepts one message at a time through the CLI or the FastAPI endpoint.
Both use the same agent loop and MCP tools.

## Setup and execution

Run commands from the repository root. Install Python, create a repository-local
`.venv` if needed, and install Ollama. Use the virtual environment's interpreter
directly; activating it is optional.

### Linux / fish

If `.venv` does not exist, create it with `python3 -m venv .venv`. Install the
project dependencies:

```sh
./.venv/bin/python -m pip install -r requirements.txt
```

Check the available Ollama models and download the default if it is missing:

```sh
ollama list
ollama pull qwen3:8b
```

Ollama must be running. If needed, start `ollama serve` in a separate terminal.
Then launch the agent:

```sh
./.venv/bin/python main.py
```

At `You:`, enter `Read README.md and explain what this project does.` Check for
`[tool call] read_file` and its `[observation]` before the final answer. Tool
selection comes from the model, so inspect the trace to verify what it used.
Run the command again for another question. The MCP server starts automatically.

For documentation questions, index the Markdown first:

```sh
./.venv/bin/python -m rag.ingest documents
./.venv/bin/python main.py
```

Ask `What does the documentation say about retrieval?` and check for a
`search_docs` call. This indexes only Markdown under `documents/`, not the root
README or `docs/milestones.md`. Update the evidence in `documents/` and re-run
ingestion after changing it. Ingestion replaces matching chunk IDs but does not
delete obsolete chunks when files disappear or become shorter.

Inspect retrieval or grounded generation separately:

```sh
./.venv/bin/python -m rag.query "How does retrieval work?"
./.venv/bin/python -m rag.answer "How does retrieval work?"
```

### Windows / PowerShell

If needed, create `.venv` with `py -3.12 -m venv .venv`. Use the environment's
Windows interpreter for installation and execution:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m rag.ingest documents
.\.venv\Scripts\python.exe main.py
```

The Ollama requirements are the same. Ingestion is required for documentation
search, not for ordinary file inspection.

### HTTP chat

With Ollama running, start the local HTTP server:

```sh
./.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

On Windows, use `.\.venv\Scripts\python.exe` with the same arguments.
In a browser, open [the interactive API docs](http://127.0.0.1:8000/docs)
and select **Try it out** under **POST /chat**. Opening `/chat` directly
sends a GET request and returns HTTP 405. From a terminal, run:

```sh
curl -sS http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  --data '{"message":"Read README.md and explain what this project does."}'
```

Send a non-blank string `message`. Example response:

```json
{
  "answer": "This project is an educational software engineering assistant.",
  "tools_used": ["read_file"]
}
```

`tools_used` lists each successfully used tool once, in order of first success.
A direct answer returns an empty list. Invalid requests return HTTP 422, and
fatal agent failures return HTTP 500. Each request starts a fresh agent run.

### Models and storage

The conversational model defaults to `qwen3:8b`; `OLLAMA_MODEL` overrides it.
The separate embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
Inference and ChromaDB storage run locally, but dependency installation, model
downloads, and MiniLM cache checks can access the internet. Normal execution is
not necessarily offline. The persistent index is in Git-ignored `chroma_db/`.

## Architecture and tool safety

```text
User -> CLI or FastAPI -> manual agent loop -> Qwen through Ollama
                               ^
                               | tool decision and observation
                               v
                          MCP client -> MCP server -> Python tool
```

The MCP client discovers tool names, descriptions, and schemas from the server.
Thin server adapters reuse the Python implementations. MCP transports tool calls;
the manual agent loop manages decisions, state, observations, and termination.

For `search_docs`, evidence flows through:

```text
Markdown -> chunks -> MiniLM embeddings -> ChromaDB
Question -> MiniLM embedding -> Top-K chunks -> grounded context -> Ollama answer
```

MiniLM creates search vectors; Qwen generates text. No retrieved chunks produces
an insufficient-context result without calling Ollama. Retrieved context does
not guarantee model accuracy, so inspect the evidence and tool trace.

| Tool | Purpose |
| --- | --- |
| `list_files(path)` | List entries within the configured project root |
| `read_file(path)` | Read a project file |
| `search_code(query)` | Find exact text without embeddings |
| `git_status()` | Inspect working-tree state |
| `git_diff(staged=False)` | Inspect unstaged or staged changes |
| `search_docs(query)` | Answer from indexed documentation |

File access is confined to the configured project root (`PROJECT_ROOT` can
override the repository default). Git tools use fixed read-only commands.
The agent has no tools for editing files, deleting files, committing, pushing,
or executing arbitrary shell commands.

The default limit is five model decisions. Successful duplicate calls are
blocked, tool failures become observations, and observations are capped at
8,000 characters with a truncation marker. Connection and transport failures
remain fatal. The CLI prints traces to stderr and the final answer to stdout.

## Testing

Run the automated suite with the repository-local interpreter:

```sh
./.venv/bin/python -m pytest -v
```

On Windows, use `.\.venv\Scripts\python.exe -m pytest -v`.
Automated tests isolate Ollama with fakes or mocks. RAG integration tests use
fake encoders with temporary or ephemeral Chroma collections; MCP tests also
exercise real local server sessions. They do not require running model inference.

For a real model check, run the CLI and inspect its tool observations. For RAG,
index `documents/` first, then use `rag.query` and `rag.answer` as shown above.
Passing automated tests does not establish that local Ollama inference works.

## Learning roadmap

The checklist describes implemented scope, not verification of your environment.
Detailed historical explanations and additional protocol examples are in
[the milestone notes](docs/milestones.md).

- [x] 0–1: Local environment and Python-to-Ollama inference.
- [x] 2–3: Manual agent loop and generic tool schemas.
- [x] 4–6: Safe filesystem access, code search, and Git inspection.
- [x] 7–10: Documents, chunking, embeddings, ChromaDB, and standalone retrieval.
- [x] 11–12: Grounded answers and RAG as an agent tool.
- [x] 13: Tool-calling safeguards.
- [x] 14–16: MCP server, tool adapters, and standalone client.
- [x] 17: Agent tools through MCP.
- [x] 18: FastAPI interface.
- [ ] 19: Agent behavior evaluations.
- [ ] 20: LangGraph version for comparison with the manual loop.
