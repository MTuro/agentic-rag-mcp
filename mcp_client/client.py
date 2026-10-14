"""Standalone stdio connection and CLI, independent of the application agent."""

import argparse
import asyncio
import json
from pathlib import Path
import sys

from mcp import Client, StdioServerParameters


def connect(server: StdioServerParameters | None = None) -> Client:
    """Construct a lazy SDK client. Use ``async with`` to initialize and close it.

    The active interpreter is the repository .venv when launched as documented.
    Tool results retain ``is_error``; transport/protocol exceptions propagate.
    """
    if server is None:
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
    return Client(server, mode="legacy")


async def _run(name: str | None, arguments: dict[str, object]) -> int:
    async with connect() as client:
        discovery = await client.list_tools()
        if name is None:
            print(discovery.model_dump_json(by_alias=True, indent=2))
            return 0
        result = await client.call_tool(name, arguments)
        print(result.model_dump_json(by_alias=True, indent=2))
        return int(result.is_error)


def main() -> int:
    """Discover tools or invoke one explicitly; no model chooses the action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument("name", nargs="?")
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON object")
    args = parser.parse_args()
    if args.list_tools == (args.name is not None):
        parser.error("choose --list-tools or a tool name with optional JSON arguments")
    try:
        arguments = json.loads(args.arguments)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid JSON arguments: {exc}")
    if not isinstance(arguments, dict):
        parser.error("arguments must be a JSON object")
    return asyncio.run(_run(args.name, arguments))


if __name__ == "__main__":
    raise SystemExit(main())
