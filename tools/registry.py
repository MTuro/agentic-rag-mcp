"""Represent tools and dispatch model-selected calls to Python functions."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    """A model-facing description paired with a Python implementation."""

    name: str
    description: str
    argument_schema: dict[str, Any]
    implementation: Callable[..., Any]

    def model_definition(self) -> dict[str, Any]:
        """Return the tool information that is shown to the language model."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.argument_schema,
        }


class ToolRegistry:
    """Store available tools and execute them by name."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool, requiring each tool name to be unique."""
        if tool.name in self._tools:
            raise ValueError(f"A tool named {tool.name!r} is already registered.")
        self._tools[tool.name] = tool

    def model_definitions(self) -> list[dict[str, Any]]:
        """Return descriptions and schemas for all registered tools."""
        return [tool.model_definition() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool using model-produced keyword arguments."""
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name!r}.")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")

        try:
            return tool.implementation(**arguments)
        except TypeError as exc:
            raise ValueError(f"Invalid arguments for tool {name!r}.") from exc
