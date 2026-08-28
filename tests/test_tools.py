"""Tests for generic tool representation and registry dispatch."""

import pytest

from tools import Tool, ToolRegistry


def make_repeat_tool() -> Tool:
    """Create a small test-only tool for exercising the generic registry."""
    return Tool(
        name="repeat",
        description="Repeat text a requested number of times.",
        argument_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "times": {"type": "integer"},
            },
            "required": ["text", "times"],
            "additionalProperties": False,
        },
        implementation=lambda text, times: text * times,
    )


def test_registry_exposes_model_facing_tool_definition() -> None:
    tool = make_repeat_tool()
    registry = ToolRegistry([tool])

    assert registry.model_definitions() == [
        {
            "name": "repeat",
            "description": "Repeat text a requested number of times.",
            "arguments": tool.argument_schema,
        }
    ]


def test_registry_executes_tool_with_keyword_arguments() -> None:
    registry = ToolRegistry([make_repeat_tool()])

    assert registry.execute("repeat", {"text": "go", "times": 3}) == "gogogo"


def test_registry_rejects_duplicate_tool_name() -> None:
    tool = make_repeat_tool()
    registry = ToolRegistry([tool])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_registry_rejects_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Unknown tool"):
        registry.execute("missing", {})


def test_registry_rejects_non_object_arguments() -> None:
    registry = ToolRegistry([make_repeat_tool()])

    with pytest.raises(ValueError, match="must be a JSON object"):
        registry.execute("repeat", "not an object")  # type: ignore[arg-type]


def test_registry_reports_arguments_that_do_not_match_python_function() -> None:
    registry = ToolRegistry([make_repeat_tool()])

    with pytest.raises(ValueError, match="Invalid arguments"):
        registry.execute("repeat", {"text": "go"})
