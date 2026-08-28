"""The single educational tool available in Milestone 3."""

from tools.registry import Tool


def word_count(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split())


WORD_COUNT_TOOL = Tool(
    name="word_count",
    description="Count whitespace-separated words in text.",
    argument_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text whose words should be counted.",
            }
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    implementation=word_count,
)
