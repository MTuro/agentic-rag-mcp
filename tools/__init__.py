"""Generic tool definitions and the milestone's educational tool."""

from tools.registry import Tool, ToolRegistry
from tools.word_count import WORD_COUNT_TOOL, word_count

__all__ = ["Tool", "ToolRegistry", "WORD_COUNT_TOOL", "word_count"]
