"""Generic, educational, and read-only filesystem tools."""

from tools.filesystem import LIST_FILES_TOOL, READ_FILE_TOOL, list_files, read_file
from tools.git import GIT_DIFF_TOOL, GIT_STATUS_TOOL, git_diff, git_status
from tools.rag import SEARCH_DOCS_TOOL, search_docs
from tools.registry import Tool, ToolRegistry
from tools.search import SEARCH_CODE_TOOL, search_code
from tools.word_count import WORD_COUNT_TOOL, word_count

__all__ = [
    "GIT_DIFF_TOOL",
    "GIT_STATUS_TOOL",
    "LIST_FILES_TOOL",
    "READ_FILE_TOOL",
    "SEARCH_CODE_TOOL",
    "SEARCH_DOCS_TOOL",
    "Tool",
    "ToolRegistry",
    "WORD_COUNT_TOOL",
    "git_diff",
    "git_status",
    "list_files",
    "read_file",
    "search_code",
    "search_docs",
    "word_count",
]
