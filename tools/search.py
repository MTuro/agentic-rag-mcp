"""Read-only literal code search restricted to the configured project root."""

from pathlib import Path

from config import PROJECT_ROOT
from tools.registry import Tool


def search_code(
    query: str, *, project_root: Path = PROJECT_ROOT
) -> list[dict[str, str | int]]:
    """Return case-insensitive text matches from files inside the project root."""
    if not query:
        raise ValueError("Search query must not be empty.")

    resolved_root = project_root.resolve()
    normalized_query = query.casefold()
    matches: list[dict[str, str | int]] = []

    for candidate in sorted(resolved_root.rglob("*")):
        relative_parts = candidate.relative_to(resolved_root).parts
        if any(part.startswith(".") or part == "__pycache__" for part in relative_parts):
            continue

        resolved_candidate = candidate.resolve()
        try:
            relative_path = resolved_candidate.relative_to(resolved_root)
        except ValueError:
            continue
        if not resolved_candidate.is_file():
            continue

        try:
            lines = resolved_candidate.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        for line_number, line in enumerate(lines, start=1):
            if normalized_query in line.casefold():
                matches.append(
                    {
                        "filename": relative_path.as_posix(),
                        "line_number": line_number,
                        "line": line,
                    }
                )

    return matches


SEARCH_CODE_TOOL = Tool(
    name="search_code",
    description="Search project files for exact text, ignoring letter case.",
    argument_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The exact text to find in project files.",
                "minLength": 1,
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    implementation=search_code,
)
