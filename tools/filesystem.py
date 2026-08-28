"""Read-only filesystem tools restricted to this project repository."""

from pathlib import Path

from tools.registry import Tool


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path: str, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a requested path and require it to stay inside the project root."""
    requested_path = Path(path)
    if ".." in requested_path.parts:
        raise ValueError("Path traversal with '..' is not allowed.")

    resolved_root = project_root.resolve()
    resolved_path = (resolved_root / requested_path).resolve()

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Path is outside the project root.") from exc

    return resolved_path


def list_files(path: str, *, project_root: Path = PROJECT_ROOT) -> list[str]:
    """List immediate entries in a project directory in deterministic order."""
    resolved_root = project_root.resolve()
    directory = _resolve_project_path(path, project_root=resolved_root)

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    return sorted(entry.relative_to(resolved_root).as_posix() for entry in directory.iterdir())


def read_file(path: str, *, project_root: Path = PROJECT_ROOT) -> str:
    """Read one UTF-8 text file inside the project root."""
    file_path = _resolve_project_path(path, project_root=project_root)

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if file_path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    return file_path.read_text(encoding="utf-8")


LIST_FILES_TOOL = Tool(
    name="list_files",
    description="List immediate files and directories inside a project directory.",
    argument_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Project-relative directory path. Use '.' for the project root.",
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    implementation=list_files,
)


READ_FILE_TOOL = Tool(
    name="read_file",
    description="Read the UTF-8 text content of a file inside the project.",
    argument_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Project-relative path to the text file to read.",
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    implementation=read_file,
)
