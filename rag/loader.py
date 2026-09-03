"""Load Markdown documents from the configured project directory."""

from dataclasses import dataclass
from pathlib import Path

from config import PROJECT_ROOT


@dataclass(frozen=True)
class Document:
    """One source document and the metadata needed to identify it."""

    content: str
    metadata: dict[str, str]


def _resolve_documents_directory(path: str, *, project_root: Path) -> tuple[Path, Path]:
    """Resolve a document directory and require it to stay inside the project."""
    requested_path = Path(path)
    if ".." in requested_path.parts:
        raise ValueError("Path traversal with '..' is not allowed.")

    resolved_root = project_root.resolve()
    directory = (resolved_root / requested_path).resolve()

    try:
        directory.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Documents path is outside the project root.") from exc

    if not directory.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {path}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Documents path is not a directory: {path}")

    return resolved_root, directory


def load_markdown_documents(
    path: str = "documents", *, project_root: Path = PROJECT_ROOT
) -> list[Document]:
    """Recursively load UTF-8 Markdown files in deterministic path order."""
    resolved_root, directory = _resolve_documents_directory(
        path, project_root=project_root
    )
    documents: list[Document] = []

    for candidate in sorted(directory.rglob("*")):
        if candidate.suffix.casefold() != ".md":
            continue

        resolved_candidate = candidate.resolve()
        try:
            relative_path = resolved_candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"Markdown file is outside the project root: {candidate}"
            ) from exc

        if not resolved_candidate.is_file():
            continue

        documents.append(
            Document(
                content=resolved_candidate.read_text(encoding="utf-8"),
                metadata={
                    "source": relative_path.as_posix(),
                    "filename": resolved_candidate.name,
                },
            )
        )

    return documents
