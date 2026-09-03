"""Tests for safe, deterministic Markdown document loading."""

from pathlib import Path

import pytest

import config
import rag.loader as loader_module
from rag.loader import Document, load_markdown_documents


def test_load_markdown_document_returns_content_and_metadata(tmp_path: Path) -> None:
    documents_directory = tmp_path / "documents"
    documents_directory.mkdir()
    (documents_directory / "guide.md").write_text(
        "# Olá\n\nConteúdo local.", encoding="utf-8"
    )

    assert load_markdown_documents(project_root=tmp_path) == [
        Document(
            content="# Olá\n\nConteúdo local.",
            metadata={"source": "documents/guide.md", "filename": "guide.md"},
        )
    ]


def test_load_markdown_documents_is_recursive_case_insensitive_and_sorted(
    tmp_path: Path,
) -> None:
    documents_directory = tmp_path / "documents"
    nested = documents_directory / "nested"
    nested.mkdir(parents=True)
    (documents_directory / "zeta.md").write_text("zeta", encoding="utf-8")
    (documents_directory / "alpha.MD").write_text("alpha", encoding="utf-8")
    (nested / "beta.md").write_text("beta", encoding="utf-8")
    (documents_directory / "ignored.txt").write_text("ignored", encoding="utf-8")

    documents = load_markdown_documents(project_root=tmp_path)

    assert [document.metadata["source"] for document in documents] == [
        "documents/alpha.MD",
        "documents/nested/beta.md",
        "documents/zeta.md",
    ]
    assert [document.content for document in documents] == ["alpha", "beta", "zeta"]


def test_load_markdown_documents_uses_central_project_root() -> None:
    assert loader_module.PROJECT_ROOT is config.PROJECT_ROOT


def test_load_markdown_documents_accepts_custom_directory(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "notes.md").write_text("notes", encoding="utf-8")

    documents = load_markdown_documents("knowledge", project_root=tmp_path)

    assert documents[0].metadata["source"] == "knowledge/notes.md"


def test_load_markdown_documents_returns_empty_list_without_markdown(
    tmp_path: Path,
) -> None:
    documents_directory = tmp_path / "documents"
    documents_directory.mkdir()
    (documents_directory / "notes.txt").write_text("notes", encoding="utf-8")

    assert load_markdown_documents(project_root=tmp_path) == []


def test_load_markdown_documents_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_markdown_documents(project_root=tmp_path)


def test_load_markdown_documents_rejects_file_path(tmp_path: Path) -> None:
    (tmp_path / "document.md").write_text("document", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="not a directory"):
        load_markdown_documents("document.md", project_root=tmp_path)


def test_load_markdown_documents_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="traversal"):
        load_markdown_documents("../outside", project_root=tmp_path)


def test_load_markdown_documents_rejects_absolute_path_outside_root(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="outside the project root"):
        load_markdown_documents(str(outside.resolve()), project_root=project_root)


def test_load_markdown_documents_rejects_symlink_escape(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    documents_directory = project_root / "documents"
    documents_directory.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("private", encoding="utf-8")
    link = documents_directory / "linked.md"

    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symbolic links is not permitted on this system.")

    with pytest.raises(ValueError, match="outside the project root"):
        load_markdown_documents(project_root=project_root)


def test_load_markdown_documents_reports_invalid_utf8(tmp_path: Path) -> None:
    documents_directory = tmp_path / "documents"
    documents_directory.mkdir()
    (documents_directory / "invalid.md").write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        load_markdown_documents(project_root=tmp_path)
