"""Tests for read-only filesystem tools and project-root safety."""

from pathlib import Path

import pytest

from tools.filesystem import (
    LIST_FILES_TOOL,
    READ_FILE_TOOL,
    list_files,
    read_file,
)
from tools.registry import ToolRegistry


def test_list_files_returns_sorted_project_relative_entries(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "nested").mkdir()
    (project_root / "zebra.txt").write_text("z", encoding="utf-8")
    (project_root / "alpha.txt").write_text("a", encoding="utf-8")

    assert list_files(".", project_root=project_root) == [
        "alpha.txt",
        "nested",
        "zebra.txt",
    ]


def test_list_files_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        list_files("missing", project_root=tmp_path)


def test_list_files_rejects_file_path(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="not a directory"):
        list_files("notes.txt", project_root=tmp_path)


def test_read_file_returns_utf8_text(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Olá, filesystem!", encoding="utf-8")

    assert read_file("README.md", project_root=tmp_path) == "Olá, filesystem!"


def test_read_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_file("missing.txt", project_root=tmp_path)


def test_read_file_rejects_directory(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()

    with pytest.raises(IsADirectoryError, match="is a directory"):
        read_file("docs", project_root=tmp_path)


def test_filesystem_tools_reject_parent_traversal(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (tmp_path / "outside.txt").write_text("private", encoding="utf-8")

    with pytest.raises(ValueError, match="Path traversal"):
        read_file("../outside.txt", project_root=project_root)
    with pytest.raises(ValueError, match="Path traversal"):
        list_files("..", project_root=project_root)


def test_filesystem_tools_reject_absolute_path_outside_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("private", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the project root"):
        read_file(str(outside_file), project_root=project_root)


def test_filesystem_tools_reject_symlink_escape(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("private", encoding="utf-8")
    link = project_root / "outside-link.txt"

    try:
        link.symlink_to(outside_file)
    except OSError:
        pytest.skip("Creating symbolic links is not permitted on this system.")

    with pytest.raises(ValueError, match="outside the project root"):
        read_file("outside-link.txt", project_root=project_root)


def test_registry_exposes_and_executes_filesystem_tools() -> None:
    registry = ToolRegistry([LIST_FILES_TOOL, READ_FILE_TOOL])
    definitions = registry.model_definitions()

    assert [definition["name"] for definition in definitions] == [
        "list_files",
        "read_file",
    ]
    assert definitions[0]["arguments"]["required"] == ["path"]
    assert definitions[1]["arguments"]["required"] == ["path"]
    assert "README.md" in registry.execute("list_files", {"path": "."})
    assert "agentic software engineering assistant" in registry.execute(
        "read_file", {"path": "README.md"}
    )
