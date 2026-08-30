"""Tests for exact, case-insensitive code search."""

from pathlib import Path

import pytest

from tools.search import SEARCH_CODE_TOOL, search_code


def test_search_code_returns_file_line_number_and_matching_line(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "auth.py").write_text(
        "def authenticate():\n    return 'AUTHENTICATION passed'\n",
        encoding="utf-8",
    )

    assert search_code("authentication", project_root=project_root) == [
        {
            "filename": "auth.py",
            "line_number": 2,
            "line": "    return 'AUTHENTICATION passed'",
        }
    ]


def test_search_code_only_reads_paths_inside_configured_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "inside.py").write_text("needle = True\n", encoding="utf-8")
    (tmp_path / "outside.py").write_text("needle = False\n", encoding="utf-8")

    assert search_code("needle", project_root=project_root) == [
        {"filename": "inside.py", "line_number": 1, "line": "needle = True"}
    ]


def test_search_code_rejects_symlink_outside_configured_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("private_needle = True\n", encoding="utf-8")
    link = project_root / "linked.py"

    try:
        link.symlink_to(outside_file)
    except OSError:
        pytest.skip("Creating symbolic links is not permitted on this system.")

    assert search_code("private_needle", project_root=project_root) == []


def test_search_code_rejects_empty_query(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        search_code("", project_root=tmp_path)


def test_search_code_tool_schema() -> None:
    definition = SEARCH_CODE_TOOL.model_definition()

    assert definition["name"] == "search_code"
    assert definition["arguments"]["required"] == ["query"]
