"""Tests for safe, read-only Git inspection tools."""

import subprocess
from pathlib import Path

import pytest

import config
import tools.git as git_module
from tools.git import (
    GIT_DIFF_TOOL,
    GIT_STATUS_TOOL,
    GitInspectionError,
    git_diff,
    git_status,
)
from tools.registry import ToolRegistry


def _run_test_git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.fixture
def git_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _run_test_git(repository, "init")
    _run_test_git(repository, "config", "user.name", "Test User")
    _run_test_git(repository, "config", "user.email", "test@example.com")
    _run_test_git(repository, "config", "core.autocrlf", "false")
    (repository / "tracked.txt").write_text("original\n", encoding="utf-8")
    _run_test_git(repository, "add", "tracked.txt")
    _run_test_git(repository, "commit", "-m", "Initial commit")
    return repository


def test_git_status_returns_clean_status(git_repository: Path) -> None:
    assert git_status(project_root=git_repository) == "Working tree clean."


def test_git_status_reports_modified_tracked_file(git_repository: Path) -> None:
    (git_repository / "tracked.txt").write_text("modified\n", encoding="utf-8")

    assert git_status(project_root=git_repository) == " M tracked.txt"


def test_git_status_reports_untracked_file(git_repository: Path) -> None:
    (git_repository / "new.txt").write_text("new\n", encoding="utf-8")

    assert git_status(project_root=git_repository) == "?? new.txt"


def test_git_diff_returns_unstaged_patch(git_repository: Path) -> None:
    (git_repository / "tracked.txt").write_text("modified\n", encoding="utf-8")

    result = git_diff(project_root=git_repository)

    assert "-original" in result
    assert "+modified" in result


def test_git_diff_returns_staged_patch(git_repository: Path) -> None:
    (git_repository / "tracked.txt").write_text("staged\n", encoding="utf-8")
    _run_test_git(git_repository, "add", "tracked.txt")

    assert git_diff(project_root=git_repository) == "No unstaged changes."
    staged_result = git_diff(staged=True, project_root=git_repository)
    assert "-original" in staged_result
    assert "+staged" in staged_result


def test_git_diff_returns_explicit_empty_messages(git_repository: Path) -> None:
    assert git_diff(project_root=git_repository) == "No unstaged changes."
    assert git_diff(staged=True, project_root=git_repository) == "No staged changes."


def test_git_tools_use_central_project_root() -> None:
    assert git_module.PROJECT_ROOT is config.PROJECT_ROOT


def test_model_facing_tools_use_configured_project_root(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    monkeypatch.setattr(git_module, "PROJECT_ROOT", git_repository)
    registry = ToolRegistry([GIT_STATUS_TOOL, GIT_DIFF_TOOL])

    assert registry.execute("git_status", {}) == "Working tree clean."
    assert registry.execute("git_diff", {}) == "No unstaged changes."


def test_git_tool_schemas_expose_only_controlled_arguments() -> None:
    status_definition = GIT_STATUS_TOOL.model_definition()
    diff_definition = GIT_DIFF_TOOL.model_definition()

    assert status_definition["arguments"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    assert diff_definition["arguments"]["properties"] == {
        "staged": {
            "type": "boolean",
            "description": "False shows unstaged changes; true shows staged changes.",
            "default": False,
        }
    }
    assert diff_definition["arguments"]["additionalProperties"] is False


def test_registry_rejects_arbitrary_git_arguments() -> None:
    registry = ToolRegistry([GIT_STATUS_TOOL, GIT_DIFF_TOOL])

    with pytest.raises(ValueError, match="Invalid arguments"):
        registry.execute("git_status", {"command": "commit"})
    with pytest.raises(ValueError, match="Invalid arguments"):
        registry.execute("git_diff", {"cwd": ".."})


def test_git_diff_rejects_non_boolean_staged(git_repository: Path) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        git_diff(staged="--stat", project_root=git_repository)  # type: ignore[arg-type]


def test_git_tools_reject_non_git_directory(tmp_path: Path) -> None:
    with pytest.raises(GitInspectionError, match="not an inspectable Git working tree"):
        git_status(project_root=tmp_path)


def test_git_tools_reject_invalid_project_roots(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    file_root = tmp_path / "file.txt"
    file_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(GitInspectionError, match="does not exist"):
        git_status(project_root=missing)
    with pytest.raises(GitInspectionError, match="not a directory"):
        git_status(project_root=file_root)


def test_git_executable_unavailable_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        git_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    with pytest.raises(GitInspectionError, match="Git executable was not found"):
        git_status(project_root=tmp_path)


def test_git_command_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 0, stdout="true\n", stderr="")
        raise subprocess.CalledProcessError(2, command, stderr="status failed")

    monkeypatch.setattr(git_module.subprocess, "run", fake_run)

    with pytest.raises(GitInspectionError, match="Git status failed: status failed"):
        git_status(project_root=git_repository)


def test_subprocess_execution_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        git_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cannot start")),
    )

    with pytest.raises(GitInspectionError, match="Could not execute Git"):
        git_status(project_root=tmp_path)


def test_git_status_uses_fixed_safe_subprocess_arguments(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        stdout = "true\n" if command[1] == "rev-parse" else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(git_module.subprocess, "run", fake_run)

    assert git_status(project_root=git_repository) == "Working tree clean."
    assert [command for command, _ in calls] == [
        ["git", "rev-parse", "--is-inside-work-tree"],
        ["git", "status", "--short"],
    ]
    assert all(kwargs["cwd"] == git_repository.resolve() for _, kwargs in calls)
    assert all(kwargs["shell"] is False for _, kwargs in calls)


@pytest.mark.parametrize(
    ("staged", "expected_command"),
    [
        (False, ["git", "diff"]),
        (True, ["git", "diff", "--cached"]),
    ],
)
def test_git_diff_selects_only_permitted_command(
    monkeypatch: pytest.MonkeyPatch,
    git_repository: Path,
    staged: bool,
    expected_command: list[str],
) -> None:
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        stdout = "true\n" if command[1] == "rev-parse" else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(git_module.subprocess, "run", fake_run)

    git_diff(staged=staged, project_root=git_repository)

    assert commands[-1] == expected_command
