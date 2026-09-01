"""Read-only Git inspection tools for the configured project repository."""

import subprocess
from pathlib import Path

from config import PROJECT_ROOT
from tools.registry import Tool


class GitInspectionError(RuntimeError):
    """Report that the configured repository could not be inspected."""


def _resolve_project_root(project_root: Path) -> Path:
    """Resolve and validate the directory in which Git will run."""
    resolved_root = project_root.resolve()
    if not resolved_root.exists():
        raise GitInspectionError(f"PROJECT_ROOT does not exist: {resolved_root}")
    if not resolved_root.is_dir():
        raise GitInspectionError(f"PROJECT_ROOT is not a directory: {resolved_root}")
    return resolved_root


def _run_git(
    command: list[str], *, project_root: Path, operation: str
) -> str:
    """Run one internally selected Git command and return its standard output."""
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            shell=False,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise GitInspectionError(
            "Git executable was not found. Install Git and ensure it is on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise GitInspectionError(f"{operation} failed{detail}") from exc
    except OSError as exc:
        raise GitInspectionError(f"Could not execute Git for {operation}: {exc}") from exc

    return completed.stdout.rstrip()


def _require_git_repository(project_root: Path) -> None:
    """Require PROJECT_ROOT to be an inspectable Git working tree."""
    try:
        result = _run_git(
            ["git", "rev-parse", "--is-inside-work-tree"],
            project_root=project_root,
            operation="Git repository check",
        )
    except GitInspectionError as exc:
        if isinstance(exc.__cause__, subprocess.CalledProcessError):
            raise GitInspectionError(
                f"PROJECT_ROOT is not an inspectable Git working tree: {project_root}"
            ) from exc
        raise

    if result.casefold() != "true":
        raise GitInspectionError(
            f"PROJECT_ROOT is not an inspectable Git working tree: {project_root}"
        )


def git_status(*, project_root: Path = PROJECT_ROOT) -> str:
    """Return short staged, unstaged, and untracked working-tree status."""
    resolved_root = _resolve_project_root(project_root)
    _require_git_repository(resolved_root)
    output = _run_git(
        ["git", "status", "--short"],
        project_root=resolved_root,
        operation="Git status",
    )
    return output or "Working tree clean."


def git_diff(
    staged: bool = False, *, project_root: Path = PROJECT_ROOT
) -> str:
    """Return the patch for staged or unstaged tracked-file changes."""
    if type(staged) is not bool:
        raise ValueError("staged must be a boolean.")

    resolved_root = _resolve_project_root(project_root)
    _require_git_repository(resolved_root)
    command = ["git", "diff", "--cached"] if staged else ["git", "diff"]
    output = _run_git(
        command,
        project_root=resolved_root,
        operation="Git diff",
    )
    if output:
        return output
    return "No staged changes." if staged else "No unstaged changes."


def _execute_git_status() -> str:
    """Expose status without accepting model-controlled arguments."""
    return git_status(project_root=PROJECT_ROOT)


def _execute_git_diff(staged: bool = False) -> str:
    """Expose only the controlled staged/unstaged choice to the model."""
    return git_diff(staged=staged, project_root=PROJECT_ROOT)


GIT_STATUS_TOOL = Tool(
    name="git_status",
    description=(
        "Inspect staged, unstaged, and untracked changes in the configured "
        "project repository."
    ),
    argument_schema={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    implementation=_execute_git_status,
)


GIT_DIFF_TOOL = Tool(
    name="git_diff",
    description=(
        "Show the patch for unstaged changes, or staged changes when staged is "
        "true, in the configured project repository."
    ),
    argument_schema={
        "type": "object",
        "properties": {
            "staged": {
                "type": "boolean",
                "description": (
                    "False shows unstaged changes; true shows staged changes."
                ),
                "default": False,
            }
        },
        "required": [],
        "additionalProperties": False,
    },
    implementation=_execute_git_diff,
)
