"""Tests for central project-root configuration."""

import importlib.util
from pathlib import Path

import pytest

import config
from tools import filesystem
from tools import search


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.py"


def _load_project_root() -> Path:
    spec = importlib.util.spec_from_file_location("test_config_module", CONFIG_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROJECT_ROOT


def test_project_root_defaults_to_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROJECT_ROOT", raising=False)

    assert _load_project_root() == CONFIG_PATH.parent.resolve()


def test_project_root_uses_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured_root = tmp_path / "configured-project"
    configured_root.mkdir()
    monkeypatch.setenv("PROJECT_ROOT", str(configured_root / ".." / configured_root.name))

    project_root = _load_project_root()

    assert project_root == configured_root.resolve()
    assert project_root.is_absolute()


def test_filesystem_uses_central_project_root() -> None:
    assert filesystem.PROJECT_ROOT is config.PROJECT_ROOT


def test_code_search_uses_central_project_root() -> None:
    assert search.PROJECT_ROOT is config.PROJECT_ROOT
