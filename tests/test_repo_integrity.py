"""Guards against the bug that made the predecessor repository unusable.

Its ``.gitignore`` carried an unanchored ``data/`` pattern intended for the
generated dataset directory at the repository root. Git applies an unanchored
pattern at every level, so it also matched ``src/simgrasp/data/`` and silently
excluded that package from all 31 commits: a fresh clone could not import the
project at all, while every local test passed and CI stayed green.

This repository has a ``/data/`` pattern for the same purpose, so it is one
missing slash away from the same failure.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "scripts", "tests")


def _is_git_repo() -> bool:
    try:
        subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-dir"],
                       capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


requires_git = pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")


@requires_git
def test_no_source_file_is_gitignored():
    candidates = [p for d in SOURCE_DIRS for p in (ROOT / d).rglob("*.py")
                  if "__pycache__" not in p.parts]
    assert candidates, "found no source files to check"
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--no-index", *map(str, candidates)],
        capture_output=True, text=True)
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert not ignored, (
        "these source files are excluded by .gitignore and would not be committed:\n  "
        + "\n  ".join(ignored) + "\nAnchor the offending pattern with a leading slash.")


@requires_git
def test_every_package_directory_is_tracked():
    tracked = set(subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True,
                                 text=True, check=True).stdout.splitlines())
    for init in (ROOT / "src").rglob("__init__.py"):
        rel = init.relative_to(ROOT).as_posix()
        assert rel in tracked, f"{rel} is not tracked; a fresh clone cannot import it"


@pytest.mark.parametrize("module", [
    "isaacgrasp", "isaacgrasp.parity", "isaacgrasp.collect", "isaacgrasp.scaling",
    "isaacgrasp.throughput", "isaacgrasp.plot", "isaacgrasp.angle",
    "isaacgrasp.backends", "isaacgrasp.backends.base",
    "isaacgrasp.backends.isaac_objects", "isaacgrasp.backends.mujoco_backend",
])
def test_module_imports(module):
    assert importlib.import_module(module) is not None


def test_isaac_modules_are_importable_as_source():
    """The Isaac modules cannot be imported without Isaac Sim, but must parse.

    They are the part of the port that has never been executed, so the least
    that can be checked off a GPU is that they are syntactically valid and that
    nothing has been left half-edited.
    """
    import ast

    for name in ("isaac_backend.py", "isaac_scene.py"):
        path = ROOT / "src" / "isaacgrasp" / "backends" / name
        tree = ast.parse(path.read_text())
        assert any(isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in tree.body)


def test_the_isaac_backend_is_marked_unverified():
    """The README's honesty depends on this marker staying in the source."""
    source = (ROOT / "src" / "isaacgrasp" / "backends" / "isaac_backend.py").read_text()
    assert "NOT YET EXECUTED" in source
