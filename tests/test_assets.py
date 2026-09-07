"""The asset path, which is a silent failure waiting to happen.

``simgrasp.paths`` finds its ``assets/`` directory by walking up from its own
file for a ``pyproject.toml`` beside a ``src/``. That works for a checkout and
for an editable install. This project depends on ``simgrasp`` pinned to a git
commit, which lands it in ``site-packages``, where the walk finds no marker and
falls back to a path inside the virtual environment. Meshes then get fetched to
one directory and looked for in another, and the first symptom is a scene that
will not compile.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from isaacgrasp import assets

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_menagerie_commit_matches_simgrasp():
    """Different commit, different gripper geometry, incomparable numbers.

    The pin is read out of simgrasp's own fetch script when that script is
    available (a checkout or an editable install) and skipped when it is not,
    rather than duplicating the hash in a second place that could drift.
    """
    import simgrasp.paths

    candidate = Path(simgrasp.paths.REPO_ROOT) / "scripts" / "fetch_assets.py"
    if not candidate.exists():
        pytest.skip("simgrasp is not installed from a checkout, so its pin is not readable")
    source = candidate.read_text()
    assert assets.MENAGERIE_COMMIT in source, (
        "this repository pins a different MuJoCo Menagerie commit than simgrasp does, "
        "so the two would use different robot geometry")


def test_configure_respects_an_explicit_setting(monkeypatch):
    monkeypatch.setenv("SIMGRASP_ASSETS_DIR", "/somewhere/else")
    assert assets.configure_assets_dir() == Path("/somewhere/else")
    assert os.environ["SIMGRASP_ASSETS_DIR"] == "/somewhere/else"


def test_configure_redirects_only_when_the_meshes_are_here(monkeypatch, tmp_path):
    """Redirecting simgrasp at an empty directory would break a working setup."""
    monkeypatch.delenv("SIMGRASP_ASSETS_DIR", raising=False)
    monkeypatch.setattr(assets, "PANDA_XML", tmp_path / "absent" / "panda.xml")
    assets.configure_assets_dir()
    assert "SIMGRASP_ASSETS_DIR" not in os.environ, (
        "with no local meshes, simgrasp must be left to resolve its own path")


def test_configure_sets_the_variable_when_the_meshes_are_here(monkeypatch, tmp_path):
    monkeypatch.delenv("SIMGRASP_ASSETS_DIR", raising=False)
    present = tmp_path / "franka_emika_panda" / "panda.xml"
    present.parent.mkdir(parents=True)
    present.write_text("<mujoco/>")
    monkeypatch.setattr(assets, "PANDA_XML", present)
    monkeypatch.setattr(assets, "ASSETS_DIR", tmp_path)
    assert assets.configure_assets_dir() == tmp_path
    assert os.environ["SIMGRASP_ASSETS_DIR"] == str(tmp_path)


def test_require_assets_names_the_fix(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMGRASP_ASSETS_DIR", str(tmp_path))
    monkeypatch.setattr(assets, "PANDA_XML", tmp_path / "nothing.xml")
    with pytest.raises(FileNotFoundError, match="scripts/fetch_assets.py"):
        assets.require_assets()


@pytest.mark.skipif(not assets.assets_present(), reason="meshes not fetched")
def test_simgrasp_finds_the_same_meshes_this_package_configured():
    """The end to end property: both packages agree on where the robot is."""
    import simgrasp.paths

    assert simgrasp.paths.PANDA_XML.exists()
    assert simgrasp.paths.PANDA_XML.resolve() == assets.PANDA_XML.resolve()


@pytest.mark.skipif(not assets.assets_present(), reason="meshes not fetched")
def test_the_scene_actually_compiles_with_them():
    """Reading the path proves nothing; building the model proves it."""
    from simgrasp.scene import build_template_model

    model = build_template_model()
    assert model.ngeom > 0 and model.nq > 0
