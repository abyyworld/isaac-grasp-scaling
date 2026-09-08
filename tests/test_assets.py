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


def _simgrasp_checkout() -> Path | None:
    """A simgrasp checkout on this machine, or None if simgrasp came from a wheel.

    ``simgrasp.paths.REPO_ROOT`` walks up from its own file for a pyproject.toml
    beside a ``src/``. From site-packages that walk does not stop at the virtual
    environment: it keeps going and lands on whatever project is *using*
    simgrasp, which is this one. This repository has a ``scripts/fetch_assets.py``
    of its own, so the walk found our file and the pin was compared against
    itself. Anything returned here has to look like simgrasp and not like us.
    """
    import simgrasp.paths

    candidates = []
    override = os.environ.get("SIMGRASP_CHECKOUT")
    if override:
        candidates.append(Path(override))
    candidates.append(Path(simgrasp.paths.REPO_ROOT))

    for root in candidates:
        root = root.resolve()
        if root == REPO_ROOT:
            continue
        if (root / "src" / "simgrasp" / "paths.py").is_file() and \
                (root / "scripts" / "fetch_assets.py").is_file():
            return root
    return None


def test_the_menagerie_commit_matches_simgrasp():
    """Different commit, different gripper geometry, incomparable numbers.

    The pin is read out of simgrasp's own fetch script when a checkout is on
    this machine, rather than duplicating the hash in a second place that could
    drift. simgrasp ships only ``src/``, so from a pinned git install the script
    is genuinely absent and there is nothing to compare against: point
    ``SIMGRASP_CHECKOUT`` at a checkout to make this run.
    """
    checkout = _simgrasp_checkout()
    if checkout is None:
        pytest.skip("simgrasp is not installed from a checkout, so its pin is not readable")

    source = (checkout / "scripts" / "fetch_assets.py").read_text()
    assert assets.MENAGERIE_COMMIT in source, (
        "this repository pins a different MuJoCo Menagerie commit than simgrasp does, "
        "so the two would use different robot geometry")


def test_this_repository_is_never_mistaken_for_the_simgrasp_checkout():
    """The bug the check above had: it compared our pin against our own file.

    A comparison that can only ever agree with itself is worse than no
    comparison, because it reads in the report as a pass.
    """
    assert _simgrasp_checkout() != REPO_ROOT


def test_the_pin_is_written_down_in_exactly_one_place():
    """A second copy of the hash is a second thing to forget to update.

    ``scripts/fetch_assets.py`` imports the constant rather than repeating it,
    and this is what keeps that true.
    """
    home = REPO_ROOT / "src" / "isaacgrasp" / "assets.py"
    strays = [
        str(path.relative_to(REPO_ROOT))
        for path in REPO_ROOT.rglob("*")
        if path.is_file()
        and path != home
        and path.suffix in {".py", ".md", ".toml", ".sh", ".ipynb"}
        and ".venv" not in path.parts
        and assets.MENAGERIE_COMMIT in path.read_text(errors="ignore")
    ]
    assert not strays, (
        "the Menagerie commit is hardcoded outside isaacgrasp.assets, in "
        f"{strays}; import MENAGERIE_COMMIT instead so there is one copy to update")


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
