"""Locating the Franka Panda meshes, which neither repository vendors.

``simgrasp.paths`` finds its ``assets/`` directory by walking up from its own
file for a ``pyproject.toml`` next to a ``src/``. That is correct for a checkout
and for an editable install, and **wrong** for the way this project depends on
it: pinned to a git commit, ``simgrasp`` lands in ``site-packages``, where the
walk finds no marker and falls back to a path inside the virtual environment.
The failure is not loud. Assets get fetched to one place and looked for in
another, and the first thing to notice is a scene that will not compile.

``simgrasp`` provides the escape hatch, ``SIMGRASP_ASSETS_DIR``. This module
sets it to this repository's own ``assets/`` directory before ``simgrasp`` is
imported, and fetches the meshes there on demand.

The Menagerie commit is pinned to the same one ``simgrasp`` pins. That is a
parity concern, not housekeeping: a different commit is different robot
geometry, and the gripper has to be identical for the numbers to be comparable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(os.environ.get("SIMGRASP_ASSETS_DIR", REPO_ROOT / "assets"))
PANDA_DIR = ASSETS_DIR / "franka_emika_panda"
PANDA_XML = PANDA_DIR / "panda.xml"

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
# The same commit simgrasp pins. Changing it changes the gripper geometry.
MENAGERIE_COMMIT = "da76818e269b82289eba39808e2fb91d679d6994"
MODEL_SUBDIR = "franka_emika_panda"
REQUIRED_FILES = ("panda.xml", "LICENSE", "assets/link0.stl", "assets/hand.stl")


def configure_assets_dir() -> Path:
    """Point ``simgrasp`` at this repository's assets directory, when there is one.

    Must run before ``simgrasp.paths`` is imported, which resolves its asset
    directory at import time. That is why :mod:`isaacgrasp.bootstrap` calls it.

    Two conditions, both deliberate. It never overrides a variable the user set.
    And it only redirects ``simgrasp`` when this repository actually has the
    meshes: pointing an editable ``simgrasp`` that has its own working assets at
    an empty directory here would break a setup that was fine, which is a worse
    failure than the one this function exists to prevent.
    """
    if "SIMGRASP_ASSETS_DIR" in os.environ:
        return Path(os.environ["SIMGRASP_ASSETS_DIR"])
    if PANDA_XML.exists():
        os.environ["SIMGRASP_ASSETS_DIR"] = str(ASSETS_DIR)
        return ASSETS_DIR
    return ASSETS_DIR


def assets_present() -> bool:
    return PANDA_XML.exists()


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE)


def _sparse_clone(dest: Path) -> None:
    """Clone only the Panda directory at the pinned commit."""
    _run(["git", "init", "-q", str(dest)])
    _run(["git", "remote", "add", "origin", MENAGERIE_URL], cwd=dest)
    _run(["git", "config", "core.sparseCheckout", "true"], cwd=dest)
    (dest / ".git" / "info" / "sparse-checkout").write_text(f"{MODEL_SUBDIR}/*\nLICENSE\n")
    _run(["git", "fetch", "--depth", "1", "origin", MENAGERIE_COMMIT], cwd=dest)
    _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)


def fetch_assets(force: bool = False, retries: int = 4) -> Path:
    """Fetch the Panda MJCF into :data:`PANDA_DIR`, retrying on network failure."""
    configure_assets_dir()
    if assets_present() and not force:
        return PANDA_DIR

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "menagerie"
            try:
                _sparse_clone(work)
                source = work / MODEL_SUBDIR
                if not source.is_dir():
                    raise RuntimeError(f"{MODEL_SUBDIR} missing from the checkout")
                if PANDA_DIR.exists():
                    shutil.rmtree(PANDA_DIR)
                shutil.copytree(source, PANDA_DIR)
                (PANDA_DIR / "panda.png").unlink(missing_ok=True)
                break
            except Exception as exc:  # noqa: BLE001 - retry any network or git failure
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(2 ** (attempt + 1))
    else:
        raise RuntimeError(
            f"could not fetch the Panda meshes after {retries} attempts") from last_error

    missing = [name for name in REQUIRED_FILES if not (PANDA_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"fetch incomplete, missing: {missing}")
    return PANDA_DIR


def require_assets() -> Path:
    """Return the Panda MJCF path, with an actionable error if it is absent."""
    configure_assets_dir()
    if not assets_present():
        raise FileNotFoundError(
            f"Franka Panda MJCF not found at {PANDA_XML}.\n"
            "The robot meshes are vendored from MuJoCo Menagerie and are not "
            "committed to this repository. Fetch them with:\n\n"
            "    python scripts/fetch_assets.py\n"
        )
    return PANDA_XML
