"""Pick a working MuJoCo render backend, and dodge the OSMesa/Triton segfault.

Importing this module has side effects by design. Every entry point that might
touch the MuJoCo backend starts with ``from isaacgrasp import bootstrap``.

Why this exists at all
----------------------
On headless Linux without a GPU driver, MuJoCo falls back to OSMesa software
rendering. Mesa's ``llvmpipe`` and the Triton compiler bundled inside the CUDA
PyTorch wheel each load their own copy of LLVM, and whichever loads second
segfaults the process: no Python traceback, no exception, just a stack dump.

This was not a theoretical risk. Writing this package, ``tests/test_collect.py``
died exactly that way the first time it ran, after the MuJoCo backend had
rendered a frame and pytest then imported torchvision. The predecessor
repository documents the same failure and guards against it in its ``scripts/``
directory, which pip does not install, so the guard had to be rebuilt here as a
real module.

Importing torch **and** Triton before any GL context exists avoids it. Importing
torch alone is not enough: torch loads Triton lazily through ``torch._dynamo``,
so the clash simply moves to whatever line first touches dynamo.
"""

from __future__ import annotations

import ctypes.util
import os
import platform
import subprocess
import sys

_PROBE = (
    "import mujoco;"
    "c = mujoco.GLContext(64, 64); c.make_current();"
    "m = mujoco.MjModel.from_xml_string("
    "'<mujoco><worldbody><geom type=\"plane\" size=\"1 1 .1\"/></worldbody></mujoco>');"
    "r = mujoco.Renderer(m, 32, 32); r.update_scene(mujoco.MjData(m)); r.render();"
    "print('ok')"
)


def _backend_works(name: str) -> bool:
    """Render one frame in a subprocess under ``MUJOCO_GL=name``.

    A subprocess, because MuJoCo binds its GL backend at import time: once it has
    been imported with a broken backend this process cannot try another.
    """
    env = {**os.environ, "MUJOCO_GL": name}
    try:
        done = subprocess.run([sys.executable, "-c", _PROBE], env=env, timeout=90,
                              capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and "ok" in done.stdout


def choose_gl_backend() -> str:
    """Select a MuJoCo GL backend, probing rather than guessing.

    A cloud image routinely ships ``libEGL.so`` with no driver behind it, so the
    presence of the library says nothing about whether a context can be created.
    The probe renders a real frame. Returns the backend name, or the empty string
    when MuJoCo's own platform default is used.
    """
    if "MUJOCO_GL" in os.environ:
        return os.environ["MUJOCO_GL"]
    if platform.system() in ("Windows", "Darwin"):
        return ""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return ""

    candidates = ["egl", "osmesa"] if ctypes.util.find_library("EGL") else ["osmesa"]
    for name in candidates:
        if _backend_works(name):
            os.environ["MUJOCO_GL"] = name
            return name

    raise RuntimeError(
        "No working MuJoCo render backend on this machine.\n"
        "On headless Linux install one of:\n"
        "    sudo apt-get install -y libosmesa6      # software, works anywhere\n"
        "    sudo apt-get install -y libegl1         # needs a GPU driver\n"
        "Or set MUJOCO_GL yourself."
    )


def preimport_torch_if_needed(backend: str) -> None:
    """Load torch and Triton before any GL context exists, on software rendering.

    EGL does not clash with Triton's LLVM, so this only runs on the OSMesa
    fallback, where a two-second import is irrelevant next to OSMesa's rendering
    cost. Entry points that never touch torch still pay it, which is the right
    trade against a segfault with no traceback.
    """
    if backend != "osmesa":
        return
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    try:
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001 - a broken triton must not break the simulator
        pass


def setup() -> str:
    """Point simgrasp at this repository's assets, choose a backend, apply the guard.

    The assets step has to come first and has to happen before anything imports
    ``simgrasp.paths``, which resolves its own asset directory at import time.
    """
    from .assets import configure_assets_dir

    configure_assets_dir()
    backend = choose_gl_backend()
    preimport_torch_if_needed(backend)
    return backend


setup()
