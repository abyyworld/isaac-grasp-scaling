"""Simulator backends.

Both backends present the same interface, which is deliberately ``simgrasp``'s
own: they hand back :class:`simgrasp.env.Observation` and
:class:`simgrasp.env.GraspResult`, and they expose the two attributes the
upstream grasp sampler and oracle policy read off an environment. Everything
downstream (the collector, the scaling driver, the label writer) is therefore
backend-agnostic, and the MuJoCo backend doubles as the reference the Isaac port
is checked against.

``build_backend`` is the only thing that should import a backend module: the
Isaac one pulls in a multi-gigabyte simulator runtime at import time.
"""

from __future__ import annotations

from typing import Any

from .base import EnvView, GraspBackend

__all__ = ["EnvView", "GraspBackend", "build_backend", "BACKENDS"]

BACKENDS = ("mujoco", "isaac")


def build_backend(name: str, **kwargs: Any) -> GraspBackend:
    """Construct a backend by name, importing only the one asked for."""
    if name == "mujoco":
        from .mujoco_backend import MujocoBackend

        return MujocoBackend(**kwargs)
    if name == "isaac":
        from .isaac_backend import IsaacBackend

        return IsaacBackend(**kwargs)
    raise KeyError(f"unknown backend {name!r}; known: {BACKENDS}")
