"""The interface both simulators implement.

The interface is batched because that is the entire point of the Isaac Lab port:
one ``execute`` call steps hundreds or thousands of environments together. The
MuJoCo backend has a batch size of one and gets its parallelism from processes,
exactly as the original study did, so the same collector code drives both and
the throughput comparison measures the simulators rather than two different
pipelines.

Why the return types are ``simgrasp``'s
---------------------------------------
:class:`simgrasp.policies.oracle.GraspSampler` defines the label distribution
and :class:`simgrasp.policies.heuristic.HeuristicPolicy` is the control this
whole project is measured against. Reimplementing either would void the
comparison. Both are called as ``policy(observation, env, rng)`` and between
them they read exactly three things off ``env``: ``state``, ``oracle_grasp``
and ``table_z``. :class:`EnvView` is that surface and nothing more, so the
upstream code runs unmodified against either simulator.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from simgrasp.env import EpisodeState, GraspResult, Observation
from simgrasp.grasp import Grasp


@runtime_checkable
class EnvView(Protocol):
    """The slice of an environment that upstream policies read.

    ``simgrasp.env.PandaGraspEnv`` satisfies this already; the Isaac backend
    supplies one view per parallel environment.
    """

    table_z: float
    state: EpisodeState | None

    def oracle_grasp(self, hint_index: int = 0) -> Grasp:
        """The scripted grasp from the object's true settled pose."""
        ...


@runtime_checkable
class GraspBackend(Protocol):
    """Reset a batch of scenes, observe them, execute one grasp in each, score it."""

    name: str
    batch_size: int
    image_size: int
    base_seed: int

    def reset(self, episodes: Sequence[int], split: str = "all") -> list[Observation]:
        """Place a fresh object in each environment and return the RGB-D views.

        ``len(episodes)`` must not exceed :attr:`batch_size`. Episode indices are
        the seeds: episode *i* must spawn the same object in the same pose on
        every run and on every backend that claims to reproduce this scene
        distribution.
        """
        ...

    def views(self) -> list[EnvView]:
        """One :class:`EnvView` per environment in the current batch."""
        ...

    def snapshot(self) -> Any:
        """Capture simulator state so several grasps can share one settled scene.

        This is what makes contrastive orientation labels affordable: the settle
        and the render are paid once for K grasps at the same point.
        """
        ...

    def restore(self, snap: Any) -> None:
        """Restore a :meth:`snapshot`."""
        ...

    def execute(self, grasps: Sequence[Grasp]) -> list[GraspResult]:
        """Run the scripted approach, grasp and lift in every environment."""
        ...

    def close(self) -> None:
        ...


def chunked(items: Sequence[int], size: int) -> list[list[int]]:
    """Split ``items`` into consecutive chunks of at most ``size``."""
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


# --------------------------------------------------------------------------- #
# The oracle grasp, expressed from episode state alone.
# --------------------------------------------------------------------------- #
#
# ``simgrasp.env.PandaGraspEnv.oracle_grasp`` is a method on the MuJoCo
# environment, but the computation behind it is pure geometry over
# ``EpisodeState``: rotate the catalogue's grasp hint into the world by the
# object's settled yaw, then pick a TCP height. The Isaac backend has no
# ``PandaGraspEnv`` to call, so the same computation is expressed here over the
# same state and through the same upstream helpers.
#
# This is the one piece of upstream logic that is restated rather than imported,
# and it is the centre of the label distribution, so it is pinned:
# ``tests/test_oracle_view.py`` asserts it agrees with the MuJoCo method to
# floating-point equality over the whole catalogue.


def oracle_grasp_for_state(state: EpisodeState, table_z: float,
                           hint_index: int = 0) -> Grasp:
    """The scripted grasp for ``state``, identical to ``PandaGraspEnv.oracle_grasp``."""
    import numpy as np
    from simgrasp.grasp import grasp_z_from_surface
    from simgrasp.transforms import wrap_grasp_angle

    hint = state.spec.grasp_hints[hint_index % len(state.spec.grasp_hints)]
    cos_yaw, sin_yaw = np.cos(state.object_yaw), np.sin(state.object_yaw)
    hx, hy = hint.xy
    x = state.object_xy[0] + cos_yaw * hx - sin_yaw * hy
    y = state.object_xy[1] + sin_yaw * hx + cos_yaw * hy
    z = grasp_z_from_surface(state.object_z0 + hint.surface_z, table_z,
                             mid_z_world=state.object_z0 + hint.grasp_z)
    yaw = float(wrap_grasp_angle(state.object_yaw + hint.yaw))
    return Grasp(x=float(x), y=float(y), z=float(z), yaw=yaw, width=float(hint.width))


class SimpleEnvView:
    """An :class:`EnvView` backed by nothing but an :class:`EpisodeState`.

    One of these stands in for a ``PandaGraspEnv`` per parallel environment, so
    the upstream grasp sampler and oracle policy run unmodified against a
    backend that has no per-environment simulator object at all.
    """

    def __init__(self, table_z: float, state: EpisodeState | None = None):
        self.table_z = float(table_z)
        self.state = state

    def oracle_grasp(self, hint_index: int = 0) -> Grasp:
        if self.state is None:
            raise RuntimeError("no episode state; call the backend's reset() first")
        return oracle_grasp_for_state(self.state, self.table_z, hint_index)


# --------------------------------------------------------------------------- #
# Outcome classification.
# --------------------------------------------------------------------------- #
#
# ``PandaGraspEnv.execute`` classifies an attempt inline. The Isaac backend has
# to reach the same verdict from the same two measurements, so the rule is
# written once here and checked against the MuJoCo environment's own output in
# ``tests/test_outcome_classification.py``.

# Fingers fully closed means they met nothing, so the grasp missed.
CLOSED_EMPTY_WIDTH = 1e-3
# Any lift at all, but not enough to count: the object was held and lost.
DROPPED_MIN_LIFT = 0.005


def classify_outcome(lift_height: float, final_gripper_width: float,
                     lift_threshold: float) -> tuple[bool, str]:
    """Map a finished attempt to ``(success, reason)``.

    ``lift_threshold`` is passed in rather than read from a constant so that the
    caller has to source it from ``simgrasp.env.LIFT_SUCCESS_THRESHOLD``; a
    backend cannot quietly grade itself on an easier criterion.
    """
    if lift_height > lift_threshold:
        return True, "success"
    if final_gripper_width < CLOSED_EMPTY_WIDTH:
        return False, "closed_empty"
    if lift_height > DROPPED_MIN_LIFT:
        return False, "dropped"
    return False, "no_lift"
