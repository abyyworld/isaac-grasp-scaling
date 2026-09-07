"""MuJoCo backend: a thin batch-of-one adapter over the original environment.

This backend adds no physics of its own. It exists for three reasons:

1. **It is the reference.** The Isaac port has to reproduce this scene
   distribution and this success criterion; without a runnable reference there
   is nothing to check the port against.
2. **It separates two variables.** Training on Isaac-generated data and
   evaluating in MuJoCo confounds "more data" with "different simulator". Being
   able to run the same scaling experiment inside MuJoCo alone is what makes the
   Isaac arm of the curve interpretable.
3. **It runs anywhere.** MuJoCo renders offscreen on a CPU, so the collector,
   the scaling driver and the whole test suite can be exercised without a GPU.

It is a batch of one on purpose. The original study got its parallelism from
processes, and copying that exactly is what makes the samples-per-hour
comparison a comparison of simulators rather than of two different pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from simgrasp.env import GraspResult, Observation, PandaGraspEnv
from simgrasp.grasp import Grasp

from .base import EnvView


class MujocoBackend:
    """One :class:`simgrasp.env.PandaGraspEnv`, presented as a batch of size one."""

    name = "mujoco"
    batch_size = 1

    def __init__(self, image_size: int = 224, base_seed: int = 0, **_ignored: Any):
        self.image_size = int(image_size)
        self.base_seed = int(base_seed)
        self.env = PandaGraspEnv(image_size=self.image_size, base_seed=self.base_seed)

    # -- lifecycle ------------------------------------------------------------ #
    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> MujocoBackend:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- interface ------------------------------------------------------------ #
    def reset(self, episodes: Sequence[int], split: str = "all") -> list[Observation]:
        self._require_batch(episodes)
        return [self.env.reset(int(episodes[0]), split=split)]

    def views(self) -> list[EnvView]:
        return [self.env]

    def snapshot(self) -> Any:
        return self.env.snapshot()

    def restore(self, snap: Any) -> None:
        self.env.restore(snap)

    def execute(self, grasps: Sequence[Grasp]) -> list[GraspResult]:
        self._require_batch(grasps)
        return [self.env.execute(grasps[0])]

    def _require_batch(self, items: Sequence[Any]) -> None:
        if len(items) != 1:
            raise ValueError(
                f"{self.name} backend has batch_size 1, got {len(items)} items. "
                "Parallelism for this backend comes from processes, not from a batch."
            )
