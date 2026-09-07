"""Dataset generation, written once and driven by either simulator.

The label semantics are ``simgrasp``'s and are not restated here. The grasp
sampler that defines the label distribution, the angle-variant spread that makes
orientation labels contrastive, the label record and the shard format are all
imported from the pinned commit. What this module adds is the batching: it walks
episodes in chunks of ``backend.batch_size`` so one Isaac Lab step drives
hundreds of environments, while the MuJoCo backend takes the same path with a
batch of one.

Being one code path is the point. The throughput comparison in
``isaacgrasp.throughput`` is only a comparison of simulators if both are driven
by the same collector; two separately-tuned pipelines would measure the tuning.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from simgrasp.collect import EpisodeLabel, _angle_variants
from simgrasp.data.writer import ShardWriter
from simgrasp.grasp import grasp_to_image
from simgrasp.objects import SEEN_CATEGORIES
from simgrasp.policies import GraspSampler
from simgrasp.seeding import rng_for_episode

from .backends import build_backend
from .backends.base import chunked
from .parity import manifest

# The offset the original study uses to derive a per-episode sampler stream that
# is independent of the scene stream, so changing the policy never changes which
# object is spawned. Imported by value because it is a magic number in upstream's
# collector rather than a named constant.
SAMPLER_SEED_OFFSET = 7_777_777


@dataclass
class CollectStats:
    """Counts and timings for one collection run."""

    n_samples: int = 0
    n_scenes: int = 0
    successes: int = 0
    unstable: int = 0
    ik_failed: int = 0
    elapsed: float = 0.0
    by_category: dict[str, list[int]] = field(default_factory=dict)
    by_mode: dict[str, list[int]] = field(default_factory=dict)

    def record(self, category: str, mode: str, success: bool) -> None:
        self.n_samples += 1
        self.successes += int(success)
        for table, key in ((self.by_category, category), (self.by_mode, mode)):
            row = table.setdefault(key, [0, 0])
            row[0] += 1
            row[1] += int(success)

    def as_dict(self) -> dict[str, Any]:
        def rates(table: dict[str, list[int]]) -> dict[str, dict[str, float]]:
            return {k: {"n": n, "positives": s, "positive_rate": s / n if n else 0.0}
                    for k, (n, s) in sorted(table.items())}

        return {
            "n": self.n_samples,
            "n_scenes": self.n_scenes,
            "successes": self.successes,
            "unstable": self.unstable,
            "ik_failed": self.ik_failed,
            "positive_rate": self.successes / self.n_samples if self.n_samples else 0.0,
            "by_category": rates(self.by_category),
            "by_mode": rates(self.by_mode),
            "elapsed_seconds": self.elapsed,
            "samples_per_second": self.n_samples / self.elapsed if self.elapsed else 0.0,
            "samples_per_hour": 3600.0 * self.n_samples / self.elapsed if self.elapsed else 0.0,
            "scenes_per_second": self.n_scenes / self.elapsed if self.elapsed else 0.0,
        }


def collect(
    backend_name: str,
    out_dir: Path | str,
    n_scenes: int,
    split: str = "seen",
    base_seed: int = 0,
    image_size: int = 224,
    angles_per_scene: int = 3,
    shard_size: int = 256,
    store_rgb: bool = True,
    episode_offset: int = 0,
    worker_id: int = 0,
    backend_kwargs: dict[str, Any] | None = None,
    sampler_kwargs: dict[str, Any] | None = None,
    progress_every: int = 50,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate ``n_scenes`` scenes of labelled grasps into ``out_dir``.

    ``angles_per_scene`` executes several grasps at the *same* point in the same
    settled scene at orientations spread over the half turn. That is the fix the
    original study measured for its orientation collapse: with one label per
    image every label is explainable by a function of the pixel alone, and
    predicting the angle-marginal success rate is a loss minimum. It is also
    cheap here, because the settle and the render are shared across the K grasps.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = GraspSampler(**(sampler_kwargs or {}))
    backend = build_backend(backend_name, image_size=image_size, base_seed=base_seed,
                            **(backend_kwargs or {}))
    stats = CollectStats()
    episodes = list(range(episode_offset, episode_offset + n_scenes))
    started = time.perf_counter()

    try:
        with ShardWriter(out_dir, image_size=image_size, shard_size=shard_size,
                         prefix=f"w{worker_id:02d}", store_rgb=store_rgb) as writer:
            for done, chunk in enumerate(chunked(episodes, backend.batch_size)):
                _collect_chunk(backend, sampler, writer, chunk, split, base_seed,
                               angles_per_scene, stats)
                stats.n_scenes += len(chunk)
                if verbose and progress_every and (done + 1) % progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(f"  [{backend_name} w{worker_id:02d}] {stats.n_scenes}/{n_scenes} "
                          f"scenes, {stats.n_samples} samples, positive rate "
                          f"{stats.successes / max(stats.n_samples, 1):.1%}, "
                          f"{stats.n_samples / max(elapsed, 1e-9):.1f} samples/s", flush=True)
    finally:
        backend.close()

    stats.elapsed = time.perf_counter() - started
    meta = _build_meta(backend_name, out_dir, n_scenes, base_seed, image_size, split,
                       angles_per_scene, shard_size, store_rgb, sampler, stats)
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return meta


def _collect_chunk(backend, sampler, writer, chunk, split, base_seed,
                   angles_per_scene, stats: CollectStats) -> None:
    """One batch: reset, sample a grasp per environment, execute K orientations."""
    observations = backend.reset(chunk, split=split)
    views = backend.views()
    snapshot = backend.snapshot()

    rngs = [rng_for_episode(base_seed + SAMPLER_SEED_OFFSET, ep) for ep in chunk]
    bases, modes = [], []
    for obs, view, rng in zip(observations, views, rngs, strict=True):
        bases.append(sampler(obs, view, rng))
        # last_mode is set per call, so it has to be read straight away.
        modes.append(sampler.last_mode)

    variants = [_angle_variants(base, angles_per_scene, rng)
                for base, rng in zip(bases, rngs, strict=True)]

    for k in range(angles_per_scene):
        if k:
            backend.restore(snapshot)
        grasps = [v[k] for v in variants]
        results = backend.execute(grasps)
        for i, result in enumerate(results):
            if result.reason == "unstable":
                # The solver diverged; anything read out of this state is
                # meaningless, so it is dropped rather than written mislabelled.
                stats.unstable += 1
                continue
            if result.reason == "ik_failed":
                stats.ik_failed += 1
            state = views[i].state
            obs = observations[i]
            image = grasp_to_image(grasps[i], obs.cam_pos, obs.cam_mat, obs.intrinsics)
            writer.add(obs.rgb, obs.height, EpisodeLabel(
                episode=int(chunk[i]), category=state.category,
                split="seen" if state.category in SEEN_CATEGORIES else "unseen",
                u=image.u, v=image.v, angle=image.angle, width_px=image.width_px,
                depth=image.depth,
                world_x=grasps[i].x, world_y=grasps[i].y, world_z=grasps[i].z,
                world_yaw=grasps[i].yaw, world_width=grasps[i].width,
                success=bool(result.success), reason=result.reason,
                lift_height=float(result.lift_height), sampler_mode=modes[i],
                object_top_z=float(state.spec.top_z),
                settle_displacement=float(state.settle_displacement),
                angle_index=k,
            ))
            stats.record(state.category, modes[i], bool(result.success))


def _build_meta(backend_name, out_dir, n_scenes, base_seed, image_size, split,
                angles_per_scene, shard_size, store_rgb, sampler, stats) -> dict[str, Any]:
    return {
        "backend": backend_name,
        "out_dir": str(out_dir),
        "n_scenes": n_scenes,
        "n_episodes": n_scenes,  # the key simgrasp's loader looks for
        "image_size": image_size,
        "base_seed": base_seed,
        "split": split,
        "angles_per_scene": angles_per_scene,
        "shard_size": shard_size,
        "store_rgb": store_rgb,
        "sampler_weights": sampler.weights,
        "stats": stats.as_dict(),
        "parity": manifest(),
        "provenance": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
