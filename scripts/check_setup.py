#!/usr/bin/env python3
"""Verify the install by running it, one stage at a time.

    python scripts/check_setup.py             # everything that works without a GPU
    python scripts/check_setup.py --isaac     # also exercise the Isaac Lab backend

Run this first on a freshly rented GPU box, before starting anything that takes
hours. The useful question on a new machine is not "does it work" but "which of
the things it needs is missing", so each stage is separately named, separately
timed, and the run stops at the first failure with the fix rather than a
traceback.

The Isaac stages are the ones that have never been executed anywhere. Stage
``isaac:object geometry`` is the highest risk in the whole port: it checks
whether PhysX updates an analytic shape's collision geometry when the prim is
rescaled at runtime. If that stage fails, the fallback is in docs/design.md and
it is a change to one module.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaacgrasp import bootstrap  # noqa: E402,F401  (picks a render backend)

PASS, FAIL, SKIP = "ok", "FAILED", "skip"


class Stages:
    """Run named checks in order, stopping at the first failure."""

    def __init__(self) -> None:
        self.failed = False
        self.results: list[tuple[str, str, str]] = []

    def run(self, name: str, func) -> None:
        if self.failed:
            self.results.append((name, SKIP, "not reached"))
            print(f"  [{SKIP}] {name:<34} not reached")
            return
        started = time.perf_counter()
        try:
            detail = func() or ""
        except Exception as exc:  # noqa: BLE001 - the point is to report any failure
            self.failed = True
            detail = f"{type(exc).__name__}: {exc}"
            self.results.append((name, FAIL, detail))
            print(f"  [{FAIL}] {name:<34} {detail}")
            return
        elapsed = (time.perf_counter() - started) * 1000.0
        self.results.append((name, PASS, detail))
        print(f"  [{PASS}]   {name:<34} {detail}  ({elapsed:.0f} ms)")


def check_imports() -> str:
    import simgrasp
    import torch

    import isaacgrasp

    return (f"isaacgrasp {isaacgrasp.__version__}, "
            f"simgrasp {getattr(simgrasp, '__version__', '?')}, torch {torch.__version__}")


def check_parity() -> str:
    from isaacgrasp.parity import check_parity as run

    report = run()
    if not report.ok:
        raise RuntimeError(report.describe())
    return f"{len(report.matched)} constants match the pinned simgrasp commit"


def check_torch_device() -> str:
    import torch
    from simgrasp.models.device import pick_device

    device = pick_device()
    if device.type == "cuda":
        return f"{device}, {torch.cuda.get_device_name(0)}"
    return f"{device} (training will be slow; this is fine for the MuJoCo control)"


def check_mujoco_backend() -> str:
    from simgrasp.policies import HeuristicPolicy
    from simgrasp.seeding import rng_for_episode

    from isaacgrasp.backends import build_backend

    backend = build_backend("mujoco", image_size=128, base_seed=0)
    try:
        observation = backend.reset([0], split="all")[0]
        if float(observation.height.max()) <= 0.005:
            raise RuntimeError(
                "the height map is flat, so the camera saw no object. On headless "
                "Linux this usually means the GL backend is rendering nothing; try "
                "MUJOCO_GL=osmesa after installing libosmesa6.")
        view = backend.views()[0]
        result = backend.execute([HeuristicPolicy()(observation, view, rng_for_episode(1, 0))])[0]
        return (f"{view.state.category}, object {100 * observation.height.max():.1f} mm tall, "
                f"heuristic grasp {result.reason}")
    finally:
        backend.close()


def check_collector(tmp_dir: Path) -> str:
    from isaacgrasp.collect import collect

    meta = collect("mujoco", tmp_dir, n_scenes=2, split="seen", image_size=96,
                   angles_per_scene=2, verbose=False)
    stats = meta["stats"]
    if stats["n"] == 0:
        raise RuntimeError("the collector wrote no samples")
    return f"{stats['n']} samples at {stats['samples_per_hour']:,.0f} samples/hour"


# --------------------------------------------------------------------------- #
# Isaac Lab stages. None of these has ever been executed.
# --------------------------------------------------------------------------- #

_ISAAC: dict[str, object] = {}


def check_isaac_import() -> str:
    import isaaclab
    import isaacsim  # noqa: F401
    from isaaclab.app import AppLauncher  # noqa: F401

    return f"isaaclab {getattr(isaaclab, '__version__', 'unknown')}"


def check_isaac_launch(num_envs: int, device: str = "cuda:0") -> str:
    from isaacgrasp.backends.isaac_backend import IsaacBackend

    backend = IsaacBackend(num_envs=num_envs, image_size=128, device=device)
    _ISAAC["backend"] = backend
    return f"{num_envs} environments launched headless on {device}"


def check_isaac_reset() -> str:
    backend = _ISAAC["backend"]
    observations = backend.reset(list(range(backend.batch_size)), split="all")
    _ISAAC["observations"] = observations
    categories = {view.state.category for view in backend.views() if view.state}
    return f"{len(observations)} scenes, categories {sorted(categories)}"


def check_isaac_camera() -> str:
    """The depth buffer must show an object, not a flat table.

    A flat height map is the signature of headless rendering being off, which
    fails silently: every array has the right shape and every value is wrong.
    """
    observations = _ISAAC["observations"]
    tallest = max(float(o.height.max()) for o in observations)
    if tallest <= 0.005:
        raise RuntimeError(
            "every height map is flat, so the camera rendered no object. Check that "
            "AppLauncher was given enable_cameras=True, and that the camera's "
            "clipping range spans the 0.55 m table distance.")
    return f"tallest object {1000 * tallest:.0f} mm above the table"


def check_isaac_geometry() -> str:
    """The highest-risk stage: does a runtime rescale change the collision shape?

    Two consecutive resets draw different categories, so the observed object
    heights must differ. If PhysX kept the collision geometry from scene build
    time the objects would settle at the same height regardless of what the
    visuals say, and every label in the dataset would be wrong in a way that
    still looks like a plausible dataset.
    """
    backend = _ISAAC["backend"]
    heights = []
    for episode_base in (0, 10_000):
        observations = backend.reset(
            [episode_base + i for i in range(backend.batch_size)], split="all")
        heights.append([float(o.height.max()) for o in observations])
    changed = sum(1 for a, b in zip(heights[0], heights[1], strict=True) if abs(a - b) > 1e-4)
    if changed == 0:
        raise RuntimeError(
            "object heights did not change between two resets with different objects. "
            "The runtime geometry rewrite is not reaching PhysX. Use the fallback in "
            "docs/design.md: fix the category per environment and rebuild the scene "
            "periodically instead of rewriting per episode.")
    return f"{changed}/{len(heights[0])} environments changed shape between resets"


def check_isaac_execute() -> str:
    from simgrasp.policies import HeuristicPolicy
    from simgrasp.seeding import rng_for_episode

    backend = _ISAAC["backend"]
    observations = backend.reset(list(range(backend.batch_size)), split="all")
    policy = HeuristicPolicy()
    grasps = [policy(obs, view, rng_for_episode(1, i))
              for i, (obs, view) in enumerate(zip(observations, backend.views(), strict=True))]
    started = time.perf_counter()
    results = backend.execute(grasps)
    elapsed = time.perf_counter() - started
    successes = sum(r.success for r in results)
    reasons = sorted({r.reason for r in results})
    if all(r.reason == "ik_failed" for r in results):
        raise RuntimeError(
            "every environment reported ik_failed, so the IK controller never "
            "converged. Check the joint and body names in IsaacBackend and the "
            "Jacobian index for the end effector.")
    return (f"{successes}/{len(results)} heuristic grasps succeeded in {elapsed:.1f} s, "
            f"outcomes {reasons}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isaac", action="store_true",
                        help="also exercise the Isaac Lab backend (needs an RTX GPU)")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="environments for the Isaac stages (keep small here)")
    parser.add_argument("--tmp", default="/tmp/isaacgrasp_check")
    parser.add_argument("--device", default="cuda:0",
                        help="device for the Isaac stages. Not every machine puts "
                             "its GPU at index 0, and a rented one often does not")
    args = parser.parse_args()

    import os

    print("isaacgrasp setup check")
    print(f"  python {sys.version.split()[0]}  MUJOCO_GL={os.environ.get('MUJOCO_GL', '(default)')}")
    print()

    tmp_dir = Path(args.tmp)
    stages = Stages()
    stages.run("package imports", check_imports)
    stages.run("parity with simgrasp", check_parity)
    stages.run("torch device", check_torch_device)
    stages.run("mujoco backend", check_mujoco_backend)
    stages.run("collector", lambda: check_collector(tmp_dir))

    if args.isaac:
        print()
        print("  Isaac Lab stages. None of these has been executed before; a failure")
        print("  here is expected to be informative rather than surprising.")
        stages.run("isaac:import", check_isaac_import)
        stages.run("isaac:launch",
                   lambda: check_isaac_launch(args.num_envs, args.device))
        stages.run("isaac:reset", check_isaac_reset)
        stages.run("isaac:camera", check_isaac_camera)
        stages.run("isaac:object geometry", check_isaac_geometry)
        stages.run("isaac:execute", check_isaac_execute)
        backend = _ISAAC.get("backend")
        if backend is not None:
            backend.close()
    else:
        print()
        print("  Isaac Lab stages skipped. Pass --isaac on a machine with an RTX GPU.")

    print()
    if stages.failed:
        name, _status, detail = next(r for r in stages.results if r[1] == FAIL)
        print(f"FAILED at stage '{name}'.")
        print(f"  {detail}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
