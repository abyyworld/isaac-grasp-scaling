#!/usr/bin/env python3
"""Generate a grasp dataset with either simulator.

    # MuJoCo, four worker processes, the control arm of the experiment
    python scripts/collect.py --backend mujoco --scenes 6000 --workers 4 --out data/mj18k

    # Isaac Lab, one process, hundreds of environments in parallel
    python scripts/collect.py --backend isaac --scenes 200000 --num-envs 1024 --out data/isaac600k

``--angles-per-scene`` executes several grasps at the same point in one settled
scene at different orientations. That is the fix the original study measured for
its orientation collapse, and it is what makes the labels contrastive about
angle rather than only about position.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaacgrasp import bootstrap  # noqa: E402,F401


def _worker(kwargs):
    from isaacgrasp.collect import collect

    return collect(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="mujoco", choices=["mujoco", "isaac"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", type=int, default=6000)
    parser.add_argument("--split", default="seen", choices=["all", "seen", "unseen"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--angles-per-scene", type=int, default=3)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes; MuJoCo only, since Isaac parallelises internally")
    parser.add_argument("--num-envs", type=int, default=256,
                        help="parallel environments; Isaac only")
    args = parser.parse_args()

    if args.backend == "isaac" and args.workers > 1:
        parser.error("the isaac backend parallelises inside one process; use --num-envs")

    common = dict(
        backend_name=args.backend, out_dir=args.out, split=args.split,
        base_seed=args.seed, image_size=args.image_size,
        angles_per_scene=args.angles_per_scene, shard_size=args.shard_size,
        store_rgb=not args.no_rgb,
        backend_kwargs={"num_envs": args.num_envs} if args.backend == "isaac" else {},
    )

    started = time.perf_counter()
    if args.workers <= 1:
        meta = _worker({**common, "n_scenes": args.scenes,
                        "episode_offset": args.offset, "worker_id": 0})
    else:
        # Contiguous blocks per worker, so each worker's shards cover a known
        # episode range and a partial run is still a usable dataset.
        per_worker = args.scenes // args.workers
        jobs = []
        for i in range(args.workers):
            count = per_worker + (args.scenes % args.workers if i == args.workers - 1 else 0)
            jobs.append({**common, "n_scenes": count,
                         "episode_offset": args.offset + i * per_worker,
                         "worker_id": i, "verbose": i == 0})
        # "spawn": each MuJoCo worker needs its own GL context, and forking a
        # process that already holds one is unreliable across platforms.
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            parts = pool.map(_worker, jobs)
        meta = _merge(parts, args, time.perf_counter() - started)
        Path(args.out, "dataset_meta.json").write_text(json.dumps(meta, indent=2, default=float))

    stats = meta["stats"]
    print()
    print(f"collected {stats['n']} samples from {stats['n_scenes']} scenes in "
          f"{stats['elapsed_seconds']:.1f}s")
    print(f"  {stats['samples_per_hour']:,.0f} samples/hour, "
          f"positive rate {stats['positive_rate']:.1%}")
    print(f"  wrote {args.out}")
    print()
    print(f"  {'category':<12}{'n':>8}{'positive':>11}")
    for category, row in stats["by_category"].items():
        print(f"  {category:<12}{row['n']:>8}{row['positive_rate']:>10.1%}")
    return 0


def _merge(parts, args, elapsed: float) -> dict:
    """Combine per-worker metadata into one dataset record.

    Wall-clock is the elapsed time of the whole run, not the sum over workers:
    the samples-per-hour figure has to describe the machine, which is what
    decides whether a dataset is affordable.
    """
    merged = dict(parts[0])
    stats = {"n": 0, "n_scenes": 0, "successes": 0, "unstable": 0, "ik_failed": 0,
             "by_category": {}, "by_mode": {}}
    for part in parts:
        source = part["stats"]
        for key in ("n", "n_scenes", "successes", "unstable", "ik_failed"):
            stats[key] += source.get(key, 0)
        for table in ("by_category", "by_mode"):
            for key, row in source[table].items():
                current = stats[table].setdefault(key, {"n": 0, "positives": 0})
                current["n"] += row["n"]
                current["positives"] += row["positives"]
    for table in ("by_category", "by_mode"):
        for row in stats[table].values():
            row["positive_rate"] = row["positives"] / row["n"] if row["n"] else 0.0
    stats["positive_rate"] = stats["successes"] / stats["n"] if stats["n"] else 0.0
    stats["elapsed_seconds"] = elapsed
    stats["samples_per_second"] = stats["n"] / elapsed if elapsed else 0.0
    stats["samples_per_hour"] = 3600.0 * stats["n"] / elapsed if elapsed else 0.0
    stats["scenes_per_second"] = stats["n_scenes"] / elapsed if elapsed else 0.0

    merged["stats"] = stats
    merged["n_scenes"] = args.scenes
    merged["n_episodes"] = args.scenes
    merged["workers"] = args.workers
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
