#!/usr/bin/env python3
"""Re-score finished checkpoints at a different evaluation episode count.

    python scripts/reevaluate.py --out results/scaling/mujoco --episodes 1500

Training is the expensive half of a scaling curve and evaluation is the noisy
half. At 200 episodes per split the 95% interval on each point is about 7 points,
which is wider than the effect the curve is trying to resolve: the first pass of
this study fitted a held-out slope whose own 95% interval ran from -4.0 to +5.6
points per doubling, which bounds nothing.

This re-runs evaluation against the checkpoints that already exist, so tightening
the intervals costs an evaluation pass rather than a retrain. Points and controls
are updated in place, with the previous result kept alongside under
``eval_previous`` so a rerun can be compared rather than silently replacing what
was published.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaacgrasp import bootstrap  # noqa: E402,F401


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/scaling/mujoco")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controls", nargs="*", default=["heuristic"])
    parser.add_argument("--skip-controls", action="store_true")
    args = parser.parse_args()

    from simgrasp.evaluation import EVAL_EPISODE_OFFSET, evaluate_policy, save_summary

    from isaacgrasp.scaling import single_threaded_torch

    root = Path(args.out)

    def score(policy: str, split: str, **kwargs) -> dict:
        with single_threaded_torch():
            return evaluate_policy(policy, n_episodes=args.episodes, split=split,
                                   workers=args.workers, base_seed=args.seed,
                                   episode_offset=EVAL_EPISODE_OFFSET,
                                   progress=False, **kwargs)

    if not args.skip_controls:
        control_file = root / "controls" / "controls.json"
        controls = json.loads(control_file.read_text()) if control_file.exists() else {}
        for name in args.controls:
            started = time.perf_counter()
            previous = controls.get(name, {})
            updated = {"eval_previous": previous} if previous else {}
            for split in ("seen", "unseen"):
                summary = score(name, split)
                save_summary(summary, root / "controls" / f"{name}_{split}.json",
                             keep_records=False)
                updated[split] = {"rate": summary["success_rate"], "ci95": summary["ci95"],
                                  "n": summary["n"], "by_category": summary["by_category"]}
            controls[name] = updated
            print(f"control {name}: seen {updated['seen']['rate']:.1%}, "
                  f"held-out {updated['unseen']['rate']:.1%} "
                  f"(n={args.episodes} per split, {time.perf_counter() - started:.0f}s)",
                  flush=True)
        control_file.parent.mkdir(parents=True, exist_ok=True)
        control_file.write_text(json.dumps(controls, indent=2, default=float))

    for point_file in sorted(root.glob("n*/point.json")):
        checkpoint = point_file.parent / "best.pt"
        if not checkpoint.exists():
            print(f"skip {point_file.parent.name}: no checkpoint", flush=True)
            continue
        started = time.perf_counter()
        point = json.loads(point_file.read_text())
        point.setdefault("eval_previous",
                         {"seen": point["seen"], "unseen": point["unseen"]})
        for split in ("seen", "unseen"):
            summary = score("cnn", split, policy_kwargs={"checkpoint": str(checkpoint)})
            save_summary(summary, point_file.parent / f"eval_{split}.json",
                         keep_records=False)
            point["seen" if split == "seen" else "unseen"] = {
                "rate": summary["success_rate"], "ci95": summary["ci95"],
                "n": summary["n"], "by_category": summary["by_category"],
                "failure_reasons": summary["failure_reasons"]}
        point["generalisation_gap_pp"] = 100.0 * (point["seen"]["rate"]
                                                  - point["unseen"]["rate"])
        point["eval_episodes"] = args.episodes
        point_file.write_text(json.dumps(point, indent=2, default=float))

        half = 100.0 * (point["unseen"]["ci95"][1] - point["unseen"]["ci95"][0]) / 2.0
        print(f"{point['train_samples']:>6} samples: seen {point['seen']['rate']:.1%}, "
              f"held-out {point['unseen']['rate']:.1%} (+/- {half:.1f} pp), "
              f"gap {point['generalisation_gap_pp']:.1f} pp "
              f"[{time.perf_counter() - started:.0f}s]", flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
