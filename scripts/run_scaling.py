#!/usr/bin/env python3
"""Train the same network at several dataset sizes and evaluate every point.

    python scripts/run_scaling.py --data data/mj18k --out results/scaling/mujoco \
        --sizes 128 256 512 1024 2048 --epochs 10 --input-size 96

Every point uses the same architecture, the same training loop and the same
held-out evaluation scenes. The only thing that changes along the curve is how
many training scenes the loader is allowed to use, which is the whole point: a
difference between points is a difference in data volume and nothing else.

The run is resumable. Each point writes ``point.json`` when it finishes and is
skipped on a re-run, because a curve is hours of compute and a disconnect is a
bad reason to lose it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaacgrasp import bootstrap  # noqa: E402,F401


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results/scaling")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[128, 256, 512, 1024, 2048],
                        help="training scenes per point")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--angle-episodes", type=int, default=10)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    from isaacgrasp.plot import plot_scaling
    from isaacgrasp.scaling import ScalingConfig, run_scaling

    cfg = ScalingConfig(
        data=args.data, out=args.out, sizes=tuple(sorted(args.sizes)),
        epochs=args.epochs, batch_size=args.batch_size, input_size=args.input_size,
        pretrained=args.pretrained, workers=args.workers, device=args.device,
        seed=args.seed, eval_episodes=args.eval_episodes,
        eval_workers=args.eval_workers, angle_episodes=args.angle_episodes,
        label=args.label,
    )
    result = run_scaling(cfg)

    figure = plot_scaling(result, Path(args.out) / "scaling_curve.png")
    print()
    print(f"{'train scenes':>13}{'samples':>10}{'seen':>9}{'held-out':>10}"
          f"{'gap':>8}{'angle err':>11}")
    print("-" * 61)
    for point in result["points"]:
        print(f"{point['train_scenes']:>13}{point['train_samples']:>10}"
              f"{point['seen']['rate']:>8.1%}{point['unseen']['rate']:>10.1%}"
              f"{point['generalisation_gap_pp']:>7.1f}pp"
              f"{point['angle']['angle_error_deg']:>8.1f} deg")
    print("-" * 61)
    for name, control in result["controls"].items():
        print(f"{name + ' control':>13}{'':>10}{control['seen']['rate']:>8.1%}"
              f"{control['unseen']['rate']:>10.1%}")
    print()
    print(f"wrote {args.out}/scaling.json, scaling.csv and {figure.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
