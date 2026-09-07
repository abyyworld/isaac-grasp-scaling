#!/usr/bin/env python3
"""Measure how much re-running training moves a point, from repeated seeds.

    python scripts/seed_variance.py --out results/scaling/mujoco

The curve's scatter decomposition infers training variance by subtracting the
binomial evaluation term from the residual about a fitted line. That inference
assumes the true relationship is log-linear, so any curvature in it gets charged
to training noise and the estimate comes out too high.

Repeated runs assume nothing. Hold the dataset size and the evaluation scenes
fixed, change only the training seed, and whatever the number does is what
re-running does. This reads those replicates back and compares the direct
measurement against what the curve inferred, which is the point: an inferred
quantity that has never been checked against a direct one is a guess with error
bars on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/scaling/mujoco")
    parser.add_argument("--curve", default=None,
                        help="scaling.json to compare against. Defaults to one in --out")
    parser.add_argument("--result", default="results/seed_variance.json")
    args = parser.parse_args()

    from isaacgrasp.scaling import load_points, measure_seed_variance

    root = Path(args.out)
    points = load_points(root)
    if not points:
        raise SystemExit(f"no points found under {root}")

    measured = measure_seed_variance(points)
    if not measured["per_size"]:
        raise SystemExit(
            "no size has more than one training seed, so there is nothing to measure.\n"
            "Run scripts/run_scaling.py again with --train-seed set to something other "
            "than --seed.")

    curve_path = Path(args.curve) if args.curve else root / "scaling.json"
    inferred = None
    if curve_path.exists():
        reading = json.loads(curve_path.read_text()).get("reading", {})
        inferred = reading.get("scatter", {}).get("training_sd_pp")
    measured["inferred_from_curve_sd_pp"] = inferred

    print(f"  {'grasps':>8}{'seeds':>7}{'rates (%)':>26}{'range':>8}"
          f"{'observed':>10}{'training':>10}")
    print("  " + "-" * 69)
    for row in measured["per_size"]:
        rates = ", ".join(f"{r:.1f}" for r in row["rates_pp"])
        print(f"  {row['train_samples']:>8}{row['n_seeds']:>7}{rates:>26}"
              f"{row['range_pp']:>7.1f} {row['observed_sd_pp']:>9.2f}"
              f"{row['training_sd_pp']:>10.2f}")
    print("  " + "-" * 69)

    pooled = measured.get("pooled")
    if pooled:
        print(f"  pooled over {pooled['degrees_of_freedom']} degrees of freedom: "
              f"observed {pooled['observed_sd_pp']:.2f} pp, "
              f"evaluation {pooled['evaluation_sd_pp']:.2f} pp, "
              f"training {pooled['training_sd_pp']:.2f} pp")
        if inferred is not None:
            print(f"  the curve's residual inferred {inferred:.2f} pp of training noise")
            difference = pooled["training_sd_pp"] - inferred
            direction = "higher" if difference > 0 else "lower"
            print(f"  direct measurement is {abs(difference):.2f} pp {direction}")

    out = Path(args.result)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(measured, indent=2, default=float))
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
