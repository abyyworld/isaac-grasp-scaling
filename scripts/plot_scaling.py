#!/usr/bin/env python3
"""Redraw the scaling curve from a finished run.

    python scripts/plot_scaling.py --result results/scaling/mujoco/scaling.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    import json

    from isaacgrasp.plot import plot_scaling

    result = json.loads(Path(args.result).read_text())
    out = Path(args.out) if args.out else Path(args.result).with_name("scaling_curve.png")
    print(f"wrote {plot_scaling(result, out, title=args.title)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
