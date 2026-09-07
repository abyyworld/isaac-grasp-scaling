#!/usr/bin/env python3
"""Compare collection throughput between the two simulators.

    python scripts/throughput.py \
        --dataset data/mj18k  --hardware "4 vCPU, no GPU" --workers 4 \
        --dataset data/isaac  --hardware "RTX 4090, 24 GB" --num-envs 1024

Rates come from the metadata each collection run wrote while it ran, so this
reports what actually happened rather than a benchmark staged afterwards under
friendlier conditions.
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
    parser.add_argument("--dataset", action="append", required=True,
                        help="a collected dataset directory; repeat for each run")
    parser.add_argument("--hardware", action="append", required=True,
                        help="what that run was collected on; repeat, in the same order")
    parser.add_argument("--workers", action="append", type=int, default=None)
    parser.add_argument("--num-envs", action="append", type=int, default=None)
    parser.add_argument("--out", default="results/throughput.json")
    args = parser.parse_args()

    if len(args.dataset) != len(args.hardware):
        parser.error("give one --hardware for each --dataset, in the same order")

    from isaacgrasp.throughput import compare, format_table, record_from_dataset

    workers = args.workers or [1] * len(args.dataset)
    num_envs = args.num_envs or [1] * len(args.dataset)
    records = [
        record_from_dataset(path, hardware=hardware,
                            workers=workers[i] if i < len(workers) else 1,
                            batch_size=num_envs[i] if i < len(num_envs) else 1)
        for i, (path, hardware) in enumerate(zip(args.dataset, args.hardware, strict=True))
    ]
    comparison = compare(records)
    print()
    print(format_table(comparison))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
