#!/usr/bin/env python3
"""Fetch the Franka Panda MJCF from MuJoCo Menagerie into ``assets/``.

    python scripts/fetch_assets.py

The robot meshes are about 34 MB and are not committed to either repository.
They are pulled from a commit pinned to the one ``simgrasp`` pins, because a
different commit is different gripper geometry and the numbers would stop being
comparable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaacgrasp.assets import MENAGERIE_COMMIT, fetch_assets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="refetch even if present")
    args = parser.parse_args()

    print(f"fetching franka_emika_panda at {MENAGERIE_COMMIT[:8]} ...")
    panda_dir = fetch_assets(force=args.force)
    meshes = len(list((panda_dir / "assets").glob("*")))
    print(f"OK -> {panda_dir} ({meshes} mesh files)")
    print("Apache-2.0, Franka Emika / Google DeepMind; see the LICENSE beside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
