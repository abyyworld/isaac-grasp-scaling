"""Orientation error: the metric the whole question turns on.

The original study's diagnosis was that the network learned grasp *position*
well and grasp *angle* not at all. Validation AP does not show this, because AP
is dominated by where to grasp and stays high while the angle head is collapsed.
So the scaling curve reports orientation error alongside success rate, and this
module measures it.

The method is the original's ``scripts/angle_ablation.py``, restated here
because that file lives in a scripts directory that pip does not install. Two
numbers per checkpoint:

``angle_error_deg``
    Mean absolute difference between the predicted grasp angle and the oracle's,
    wrapped to a half turn, over elongated objects only. **Random guessing
    scores 45 degrees**, so a number at or above 45 means no orientation
    information at all.

``bin_spread``
    Range of predicted quality across the twelve angle bins at the pixel the
    model chose. A collapsed angle head gives approximately zero: the same
    success probability whichever way the gripper is turned.

The determinacy filter is load-bearing. A cylinder or a sphere grasps equally
well at every angle, so scoring a prediction against "the" oracle angle measures
nothing, and a nearly square box is the same problem in weaker form. An earlier
version of the original script averaged those in and reported no effect where
there was a large one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from simgrasp.objects import ALL_CATEGORIES, SEEN_CATEGORIES

# Keep only objects whose horizontal extents differ by at least this ratio.
ASPECT_THRESHOLD = 1.3
# Scenes for the angle measurement. Disjoint from both the collection range and
# the success-rate evaluation range, so no scene is scored twice for two
# different purposes.
ANGLE_EPISODE_OFFSET = 2_000_000


def angle_is_determinate(spec) -> bool:
    """True when the object has a clearly narrower horizontal axis to grasp across."""
    geoms = spec.geoms
    if len(geoms) > 1:
        return True  # L, T, mug, dumbbell: multi-part, orientation always matters
    geom = geoms[0]
    if geom.type in ("sphere", "cylinder"):
        return False  # rotationally symmetric about the vertical
    if geom.type == "capsule":
        return True  # lying down: the shaft defines the grasp axis
    half_x, half_y = geom.size[0], geom.size[1]
    lo, hi = min(half_x, half_y), max(half_x, half_y)
    return lo > 0 and hi / lo >= ASPECT_THRESHOLD


def measure_angle_error(checkpoint: str | Path, episodes: int = 10,
                        offset: int = ANGLE_EPISODE_OFFSET, base_seed: int = 0,
                        device: str = "cpu", image_size: int = 224) -> dict[str, Any]:
    """Mean orientation error and bin spread for one checkpoint."""
    from simgrasp.env import PandaGraspEnv
    from simgrasp.policies.learned import LearnedPolicy
    from simgrasp.transforms import wrap_grasp_angle

    policy = LearnedPolicy(checkpoint=str(checkpoint), device=device)
    errors: dict[str, list[float]] = {}
    spreads: dict[str, list[float]] = {}
    skipped = 0

    with PandaGraspEnv(image_size=image_size, base_seed=base_seed) as env:
        for category in ALL_CATEGORIES:
            for k in range(episodes):
                obs = env.reset(offset + k, category=category)
                if not angle_is_determinate(env.state.spec):
                    skipped += 1
                    continue

                quality, _width, _scale = policy.predict_maps(obs)
                mask = policy._workspace_mask(obs, quality.shape[1:])
                masked = np.where(mask[None], quality, -1.0)
                _b, v, u = np.unravel_index(int(np.argmax(masked)), masked.shape)
                column = masked[:, v, u]
                spreads.setdefault(category, []).append(float(column.max() - column.min()))

                predicted = policy(obs, env, np.random.default_rng(0)).yaw
                oracle = env.oracle_grasp().yaw
                delta = abs(float(wrap_grasp_angle(predicted - oracle)))
                errors.setdefault(category, []).append(float(np.degrees(delta)))

    def mean_over(categories) -> tuple[float, float, int]:
        e = [x for c in categories for x in errors.get(c, [])]
        s = [x for c in categories for x in spreads.get(c, [])]
        return (float(np.mean(e)) if e else float("nan"),
                float(np.mean(s)) if s else float("nan"), len(e))

    held_out = [c for c in ALL_CATEGORIES if c not in SEEN_CATEGORIES]
    error_all, spread_all, n_all = mean_over(ALL_CATEGORIES)
    error_seen, _spread_seen, n_seen = mean_over(SEEN_CATEGORIES)
    error_held, _spread_held, n_held = mean_over(held_out)

    return {
        "checkpoint": str(checkpoint),
        "angle_error_deg": error_all,
        "angle_error_deg_seen": error_seen,
        "angle_error_deg_heldout": error_held,
        "bin_spread": spread_all,
        "n_scored": n_all,
        "n_seen": n_seen,
        "n_heldout": n_held,
        "n_skipped_indeterminate": skipped,
        "chance_level_deg": 45.0,
        "per_category_error_deg": {c: float(np.mean(v)) for c, v in sorted(errors.items())},
        "per_category_bin_spread": {c: float(np.mean(v)) for c, v in sorted(spreads.items())},
    }
