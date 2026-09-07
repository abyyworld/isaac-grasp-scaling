"""The shared success criterion must reproduce MuJoCo's verdict exactly.

Success is defined as "the object rose more than 0.08 m and was still up when
the hold ended". If the two backends disagree about what counts, every number in
this project is incomparable with the original study, and the disagreement would
show up as a plausible-looking difference in success rate rather than as an
error.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from simgrasp.env import LIFT_SUCCESS_THRESHOLD  # noqa: E402
from simgrasp.policies import GraspSampler  # noqa: E402
from simgrasp.seeding import rng_for_episode  # noqa: E402

from isaacgrasp.backends import build_backend  # noqa: E402
from isaacgrasp.backends.base import classify_outcome  # noqa: E402


def test_classifier_agrees_with_the_mujoco_environment():
    """Run real episodes and re-derive each verdict from the two measurements."""
    sampler = GraspSampler()
    backend = build_backend("mujoco", image_size=96, base_seed=0)
    seen = set()
    try:
        for episode in range(24):
            obs = backend.reset([episode], split="all")[0]
            view = backend.views()[0]
            grasp = sampler(obs, view, rng_for_episode(7_777_777, episode))
            result = backend.execute([grasp])[0]
            if result.reason in ("ik_failed", "unstable"):
                continue  # decided before the attempt finished, so not this rule
            success, reason = classify_outcome(
                result.lift_height, result.final_gripper_width, LIFT_SUCCESS_THRESHOLD)
            assert (success, reason) == (result.success, result.reason), (
                f"episode {episode}: classifier said {(success, reason)}, "
                f"environment said {(result.success, result.reason)}")
            seen.add(reason)
    finally:
        backend.close()
    assert len(seen) >= 2, f"only saw outcomes {seen}; the test is not exercising the rule"


@pytest.mark.parametrize("lift,width,expected", [
    (0.20, 0.03, (True, "success")),
    (LIFT_SUCCESS_THRESHOLD + 1e-6, 0.0, (True, "success")),
    (0.0, 0.0, (False, "closed_empty")),
    (0.04, 0.02, (False, "dropped")),
    (0.001, 0.02, (False, "no_lift")),
])
def test_rule_boundaries(lift, width, expected):
    assert classify_outcome(lift, width, LIFT_SUCCESS_THRESHOLD) == expected


def test_success_takes_priority_over_an_empty_gripper():
    """A lifted object with fully closed fingers is still a success.

    This ordering is not cosmetic: a thin object held between closed pads reads
    as zero width, and grading it as closed_empty would under-count exactly the
    flat non-convex shapes that are the hardest category in the study.
    """
    assert classify_outcome(0.15, 0.0, LIFT_SUCCESS_THRESHOLD) == (True, "success")
    assert np.isfinite(LIFT_SUCCESS_THRESHOLD)
