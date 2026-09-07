"""The restated oracle grasp must match the MuJoCo method exactly.

``oracle_grasp_for_state`` is the only piece of upstream logic this port
restates rather than imports, and it defines the centre of the label
distribution: the collector's dominant sampling mode is a perturbation of it.
A discrepancy here would shift every training label in the Isaac dataset
relative to the MuJoCo one, and the scaling curve would be measuring that shift
instead of the effect of data volume.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from simgrasp.objects import ALL_CATEGORIES  # noqa: E402

from isaacgrasp.backends.base import SimpleEnvView, oracle_grasp_for_state  # noqa: E402


@pytest.fixture(scope="module")
def env():
    from simgrasp.env import PandaGraspEnv

    e = PandaGraspEnv(image_size=96, base_seed=0)
    yield e
    e.close()


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_matches_the_mujoco_oracle(env, category):
    for episode in range(3):
        env.reset(episode, category=category)
        state = env.state
        for hint_index in range(len(state.spec.grasp_hints)):
            expected = env.oracle_grasp(hint_index)
            got = oracle_grasp_for_state(state, env.table_z, hint_index)
            assert got.as_array() == pytest.approx(expected.as_array(), abs=0.0, rel=0.0), (
                f"{category}: restated oracle disagrees with PandaGraspEnv.oracle_grasp")


def test_view_wraps_the_helper(env):
    env.reset(0)
    view = SimpleEnvView(env.table_z, env.state)
    assert view.oracle_grasp(0).as_array() == pytest.approx(env.oracle_grasp(0).as_array())


def test_view_without_state_is_an_error():
    with pytest.raises(RuntimeError, match="reset"):
        SimpleEnvView(0.4).oracle_grasp()
