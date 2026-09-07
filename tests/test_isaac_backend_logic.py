"""Execute the Isaac backend's own logic, against a stub Isaac Lab.

The backend has never run on a GPU, and until this file existed it had never run
at all: not one line of ``IsaacBackend`` had been executed by anything. That is
the state these tests change.

**What this does not do.** It does not verify that real Isaac Lab behaves as the
stubs do. The stub API is written from the same understanding that wrote the
backend, so a shared misunderstanding survives both.
``scripts/check_setup.py --isaac`` on a real machine is the only thing that
settles that, and the module's status banner still says so.

**What it does do.** It runs every code path, and it checks the arithmetic that
would otherwise be invisible until success rates came out wrong: the TCP offset,
the base-frame conversion, the per-environment ordering of a batch, the object
scaling, the mass, and the outcome rule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from fakes.isaac import install_fake_isaac, uninstall_fake_isaac

NUM_ENVS = 4
IMAGE_SIZE = 32


@pytest.fixture
def backend():
    """A backend wired to the stub API, torn down afterwards."""
    install_fake_isaac(NUM_ENVS)
    from isaacgrasp.backends.isaac_backend import IsaacBackend

    instance = IsaacBackend(num_envs=NUM_ENVS, image_size=IMAGE_SIZE, base_seed=0,
                            device="cpu")
    try:
        yield instance
    finally:
        uninstall_fake_isaac()


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_the_backend_builds_a_scene(backend):
    assert backend.batch_size == NUM_ENVS
    assert backend.image_size == IMAGE_SIZE
    assert backend.table_z == pytest.approx(0.40)
    assert backend._scene is not None and backend._sim is not None


def test_camera_intrinsics_match_the_mujoco_formula(backend):
    """Both cameras must see the same solid angle or no pixel means the same thing."""
    from simgrasp.scene import CAMERA_FOVY_DEG

    intrinsics = backend._intrinsics
    expected_focal = (IMAGE_SIZE / 2.0) / math.tan(math.radians(CAMERA_FOVY_DEG) / 2.0)
    assert intrinsics.fx == pytest.approx(expected_focal)
    assert intrinsics.fx == intrinsics.fy, "square pixels, as MuJoCo's fovy implies"
    assert intrinsics.cx == pytest.approx((IMAGE_SIZE - 1) / 2.0)


def test_the_scene_puts_the_arm_on_a_plinth_at_table_height():
    """Mounting it on the ground would put every grasp 0.40 m out in its own frame."""
    install_fake_isaac(NUM_ENVS)
    try:
        from simgrasp.scene import PLINTH_HALF, TABLE_HEIGHT

        from isaacgrasp.backends.isaac_scene import build_scene_cfg

        cfg = build_scene_cfg(num_envs=NUM_ENVS, env_spacing=2.5, image_size=IMAGE_SIZE)
        assert cfg.robot.init_state.pos == (0.0, 0.0, TABLE_HEIGHT)
        assert cfg.plinth.init_state.pos == (0.0, 0.0, PLINTH_HALF[2])
        assert cfg.robot.spawn.rigid_props.disable_gravity is True
    finally:
        uninstall_fake_isaac()


def test_disabling_gravity_does_not_leak_into_the_shared_franka_config():
    """.replace() does not deep-copy spawn, so the class body would mutate upstream."""
    install_fake_isaac(NUM_ENVS)
    try:
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        from isaacgrasp.backends.isaac_scene import build_scene_cfg

        build_scene_cfg(num_envs=NUM_ENVS, env_spacing=2.5, image_size=IMAGE_SIZE)
        assert FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity is False
    finally:
        uninstall_fake_isaac()


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #


def test_reset_draws_the_same_objects_as_the_mujoco_backend(backend):
    """Episode i must mean the same object in both simulators."""
    from simgrasp.objects import sample_category, sample_object
    from simgrasp.seeding import rng_for_episode

    episodes = [0, 1, 2, 3]
    backend.reset(episodes, split="all")
    for index, episode in enumerate(episodes):
        rng = rng_for_episode(0, episode)
        expected_category = sample_category(rng, "all")
        expected = sample_object(rng, expected_category)
        state = backend.views()[index].state
        assert state.category == expected_category
        assert state.spec.top_z == pytest.approx(expected.top_z)


def test_reset_returns_one_observation_per_episode(backend):
    observations = backend.reset([0, 1], split="all")
    assert len(observations) == 2
    for observation in observations:
        assert observation.height.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert observation.rgb.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert observation.table_z == pytest.approx(0.40)


def test_unused_environments_have_no_state(backend):
    """A partial batch must not leave a previous episode's state readable."""
    backend.reset([0, 1, 2, 3], split="all")
    backend.reset([7], split="all")
    assert backend.views()[0].state is not None
    for view in backend.views()[1:]:
        assert view.state is None


def test_object_prims_are_scaled_to_the_sampled_dimensions(backend):
    """The runtime rewrite must reach the prim, with the catalogue's extents."""
    from isaacgrasp.backends.isaac_objects import prims_for_spec

    backend.reset([0, 1, 2, 3], split="all")
    stage = backend._stage()
    for env_index in range(NUM_ENVS):
        spec = backend.views()[env_index].state.spec
        for slot, placement in enumerate(prims_for_spec(spec)):
            path = backend._prim_path(env_index, slot, placement.prim_type)
            ops = stage.GetPrimAtPath(path).ops
            assert tuple(ops["scale"]) == pytest.approx(placement.scale)
            assert tuple(ops["translate"]) == pytest.approx(placement.translate)


def test_unused_prims_are_shrunk_and_parked_below_the_table(backend):
    """A slot holds every prim type; the inactive ones must not touch anything."""
    from isaacgrasp.backends.isaac_objects import (
        INACTIVE_TRANSLATE,
        PRIM_TYPES,
        prims_for_spec,
    )

    backend.reset([0], split="all")
    stage = backend._stage()
    spec = backend.views()[0].state.spec
    placements = prims_for_spec(spec)
    for slot in range(3):
        active = placements[slot].prim_type if slot < len(placements) else None
        for prim_type in PRIM_TYPES:
            if prim_type == active:
                continue
            ops = stage.GetPrimAtPath(backend._prim_path(0, slot, prim_type)).ops
            assert max(ops["scale"]) < 1e-4
            assert tuple(ops["translate"]) == pytest.approx(INACTIVE_TRANSLATE)


def test_object_mass_comes_from_the_catalogue(backend):
    from isaacgrasp.backends.isaac_objects import spec_mass

    backend.reset([0, 1, 2, 3], split="all")
    masses = backend._objects.root_physx_view.get_masses()
    for index in range(NUM_ENVS):
        expected = spec_mass(backend.views()[index].state.spec)
        assert float(masses[index, 0]) == pytest.approx(expected, rel=1e-5)


def test_object_friction_comes_from_the_catalogue(backend):
    backend.reset([0, 1, 2, 3], split="all")
    materials = backend._objects.root_physx_view.get_material_properties()
    for index in range(NUM_ENVS):
        expected = backend.views()[index].state.spec.friction[0]
        assert float(materials[index, 0, 0]) == pytest.approx(expected)
        assert float(materials[index, 0, 1]) == pytest.approx(expected)


def test_objects_are_placed_inside_the_workspace(backend):
    from simgrasp.scene import WORKSPACE_X, WORKSPACE_Y

    backend.reset([0, 1, 2, 3], split="all")
    origins = backend._scene.env_origins.numpy()
    root = backend._objects.data.root_state_w.numpy()
    for index in range(NUM_ENVS):
        x, y, z = root[index, :3] - origins[index]
        assert WORKSPACE_X[0] <= x <= WORKSPACE_X[1]
        assert WORKSPACE_Y[0] <= y <= WORKSPACE_Y[1]
        assert z == pytest.approx(0.40), "objects spawn on the table top"


def test_the_arm_is_retracted_to_the_capture_pose(backend):
    """Addressed by joint name: the arm must be out of the camera's view."""
    from simgrasp.scene import CAPTURE_QPOS

    backend.reset([0], split="all")
    joint_pos = backend._robot.data.joint_pos
    arm = backend._arm_joint_indices()
    assert joint_pos[0, arm].numpy() == pytest.approx(CAPTURE_QPOS, abs=1e-5)
    fingers = backend._finger_joint_indices()
    assert joint_pos[0, fingers].numpy() == pytest.approx([0.04, 0.04])


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #


def test_a_nearer_depth_reading_becomes_a_taller_height(backend):
    """The height map is what every label is derived from."""
    camera = backend._scene["camera"]
    # The camera sits 0.55 m above the table; a reading of 0.52 is 30 mm of object.
    camera.data.output["distance_to_image_plane"][0, 16, 16, 0] = 0.52
    observation = backend.reset([0], split="all")[0]
    assert observation.height[16, 16] == pytest.approx(0.03, abs=2e-3)
    assert observation.height[0, 0] == pytest.approx(0.0, abs=2e-3)


def test_rendering_happens_once_per_observation(backend):
    """A double sensor update would desynchronise the camera from the physics."""
    before = backend._scene["camera"].updates
    backend.observe()
    assert backend._scene["camera"].updates == before + 1


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _grasp(x=0.55, y=0.0, z=0.44, yaw=0.0, width=0.03):
    from simgrasp.grasp import Grasp

    return Grasp(x=x, y=y, z=z, yaw=yaw, width=width)


def test_the_ik_command_targets_the_hand_body_not_the_tcp(backend):
    """A grasp is specified at the pad centre, 103.4 mm along the hand's +z.

    For a top-down grasp the hand's +z points down, so the hand belongs that far
    ABOVE the grasp point. Commanding the hand to the grasp position would drive
    the gripper through the table, and nothing would raise.
    """
    from simgrasp.scene import TCP_OFFSET_Z

    backend.reset([0], split="all")
    grasp = _grasp(x=0.55, y=0.10, z=0.44)
    command = backend._grasp_targets([grasp]).numpy()
    # The robot base is on the plinth at z = 0.40, and the command is in its frame.
    assert command[0, 0] == pytest.approx(0.55)
    assert command[0, 1] == pytest.approx(0.10)
    assert command[0, 2] == pytest.approx(0.44 + TCP_OFFSET_Z - 0.40, abs=1e-6)


def test_the_command_is_expressed_in_the_robot_base_frame(backend):
    """Read from the robot's actual root state, not assumed."""
    backend.reset([0], split="all")
    backend._robot.data.root_state_w[:, 2] = 0.75  # move the mount
    command = backend._grasp_targets([_grasp(z=0.44)]).numpy()
    from simgrasp.scene import TCP_OFFSET_Z

    assert command[0, 2] == pytest.approx(0.44 + TCP_OFFSET_Z - 0.75, abs=1e-6)


def test_the_command_orientation_is_the_top_down_grasp_matrix(backend):
    from simgrasp.transforms import mat_to_quat, topdown_grasp_mat

    backend.reset([0], split="all")
    for yaw in (0.0, 0.4, -1.1):
        command = backend._grasp_targets([_grasp(yaw=yaw)]).numpy()
        expected = mat_to_quat(topdown_grasp_mat(yaw))
        assert command[0, 3:7] == pytest.approx(expected, abs=1e-6)


def test_each_environment_gets_its_own_grasp(backend):
    """Ordering: environment i's command must come from environment i's grasp."""
    backend.reset([0, 1, 2, 3], split="all")
    grasps = [_grasp(x=0.50 + 0.02 * i, y=-0.05 + 0.03 * i) for i in range(NUM_ENVS)]
    command = backend._grasp_targets(grasps).numpy()
    for i, grasp in enumerate(grasps):
        assert command[i, 0] == pytest.approx(grasp.x)
        assert command[i, 1] == pytest.approx(grasp.y)


def test_execute_returns_one_result_per_grasp_in_order(backend):
    backend.reset([0, 1, 2, 3], split="all")
    grasps = [_grasp(x=0.50 + 0.02 * i) for i in range(NUM_ENVS)]
    results = backend.execute(grasps)
    assert len(results) == NUM_ENVS
    for result, grasp in zip(results, grasps, strict=True):
        assert result.grasp is grasp
        assert result.reason in {"success", "closed_empty", "dropped", "no_lift",
                                 "ik_failed", "unstable"}


def test_a_lifted_object_is_scored_a_success(backend):
    """The success rule must be the shared one, on the object's real rise."""
    from simgrasp.env import LIFT_SUCCESS_THRESHOLD

    backend.reset([0], split="all")
    grasps = [_grasp()] * NUM_ENVS
    # Raise the object well past the threshold before the outcome is read.
    original = backend._object_poses

    def lifted():
        positions, yaws = original()
        positions[0, 2] += LIFT_SUCCESS_THRESHOLD + 0.05
        return positions, yaws

    backend._object_poses = lifted
    result = backend.execute(grasps)[0]
    assert result.success is True
    assert result.reason == "success"
    assert result.lift_height > LIFT_SUCCESS_THRESHOLD


def test_an_unlifted_object_with_closed_fingers_is_closed_empty(backend):
    backend.reset([0], split="all")
    result = backend.execute([_grasp()] * NUM_ENVS)[0]
    assert result.success is False
    # The stub gripper closes to zero and the object does not move.
    assert result.reason == "closed_empty"


def test_execute_rejects_more_grasps_than_environments(backend):
    backend.reset([0], split="all")
    with pytest.raises(ValueError, match="environments"):
        backend.execute([_grasp()] * (NUM_ENVS + 1))


def test_reset_rejects_more_episodes_than_environments(backend):
    with pytest.raises(ValueError, match="environments"):
        backend.reset(list(range(NUM_ENVS + 1)), split="all")


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


def test_snapshot_and_restore_round_trip(backend):
    """Several grasps share one settled scene; restore must actually restore it."""
    backend.reset([0, 1, 2, 3], split="all")
    snapshot = backend.snapshot()
    before = backend._objects.data.root_state_w.clone()

    backend._objects.data.root_state_w += 1.0
    backend._robot.data.joint_pos += 0.5
    backend.restore(snapshot)

    assert torch.allclose(backend._objects.data.root_state_w, before)
    assert torch.allclose(backend._robot.data.joint_pos, snapshot["joint_pos"])


def test_the_backend_satisfies_the_shared_interface(backend):
    from isaacgrasp.backends.base import GraspBackend

    assert isinstance(backend, GraspBackend)
    _ = np
