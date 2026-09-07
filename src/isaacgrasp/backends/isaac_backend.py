"""Isaac Lab backend: the same task, stepped in parallel on a GPU.

STATUS: NOT YET EXECUTED
------------------------
This module has never been run. It was written on a machine with no NVIDIA GPU,
where Isaac Sim cannot be installed, so every claim it makes about the Isaac Lab
API is unverified. The parts of the port that *can* be checked without a GPU are
checked and pass: the object catalogue translation
(``tests/test_isaac_objects.py``), the oracle grasp
(``tests/test_oracle_view.py``), the success criterion
(``tests/test_outcome_classification.py``) and the collector
(``tests/test_collect.py``). What remains unverified is this file's contact with
the simulator.

Run ``scripts/check_setup.py`` before anything else on a GPU box. It exercises
this module in stages and reports which stage fails, because the useful question
is not "does it work" but "which of the six things it needs is missing".

Design
------
The expensive part of an episode in the MuJoCo original is not the physics, it
is the four-second scripted rollout and the render, paid once per grasp. Isaac
Lab pays them once per grasp *per batch*, so the whole batch is the unit of work
and the interface in :mod:`isaacgrasp.backends.base` is batched throughout.

The scene is built once and never rebuilt. Each environment holds one object
body with :data:`~isaacgrasp.backends.isaac_objects.MAX_PRIMS` slots, each slot
carrying one primitive of every type the catalogue uses. An episode activates
one primitive per slot and scales it to the sampled dimensions. This mirrors
what the MuJoCo original does (compile once, rewrite the scene in place) and it
exists for the same reason: recompiling a scene per episode costs more than the
episode.

The known risk in that design, and the first thing ``check_setup.py`` tests, is
whether PhysX updates an analytic shape's collision geometry when its
``xformOp:scale`` changes at runtime. If it does not, the fallback is documented
in ``docs/design.md``: hold the category fixed per environment and rebuild the
scene every few thousand episodes instead of every episode. That trades a small
amount of throughput for certainty, and it is a change to this file alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from simgrasp.camera import CameraIntrinsics, height_map
from simgrasp.env import (
    APPROACH_HEIGHT,
    LIFT_DISTANCE,
    LIFT_SUCCESS_THRESHOLD,
    SETTLE_TIME,
    EpisodeState,
    GraspResult,
    Observation,
)
from simgrasp.grasp import Grasp
from simgrasp.objects import sample_category, sample_object
from simgrasp.scene import (
    CAMERA_FOVY_DEG,
    TABLE_HEIGHT,
    WORKSPACE_X,
    WORKSPACE_Y,
)
from simgrasp.seeding import rng_for_episode
from simgrasp.transforms import quat_to_mat

from .base import SimpleEnvView, classify_outcome
from .isaac_objects import (
    INACTIVE_TRANSLATE,
    MAX_PRIMS,
    PRIM_TYPES,
    inactive_scale,
    prims_for_spec,
)

# Physics step. The MuJoCo original runs at its model default; what has to match
# is the settle duration and the motion durations, not the integrator step.
PHYSICS_DT = 1.0 / 120.0
# Environments are spaced far enough apart that no arm can reach a neighbour's
# table. The table is 0.70 m by 0.90 m.
ENV_SPACING = 2.5

# Durations of the scripted phases, copied from the MuJoCo controller sequence so
# the executed trajectory matches. See ``simgrasp.env.PandaGraspEnv.execute``.
PRESHAPE_TIME = 0.3
APPROACH_TIME = 1.3
CONVERGE_TIME = 0.5
DESCEND_TIME = 0.9
CLOSE_TIME = 0.8
LIFT_TIME = 1.3
HOLD_TIME = 0.4


@dataclass
class IsaacConfig:
    num_envs: int = 256
    image_size: int = 224
    base_seed: int = 0
    device: str = "cuda:0"
    headless: bool = True
    # Isaac Sim will not produce camera output unless rendering is switched on
    # explicitly in headless mode. Without this the depth buffer is all zeros and
    # every height map is flat, which looks like a scene bug rather than a
    # configuration one.
    enable_cameras: bool = True


class IsaacBackend:
    """Batched Franka grasping in Isaac Lab, scored by the original criterion."""

    name = "isaac"

    def __init__(self, num_envs: int = 256, image_size: int = 224, base_seed: int = 0,
                 device: str = "cuda:0", headless: bool = True, **_ignored: Any):
        self.cfg = IsaacConfig(num_envs=int(num_envs), image_size=int(image_size),
                               base_seed=int(base_seed), device=device, headless=headless)
        self.batch_size = self.cfg.num_envs
        self.image_size = self.cfg.image_size
        self.base_seed = self.cfg.base_seed
        self.table_z = TABLE_HEIGHT

        self._app = None
        self._sim = None
        self._scene = None
        self._states: list[EpisodeState | None] = [None] * self.batch_size
        self._views = [SimpleEnvView(self.table_z) for _ in range(self.batch_size)]
        self._intrinsics: CameraIntrinsics | None = None

        self._launch()
        self._build_scene()

    # -- startup -------------------------------------------------------------- #
    def _launch(self) -> None:
        """Start Isaac Sim. Nothing from ``isaaclab`` may be imported before this."""
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=self.cfg.headless,
                               enable_cameras=self.cfg.enable_cameras,
                               device=self.cfg.device)
        self._app = launcher.app

    def _build_scene(self) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass

        from .isaac_scene import build_scene_cfg

        self._sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=PHYSICS_DT, device=self.cfg.device))

        scene_cfg = build_scene_cfg(num_envs=self.cfg.num_envs, env_spacing=ENV_SPACING,
                                    image_size=self.cfg.image_size)
        self._scene = InteractiveScene(scene_cfg)
        self._sim.reset()

        self._robot = self._scene["robot"]
        self._objects = self._scene["object"]
        self._camera = self._scene["camera"]
        self._controller = self._make_ik_controller()
        # Recorded once: the camera is fixed, so its intrinsics never change.
        self._intrinsics = self._camera_intrinsics()
        _ = configclass, InteractiveSceneCfg  # imported for the scene module's benefit

    def _make_ik_controller(self):
        """Damped least squares, matching the MuJoCo controller's IK method."""
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                          ik_method="dls")
        return DifferentialIKController(cfg, num_envs=self.batch_size,
                                        device=self.cfg.device)

    def _camera_intrinsics(self) -> CameraIntrinsics:
        """Intrinsics matching ``simgrasp.camera.intrinsics_from_model``.

        The MuJoCo camera is specified by a vertical field of view and a square
        image, which makes pixels square and ``fx == fy``. The Isaac camera is
        configured in ``isaac_scene.py`` to the same field of view, so the same
        formula gives the same numbers, and the whole projection and
        deprojection path can be reused from ``simgrasp.camera`` unchanged.
        """
        size = self.cfg.image_size
        focal = (size / 2.0) / np.tan(np.deg2rad(CAMERA_FOVY_DEG) / 2.0)
        return CameraIntrinsics(fx=focal, fy=focal, cx=(size - 1) / 2.0,
                                cy=(size - 1) / 2.0, width=size, height=size)

    # -- lifecycle ------------------------------------------------------------ #
    def close(self) -> None:
        if self._app is not None:
            self._app.close()
            self._app = None

    def __enter__(self) -> IsaacBackend:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reset ---------------------------------------------------------------- #
    def reset(self, episodes: Sequence[int], split: str = "all") -> list[Observation]:
        """Place a fresh object in each environment and return the RGB-D views.

        Object identity comes from ``simgrasp``: episode *i* draws its category
        and dimensions from ``rng_for_episode(base_seed, i)``, exactly as the
        MuJoCo original does, so the two backends see the same scene for the same
        index. Only the placement into the simulator is new here.
        """
        if len(episodes) > self.batch_size:
            raise ValueError(f"got {len(episodes)} episodes for {self.batch_size} environments")

        specs, placements = [], []
        for episode in episodes:
            rng = rng_for_episode(self.base_seed, int(episode))
            category = sample_category(rng, split)
            spec = sample_object(rng, category)
            specs.append((int(episode), category, spec))
            placements.append(self._sample_placement(rng, spec))

        self._apply_object_geometry([s for _, _, s in specs])
        self._apply_object_materials([s for _, _, s in specs])
        self._set_object_poses(placements)
        self._retract_arm()
        self._settle(SETTLE_TIME)

        positions, yaws = self._object_poses()
        for i, ((episode, category, spec), (spawn_xy, _spawn_yaw)) in enumerate(
                zip(specs, placements, strict=True)):
            settled_xy = (float(positions[i, 0]), float(positions[i, 1]))
            self._states[i] = EpisodeState(
                index=episode, seed=self.base_seed, category=category, spec=spec,
                object_xy=settled_xy, object_yaw=float(yaws[i]),
                object_z0=float(positions[i, 2]), spawn_xy=spawn_xy,
                settle_displacement=float(np.linalg.norm(
                    np.array(settled_xy) - np.array(spawn_xy))),
            )
            self._views[i].state = self._states[i]
        for view in self._views[len(episodes):]:
            view.state = None

        return self.observe()[: len(episodes)]

    @staticmethod
    def _sample_placement(rng, spec) -> tuple[tuple[float, float], float]:
        """Where the object is dropped, using the same margins as the MuJoCo scene."""
        margin = spec.footprint_radius + 0.02
        x = float(rng.uniform(WORKSPACE_X[0] + margin, WORKSPACE_X[1] - margin))
        y = float(rng.uniform(WORKSPACE_Y[0] + margin, WORKSPACE_Y[1] - margin))
        return (x, y), float(rng.uniform(-np.pi, np.pi))

    def _apply_object_geometry(self, specs) -> None:
        """Rewrite each environment's object prims to the sampled dimensions.

        This is the operation whose runtime behaviour is unverified. See the
        module docstring and ``scripts/check_setup.py`` stage 4.
        """
        from pxr import Gf, UsdGeom

        stage = self._stage()
        for env_index, spec in enumerate(specs):
            prims = prims_for_spec(spec)
            for slot in range(MAX_PRIMS):
                placement = prims[slot] if slot < len(prims) else None
                for prim_type in PRIM_TYPES:
                    path = self._prim_path(env_index, slot, prim_type)
                    prim = stage.GetPrimAtPath(path)
                    if not prim.IsValid():
                        raise RuntimeError(
                            f"expected prim {path}; the scene was not built with the "
                            f"{MAX_PRIMS}-slot layout this backend requires")
                    xform = UsdGeom.Xformable(prim)
                    active = placement is not None and placement.prim_type == prim_type
                    scale = placement.scale if active else inactive_scale()
                    translate = placement.translate if active else INACTIVE_TRANSLATE
                    orient = placement.orient_wxyz if active else (1.0, 0.0, 0.0, 0.0)
                    self._set_xform(xform, Gf, translate, orient, scale)

    @staticmethod
    def _set_xform(xform, Gf, translate, orient, scale) -> None:
        """Write translate, orient and scale ops, reusing them if they exist."""
        ops = {op.GetOpName().split(":")[-1]: op for op in xform.GetOrderedXformOps()}
        translate_op = ops.get("translate") or xform.AddTranslateOp()
        orient_op = ops.get("orient") or xform.AddOrientOp()
        scale_op = ops.get("scale") or xform.AddScaleOp()
        translate_op.Set(Gf.Vec3d(*translate))
        orient_op.Set(Gf.Quatf(orient[0], Gf.Vec3f(orient[1], orient[2], orient[3])))
        scale_op.Set(Gf.Vec3f(*scale))

    def _apply_object_materials(self, specs) -> None:
        """Set each object's mass and friction from the catalogue.

        MuJoCo derives a body's mass from its geoms' volumes and the sampled
        density, and samples a sliding friction per object. Neither carries over
        through the geometry, and both change outcomes: mass sets the moment the
        jaws have to resist during the 0.20 m lift, which is what makes the
        dumbbell the hardest category, and friction decides whether a curved
        surface slips out during closing.
        """
        import torch

        from .isaac_objects import spec_mass

        masses = torch.tensor([spec_mass(spec) for spec in specs], dtype=torch.float32)
        view = self._objects.root_physx_view
        current = view.get_masses()
        current[: len(specs), 0] = masses
        view.set_masses(current, torch.arange(current.shape[0]))

        materials = view.get_material_properties()
        for i, spec in enumerate(specs):
            slide = float(spec.friction[0])
            # PhysX takes (static friction, dynamic friction, restitution) per
            # collision shape; the catalogue's single sliding coefficient is used
            # for both, and restitution stays at the scene default of zero.
            materials[i, :, 0] = slide
            materials[i, :, 1] = slide
        view.set_material_properties(materials, torch.arange(materials.shape[0]))

    def _prim_path(self, env_index: int, slot: int, prim_type: str) -> str:
        return (f"/World/envs/env_{env_index}/Object/slot_{slot}/{prim_type}")

    def _stage(self):
        import omni.usd

        return omni.usd.get_context().get_stage()

    def _set_object_poses(self, placements) -> None:
        origins = self._scene.env_origins
        root = self._objects.data.default_root_state.clone()
        for i, ((x, y), yaw) in enumerate(placements):
            root[i, 0] = origins[i, 0] + x
            root[i, 1] = origins[i, 1] + y
            root[i, 2] = origins[i, 2] + TABLE_HEIGHT
            half = 0.5 * yaw
            root[i, 3] = float(np.cos(half))
            root[i, 4] = 0.0
            root[i, 5] = 0.0
            root[i, 6] = float(np.sin(half))
        root[:, 7:] = 0.0
        self._objects.write_root_state_to_sim(root)

    def _retract_arm(self) -> None:
        """Park the arm in the capture pose so it never occludes the object."""
        import torch
        from simgrasp.scene import CAPTURE_QPOS

        joints = self._robot.data.default_joint_pos.clone()
        capture = torch.tensor(CAPTURE_QPOS, dtype=joints.dtype, device=joints.device)
        joints[:, : capture.numel()] = capture
        # Fingers open.
        joints[:, capture.numel():] = 0.04
        velocities = torch.zeros_like(joints)
        self._robot.write_joint_state_to_sim(joints, velocities)
        self._robot.set_joint_position_target(joints)

    def _settle(self, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds / PHYSICS_DT)))):
            self._scene.write_data_to_sim()
            self._sim.step(render=False)
            self._scene.update(PHYSICS_DT)

    def _object_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Settled object positions in table coordinates, and their yaws."""
        root = self._objects.data.root_state_w.detach().cpu().numpy()
        origins = self._scene.env_origins.detach().cpu().numpy()
        positions = root[:, :3] - origins
        yaws = np.empty(root.shape[0])
        for i, quat in enumerate(root[:, 3:7]):
            mat = quat_to_mat(quat)
            yaws[i] = np.arctan2(mat[1, 0], mat[0, 0])
        return positions, yaws

    # -- observation ---------------------------------------------------------- #
    def observe(self) -> list[Observation]:
        """Render every environment and build the height maps.

        The depth product is ``distance_to_image_plane``, which is the
        perpendicular distance along the optical axis, the same quantity MuJoCo's
        depth buffer holds. Using ``distance_to_camera`` instead would be a ray
        length, and deprojecting it with the perpendicular formula bows the table
        into a bowl and puts every off-centre grasp at the wrong height.
        """
        self._scene.write_data_to_sim()
        self._sim.step(render=True)
        self._scene.update(PHYSICS_DT)
        self._camera.update(PHYSICS_DT)

        rgb = self._camera.data.output["rgb"].detach().cpu().numpy()[..., :3].astype(np.uint8)
        depth = self._camera.data.output["distance_to_image_plane"].detach().cpu().numpy()
        depth = np.squeeze(depth, axis=-1).astype(np.float32)

        cam_pos_w = self._camera.data.pos_w.detach().cpu().numpy()
        cam_quat = self._camera.data.quat_w_opengl.detach().cpu().numpy()
        origins = self._scene.env_origins.detach().cpu().numpy()

        observations = []
        for i in range(self.batch_size):
            # Express the camera in the environment's own frame, which is the
            # frame the MuJoCo scene and every grasp are defined in.
            cam_pos = cam_pos_w[i] - origins[i]
            cam_mat = quat_to_mat(cam_quat[i])
            observations.append(Observation(
                rgb=rgb[i],
                depth=depth[i],
                height=height_map(depth[i], cam_pos, cam_mat, self._intrinsics, self.table_z),
                cam_pos=cam_pos,
                cam_mat=cam_mat,
                intrinsics=self._intrinsics,
                table_z=self.table_z,
            ))
        return observations

    def views(self):
        return list(self._views)

    # -- state ---------------------------------------------------------------- #
    def snapshot(self) -> Any:
        """Full simulator state, so K grasps can share one settled scene."""
        return {
            "object": self._objects.data.root_state_w.clone(),
            "joint_pos": self._robot.data.joint_pos.clone(),
            "joint_vel": self._robot.data.joint_vel.clone(),
        }

    def restore(self, snap: Any) -> None:
        self._objects.write_root_state_to_sim(snap["object"].clone())
        self._robot.write_joint_state_to_sim(snap["joint_pos"].clone(),
                                             snap["joint_vel"].clone())
        self._scene.write_data_to_sim()
        self._sim.step(render=False)
        self._scene.update(PHYSICS_DT)

    # -- execution ------------------------------------------------------------ #
    def execute(self, grasps: Sequence[Grasp]) -> list[GraspResult]:
        """Run the scripted approach, grasp and lift in every environment.

        The phase sequence and its durations are those of the MuJoCo controller:
        pre-shape the gripper, move above the grasp, converge, descend straight
        down, close, lift 0.20 m, hold. Every environment runs the same phases at
        the same time, which is the entire throughput argument for this port.
        """
        import torch

        n = len(grasps)
        if n > self.batch_size:
            raise ValueError(f"got {n} grasps for {self.batch_size} environments")

        z0 = np.array([state.object_z0 if state else 0.0 for state in self._states])
        targets = self._grasp_targets(grasps)
        approach = targets.clone()
        approach[:n, 2] += APPROACH_HEIGHT
        lift = targets.clone()
        lift[:n, 2] += LIFT_DISTANCE

        preshape = torch.tensor(
            [0.5 * g.preshape_width() for g in grasps] + [0.04] * (self.batch_size - n),
            device=self.cfg.device, dtype=torch.float32)

        self._drive_gripper(preshape, PRESHAPE_TIME)
        ik_failures = self._move_to_pose(approach, APPROACH_TIME + CONVERGE_TIME)
        self._move_to_pose(targets, DESCEND_TIME)
        self._drive_gripper(torch.zeros_like(preshape), CLOSE_TIME)
        self._move_to_pose(lift, LIFT_TIME)
        self._settle(HOLD_TIME)

        positions, _ = self._object_poses()
        widths = self._gripper_widths()
        results = []
        for i in range(n):
            if ik_failures[i]:
                results.append(GraspResult(False, "ik_failed", grasps[i], ik_failures=1))
                continue
            lift_height = float(positions[i, 2] - z0[i])
            width = float(widths[i])
            success, reason = classify_outcome(lift_height, width, LIFT_SUCCESS_THRESHOLD)
            state = self._states[i]
            displacement = float(np.linalg.norm(
                positions[i, :2] - np.array(state.object_xy))) if state else 0.0
            results.append(GraspResult(
                success=success, reason=reason, grasp=grasps[i],
                lift_height=lift_height, final_gripper_width=width,
                object_displacement=displacement))
        return results

    def _grasp_targets(self, grasps: Sequence[Grasp]):
        """Grasp poses as an Isaac IK command: position then quaternion.

        Two conversions happen here, and both are silent when wrong.

        **TCP to hand body.** A grasp is specified at the fingertip-pad centre,
        which sits ``TCP_OFFSET_Z`` along the hand's +z. The IK controller
        drives the hand body, so the commanded position is the grasp position
        minus that offset rotated into the world. For a top-down grasp the
        hand's +z points down, so the hand sits 103.4 mm *above* the grasp
        point. Commanding the hand to the grasp position would drive the gripper
        straight through the table.

        **World to robot base.** The controller works in the robot's base frame,
        and the base stands on a plinth 0.40 m up so that it is level with the
        table top. The offset is read from the robot's actual root state rather
        than assumed, so this stays correct if the mount ever moves.
        """
        import torch
        from simgrasp.scene import TCP_OFFSET_Z
        from simgrasp.transforms import mat_to_quat, topdown_grasp_mat

        base = (self._robot.data.root_state_w[:, :3]
                - self._scene.env_origins).detach().cpu().numpy()

        command = torch.zeros((self.batch_size, 7), device=self.cfg.device,
                              dtype=torch.float32)
        # A neutral, reachable pose for any unused environment, so an
        # under-filled batch cannot drag the solver somewhere degenerate.
        command[:, 0] = float(np.mean(WORKSPACE_X)) - float(np.mean(base[:, 0]))
        command[:, 2] = TABLE_HEIGHT + 0.25 - float(np.mean(base[:, 2]))
        command[:, 3] = 1.0

        for i, grasp in enumerate(grasps):
            mat = topdown_grasp_mat(grasp.yaw)
            tcp = np.array([grasp.x, grasp.y, grasp.z], dtype=np.float64)
            hand = tcp - mat @ np.array([0.0, 0.0, TCP_OFFSET_Z])
            command[i, :3] = torch.tensor(hand - base[i], device=self.cfg.device,
                                          dtype=torch.float32)
            command[i, 3:7] = torch.tensor(mat_to_quat(mat), device=self.cfg.device,
                                           dtype=torch.float32)
        return command

    def _move_to_pose(self, command, seconds: float) -> np.ndarray:
        """Servo every arm to its commanded TCP pose for ``seconds``.

        Returns a per-environment flag for whether the solver failed to make
        progress. An unreachable proposal is a policy error, not a free pass, and
        is recorded as a failed attempt exactly as the MuJoCo environment does.
        """
        import torch
        from isaaclab.utils.math import subtract_frame_transforms

        steps = max(1, int(round(seconds / PHYSICS_DT)))
        arm_joints = self._arm_joint_indices()
        ee_index = self._ee_body_index()

        self._controller.reset()
        self._controller.set_command(command)
        finite = torch.ones(self.batch_size, dtype=torch.bool, device=self.cfg.device)

        for _ in range(steps):
            root_pose = self._robot.data.root_state_w[:, :7]
            ee_pose_w = self._robot.data.body_state_w[:, ee_index, :7]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose[:, :3], root_pose[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7])
            jacobian = self._robot.root_physx_view.get_jacobians()[
                :, ee_index - 1, :, arm_joints]
            joint_pos = self._robot.data.joint_pos[:, arm_joints]
            target = self._controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            finite &= torch.isfinite(target).all(dim=-1)
            target = torch.nan_to_num(target, nan=0.0)
            self._robot.set_joint_position_target(target, joint_ids=arm_joints)
            self._scene.write_data_to_sim()
            self._sim.step(render=False)
            self._scene.update(PHYSICS_DT)

        return (~finite).detach().cpu().numpy()

    def _drive_gripper(self, half_widths, seconds: float) -> None:
        """Command both finger joints to half the requested opening."""
        import torch

        finger_joints = self._finger_joint_indices()
        target = torch.stack([half_widths, half_widths], dim=-1)
        for _ in range(max(1, int(round(seconds / PHYSICS_DT)))):
            self._robot.set_joint_position_target(target, joint_ids=finger_joints)
            self._scene.write_data_to_sim()
            self._sim.step(render=False)
            self._scene.update(PHYSICS_DT)

    def _gripper_widths(self) -> np.ndarray:
        joints = self._finger_joint_indices()
        pos = self._robot.data.joint_pos[:, joints].detach().cpu().numpy()
        return pos.sum(axis=-1)

    def _arm_joint_indices(self):
        if not hasattr(self, "_arm_ids"):
            names = [f"panda_joint{i}" for i in range(1, 8)]
            self._arm_ids = self._robot.find_joints(names)[0]
        return self._arm_ids

    def _finger_joint_indices(self):
        if not hasattr(self, "_finger_ids"):
            self._finger_ids = self._robot.find_joints(["panda_finger_joint.*"])[0]
        return self._finger_ids

    def _ee_body_index(self) -> int:
        if not hasattr(self, "_ee_id"):
            self._ee_id = self._robot.find_bodies(["panda_hand"])[0][0]
        return self._ee_id
