"""The stub modules themselves.

Installed into ``sys.modules`` before ``isaacgrasp.backends.isaac_backend`` is
imported, so its lazy imports resolve to these instead of failing.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

INSTALLED: list[str] = []


# --------------------------------------------------------------------------- #
# USD
# --------------------------------------------------------------------------- #


class FakePrim:
    """A USD prim, remembering the transform ops written to it."""

    def __init__(self, path: str):
        self.path = path
        self.ops: dict[str, Any] = {}
        self.attributes: dict[str, Any] = {}

    def IsValid(self) -> bool:  # noqa: N802 - USD's own casing
        return True

    def GetPrim(self):  # noqa: N802
        return self


class FakeXformOp:
    def __init__(self, prim: FakePrim, name: str):
        self.prim = prim
        self.name = name

    def GetOpName(self) -> str:  # noqa: N802
        return f"xformOp:{self.name}"

    def Set(self, value) -> None:  # noqa: N802
        self.prim.ops[self.name] = value


class FakeXformable:
    def __init__(self, prim: FakePrim):
        self.prim = prim

    def GetOrderedXformOps(self):  # noqa: N802
        return [FakeXformOp(self.prim, name) for name in self.prim.ops]

    def AddTranslateOp(self):  # noqa: N802
        return FakeXformOp(self.prim, "translate")

    def AddOrientOp(self):  # noqa: N802
        return FakeXformOp(self.prim, "orient")

    def AddScaleOp(self):  # noqa: N802
        return FakeXformOp(self.prim, "scale")


class FakeStage:
    def __init__(self):
        self.prims: dict[str, FakePrim] = {}

    def GetPrimAtPath(self, path: str) -> FakePrim:  # noqa: N802
        return self.prims.setdefault(str(path), FakePrim(str(path)))

    def DefinePrim(self, path: str) -> FakePrim:  # noqa: N802
        return self.GetPrimAtPath(path)


STAGE = FakeStage()


def _gf_module() -> types.ModuleType:
    module = types.ModuleType("pxr.Gf")

    class Vec3d(tuple):
        def __new__(cls, *values):
            return super().__new__(cls, values)

    class Vec3f(Vec3d):
        pass

    class Quatf:
        def __init__(self, real, imaginary):
            self.real = real
            self.imaginary = tuple(imaginary)

        def __eq__(self, other):
            return (self.real, self.imaginary) == (other.real, other.imaginary)

    module.Vec3d = Vec3d
    module.Vec3f = Vec3f
    module.Quatf = Quatf
    return module


def _usdgeom_module() -> types.ModuleType:
    module = types.ModuleType("pxr.UsdGeom")

    class _Definable:
        @staticmethod
        def Define(stage, path):  # noqa: N802
            prim = stage.GetPrimAtPath(path)
            prim.attributes.setdefault("type", "shape")
            return _Shape(prim)

    class _Shape:
        def __init__(self, prim: FakePrim):
            self.prim = prim

        def GetPrim(self):  # noqa: N802
            return self.prim

        def CreateSizeAttr(self, value):  # noqa: N802
            self.prim.attributes["size"] = value

        def CreateRadiusAttr(self, value):  # noqa: N802
            self.prim.attributes["radius"] = value

        def CreateHeightAttr(self, value):  # noqa: N802
            self.prim.attributes["height"] = value

        def CreateAxisAttr(self, value):  # noqa: N802
            self.prim.attributes["axis"] = value

        def AddTranslateOp(self):  # noqa: N802
            return FakeXformOp(self.prim, "translate")

        def AddOrientOp(self):  # noqa: N802
            return FakeXformOp(self.prim, "orient")

    for name in ("Cube", "Sphere", "Cylinder", "Capsule", "Xform"):
        setattr(module, name, type(name, (_Definable,), {}))
    module.Xformable = FakeXformable
    return module


def _usdphysics_module() -> types.ModuleType:
    module = types.ModuleType("pxr.UsdPhysics")

    class _Api:
        applied: list[str] = []

        @classmethod
        def Apply(cls, prim):  # noqa: N802
            cls.applied.append(prim.path)
            return prim

    for name in ("RigidBodyAPI", "MassAPI", "CollisionAPI"):
        setattr(module, name, type(name, (_Api,), {"applied": []}))
    return module


# --------------------------------------------------------------------------- #
# Isaac Lab
# --------------------------------------------------------------------------- #


@dataclass
class FakeArticulationData:
    num_envs: int
    num_joints: int = 9
    joint_pos: torch.Tensor = None
    joint_vel: torch.Tensor = None
    default_joint_pos: torch.Tensor = None
    root_state_w: torch.Tensor = None
    body_state_w: torch.Tensor = None

    def __post_init__(self):
        self.joint_pos = torch.zeros(self.num_envs, self.num_joints)
        self.joint_vel = torch.zeros(self.num_envs, self.num_joints)
        self.default_joint_pos = torch.zeros(self.num_envs, self.num_joints)
        # Root at the top of the plinth: the scene mounts the arm at z = 0.40.
        self.root_state_w = torch.zeros(self.num_envs, 13)
        self.root_state_w[:, 2] = 0.40
        self.root_state_w[:, 3] = 1.0
        # One body per link plus the hand; only the hand's index is read.
        self.body_state_w = torch.zeros(self.num_envs, 12, 13)
        self.body_state_w[..., 3] = 1.0


class FakePhysxView:
    def __init__(self, num_envs: int, num_shapes: int = 12):
        self.num_envs = num_envs
        self.num_shapes = num_shapes
        self.masses = torch.ones(num_envs, 1)
        self.materials = torch.zeros(num_envs, num_shapes, 3)
        self.jacobians = torch.zeros(num_envs, 11, 6, 9)
        # A non-degenerate Jacobian, so the damped least squares solve returns
        # something finite rather than NaN.
        for i in range(6):
            self.jacobians[:, :, i, i] = 1.0

    def get_masses(self):
        return self.masses.clone()

    def set_masses(self, values, indices):
        self.masses = values.clone()

    def get_material_properties(self):
        return self.materials.clone()

    def set_material_properties(self, values, indices):
        self.materials = values.clone()

    def get_jacobians(self):
        return self.jacobians


class FakeArticulation:
    def __init__(self, num_envs: int):
        self.data = FakeArticulationData(num_envs)
        self.root_physx_view = FakePhysxView(num_envs)
        self.joint_targets: list[tuple] = []
        self.written_states: list[tuple] = []

    def find_joints(self, names):
        # panda_joint1..7 then the two fingers, in that order.
        if any("finger" in str(n) for n in names):
            return ([7, 8], ["panda_finger_joint1", "panda_finger_joint2"])
        return ([0, 1, 2, 3, 4, 5, 6], list(names))

    def find_bodies(self, names):
        return ([8], list(names))

    def write_joint_state_to_sim(self, pos, vel):
        self.data.joint_pos = pos.clone()
        self.data.joint_vel = vel.clone()
        self.written_states.append((pos.clone(), vel.clone()))

    def set_joint_position_target(self, target, joint_ids=None):
        self.joint_targets.append((target.clone(), joint_ids))
        if joint_ids is None:
            self.data.joint_pos = target.clone()
        else:
            self.data.joint_pos[:, joint_ids] = target


class FakeRigidObjectData:
    def __init__(self, num_envs: int):
        self.default_root_state = torch.zeros(num_envs, 13)
        self.default_root_state[:, 3] = 1.0
        self.root_state_w = torch.zeros(num_envs, 13)
        self.root_state_w[:, 3] = 1.0


class FakeRigidObject:
    def __init__(self, num_envs: int):
        self.data = FakeRigidObjectData(num_envs)
        self.root_physx_view = FakePhysxView(num_envs)
        self.writes: list[torch.Tensor] = []

    def write_root_state_to_sim(self, state):
        self.data.root_state_w = state.clone()
        self.writes.append(state.clone())


class FakeCameraData:
    def __init__(self, num_envs: int, size: int):
        self.num_envs = num_envs
        self.size = size
        self.output = {
            "rgb": torch.zeros(num_envs, size, size, 3, dtype=torch.uint8),
            # A flat table by default; tests that need an object write into it.
            "distance_to_image_plane": torch.full((num_envs, size, size, 1), 0.55),
        }
        self.pos_w = torch.zeros(num_envs, 3)
        self.pos_w[:, 0] = 0.54
        self.pos_w[:, 2] = 0.95
        # Identity in the OpenGL convention: looking straight down.
        self.quat_w_opengl = torch.zeros(num_envs, 4)
        self.quat_w_opengl[:, 0] = 1.0


class FakeCamera:
    """A stub camera that can be told what the scene contains.

    By default it renders a flat table, which is what a misconfigured headless
    Isaac Sim produces and is therefore worth being the default. Setting
    ``height_source`` to a callable returning one height per environment makes it
    render an object of that height, which is what the geometry and camera stages
    of the setup gate need in order to be exercised at all.
    """

    def __init__(self, num_envs: int, size: int):
        self.data = FakeCameraData(num_envs, size)
        self.updates = 0
        self.height_source = None

    def update(self, dt):
        self.updates += 1
        if self.height_source is None:
            return
        size = self.data.size
        depth = self.data.output["distance_to_image_plane"]
        depth[:] = 0.55
        centre = size // 2
        half = max(1, size // 8)
        for env_index, height in enumerate(self.height_source()):
            if env_index >= depth.shape[0]:
                break
            depth[env_index, centre - half:centre + half,
                  centre - half:centre + half, 0] = 0.55 - float(height)


class FakeScene:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        size = getattr(cfg.camera, "width", 224)
        self._assets = {
            "robot": FakeArticulation(self.num_envs),
            "object": FakeRigidObject(self.num_envs),
            "camera": FakeCamera(self.num_envs, size),
        }
        self.env_origins = torch.zeros(self.num_envs, 3)
        for i in range(self.num_envs):
            self.env_origins[i, 0] = 2.5 * i
        # root_state_w is the WORLD pose, so it includes the environment origin.
        # Getting this wrong in the stub is what made the first run of
        # test_each_environment_gets_its_own_grasp fail: the backend recovers the
        # mount position as root_state_w - env_origins, which is only the local
        # 0.40 m plinth height when the world pose actually carries the origin.
        robot = self._assets["robot"]
        robot.data.root_state_w[:, :3] += self.env_origins
        self.writes = 0
        self.updates = 0

    def __getitem__(self, key):
        return self._assets[key]

    def write_data_to_sim(self):
        self.writes += 1

    def update(self, dt):
        self.updates += 1
        self._assets["camera"].update(dt)


class FakeSim:
    def __init__(self, cfg):
        self.cfg = cfg
        self.steps = 0
        self.renders = 0

    def reset(self):
        pass

    def step(self, render=False):
        self.steps += 1
        if render:
            self.renders += 1


class FakeIKController:
    """A stand-in that returns a finite joint target of the right shape."""

    def __init__(self, cfg, num_envs, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.commands: list[torch.Tensor] = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def set_command(self, command):
        self.commands.append(command.clone())

    def compute(self, ee_pos_b, ee_quat_b, jacobian, joint_pos):
        return joint_pos + 0.01


@dataclass
class _Cfg:
    """A permissive config object: any keyword is accepted and recorded."""

    _values: dict = field(default_factory=dict)

    def __init__(self, **kwargs):
        object.__setattr__(self, "_values", dict(kwargs))
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)

    def replace(self, **kwargs):
        merged = {**self._values, **kwargs}
        clone = type(self)(**merged)
        for key, value in vars(self).items():
            if key != "_values" and key not in merged:
                object.__setattr__(clone, key, value)
        return clone


def _permissive(name: str):
    return type(name, (_Cfg,), {})


def install_fake_isaac(num_envs_hint: int = 4) -> FakeStage:
    """Put the stub modules into ``sys.modules``. Returns the shared fake stage."""
    global STAGE
    STAGE = FakeStage()

    modules: dict[str, types.ModuleType] = {}

    def module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        modules[name] = mod
        return mod

    # pxr
    pxr = module("pxr")
    pxr.Gf = _gf_module()
    pxr.UsdGeom = _usdgeom_module()
    pxr.UsdPhysics = _usdphysics_module()
    modules["pxr.Gf"] = pxr.Gf
    modules["pxr.UsdGeom"] = pxr.UsdGeom
    modules["pxr.UsdPhysics"] = pxr.UsdPhysics

    # omni.usd
    omni = module("omni")
    omni_usd = module("omni.usd")
    omni_usd.get_context = lambda: types.SimpleNamespace(get_stage=lambda: STAGE)
    omni.usd = omni_usd

    # isaacsim, present only so an import check can succeed
    module("isaacsim")

    # isaaclab
    isaaclab = module("isaaclab")
    isaaclab.__version__ = "fake"

    app = module("isaaclab.app")

    class AppLauncher:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.app = types.SimpleNamespace(close=lambda: None)

    app.AppLauncher = AppLauncher
    isaaclab.app = app

    sim = module("isaaclab.sim")
    sim.SimulationCfg = _permissive("SimulationCfg")
    sim.SimulationContext = FakeSim
    for name in ("GroundPlaneCfg", "DomeLightCfg", "CuboidCfg", "PreviewSurfaceCfg",
                 "CollisionPropertiesCfg", "PinholeCameraCfg", "SpawnerCfg",
                 "RigidBodyPropertiesCfg"):
        setattr(sim, name, _permissive(name))
    isaaclab.sim = sim

    scene = module("isaaclab.scene")
    scene.InteractiveScene = FakeScene

    class InteractiveSceneCfg:
        def __init__(self, num_envs=1, env_spacing=2.0, **kwargs):
            self.num_envs = num_envs
            self.env_spacing = env_spacing
            for key, value in kwargs.items():
                setattr(self, key, value)

    scene.InteractiveSceneCfg = InteractiveSceneCfg
    isaaclab.scene = scene

    sensors = module("isaaclab.sensors")

    class TiledCameraCfg(_Cfg):
        OffsetCfg = _permissive("OffsetCfg")

    sensors.TiledCameraCfg = TiledCameraCfg
    sensors.TiledCamera = FakeCamera
    isaaclab.sensors = sensors

    assets = module("isaaclab.assets")

    class AssetBaseCfg(_Cfg):
        InitialStateCfg = _permissive("InitialStateCfg")

    class RigidObjectCfg(_Cfg):
        InitialStateCfg = _permissive("InitialStateCfg")

    assets.AssetBaseCfg = AssetBaseCfg
    assets.RigidObjectCfg = RigidObjectCfg
    assets.Articulation = FakeArticulation
    assets.RigidObject = FakeRigidObject
    isaaclab.assets = assets

    controllers = module("isaaclab.controllers")
    controllers.DifferentialIKController = FakeIKController
    controllers.DifferentialIKControllerCfg = _permissive("DifferentialIKControllerCfg")
    isaaclab.controllers = controllers

    utils = module("isaaclab.utils")
    utils.configclass = lambda cls: cls
    utils_math = module("isaaclab.utils.math")

    def subtract_frame_transforms(t_pos, t_quat, pos, quat):
        return pos - t_pos, quat

    utils_math.subtract_frame_transforms = subtract_frame_transforms
    utils.math = utils_math
    isaaclab.utils = utils

    isaaclab_assets = module("isaaclab_assets")
    robots = module("isaaclab_assets.robots")
    franka = module("isaaclab_assets.robots.franka")

    class _ArticulationCfg(_Cfg):
        InitialStateCfg = _permissive("InitialStateCfg")

    franka.FRANKA_PANDA_HIGH_PD_CFG = _ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=_permissive("InitialStateCfg")(pos=(0.0, 0.0, 0.0)),
        spawn=_permissive("UsdFileCfg")(
            rigid_props=_permissive("RigidBodyPropertiesCfg")(disable_gravity=False)),
    )
    robots.franka = franka
    isaaclab_assets.robots = robots

    for name, mod in modules.items():
        sys.modules[name] = mod
        INSTALLED.append(name)
    return STAGE


def uninstall_fake_isaac() -> None:
    for name in INSTALLED:
        sys.modules.pop(name, None)
    for name in list(sys.modules):
        if name.startswith("isaacgrasp.backends.isaac"):
            sys.modules.pop(name, None)
    INSTALLED.clear()
    _ = np
