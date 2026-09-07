"""The Isaac Lab scene: table, Franka, one multi-slot object, one overhead camera.

STATUS: NOT YET EXECUTED. See :mod:`isaacgrasp.backends.isaac_backend`.

Everything positional in here is read from ``simgrasp.scene`` rather than
retyped, so the table, the workspace and the camera cannot drift away from the
study this is compared against. The only numbers written out are the ones USD
needs and MuJoCo does not: the camera's focal length and aperture, which are
derived from the MuJoCo field of view so both cameras see the same solid angle.

Object bodies
-------------
The catalogue's objects are compositions of up to three primitives, and the
category changes every episode. PhysX creates its collision shapes when the
scene is built, so a prim absent at build time cannot appear later without
rebuilding. Each object body therefore carries
:data:`~isaacgrasp.backends.isaac_objects.MAX_PRIMS` slots, and each slot carries
one prim of every type the catalogue uses. An episode activates one per slot and
shrinks the rest. That is twelve prims per object where at most three are ever
visible, which is wasteful in memory and free at runtime, and it is what allows
the scene to be built once for a run of any length.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from simgrasp.scene import (
    CAMERA_FOVY_DEG,
    CAMERA_HEIGHT,
    CAMERA_X,
    TABLE_CENTER_X,
    TABLE_HALF,
    TABLE_HEIGHT,
)

from .isaac_objects import INACTIVE_TRANSLATE, MAX_PRIMS, PRIM_TYPES, inactive_scale

# USD sensor aperture, in millimetres. Any value works as long as the focal
# length is derived from it; this is the Isaac default, kept so the numbers look
# familiar next to other Isaac Lab scenes.
HORIZONTAL_APERTURE = 20.955


def focal_length_for_fovy(fovy_deg: float, aperture: float = HORIZONTAL_APERTURE) -> float:
    """Focal length giving ``fovy_deg`` on a square sensor of ``aperture``.

    The image is square, so the horizontal and vertical apertures are equal and
    the vertical field of view is ``2 * atan(aperture / (2 * focal))``. Matching
    the MuJoCo camera's field of view is what lets the whole projection and
    deprojection path be reused from ``simgrasp.camera`` unchanged.
    """
    return float(aperture / (2.0 * math.tan(math.radians(fovy_deg) / 2.0)))


def spawn_multi_slot_object(prim_path: str, cfg, translation=None, orientation=None):
    """Build one object body: ``MAX_PRIMS`` slots, each holding every prim type.

    Returns the body prim. The rigid body and mass APIs go on the parent so the
    slots move as one object, which is what makes an L-shape or a dumbbell a
    single rigid body rather than three loose pieces.
    """
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    body = UsdGeom.Xform.Define(stage, prim_path)
    if translation is not None:
        body.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        body.AddOrientOp().Set(Gf.Quatf(orientation[0], Gf.Vec3f(*orientation[1:])))

    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim())

    makers = {
        "Cube": UsdGeom.Cube,
        "Sphere": UsdGeom.Sphere,
        "Cylinder": UsdGeom.Cylinder,
        "Capsule": UsdGeom.Capsule,
    }
    for slot in range(MAX_PRIMS):
        UsdGeom.Xform.Define(stage, f"{prim_path}/slot_{slot}")
        for prim_type in PRIM_TYPES:
            path = f"{prim_path}/slot_{slot}/{prim_type}"
            geom = makers[prim_type].Define(stage, path)
            # Unit dimensions, so a single scale drives every shape. The runtime
            # rewrite in isaac_backend only ever touches the transform ops.
            if prim_type == "Cube":
                geom.CreateSizeAttr(1.0)
            else:
                geom.CreateRadiusAttr(1.0)
                if prim_type in ("Cylinder", "Capsule"):
                    geom.CreateHeightAttr(1.0)
                    geom.CreateAxisAttr("Z")
            xform = UsdGeom.Xformable(geom.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*INACTIVE_TRANSLATE))
            xform.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            xform.AddScaleOp().Set(Gf.Vec3f(*inactive_scale()))
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())

    return body.GetPrim()


@configclass
class MultiSlotObjectCfg(sim_utils.SpawnerCfg):
    """Spawner config selecting :func:`spawn_multi_slot_object`."""

    func = spawn_multi_slot_object


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    """One table, one Franka, one object and one overhead camera per environment."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.95)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(2.0 * TABLE_HALF[0], 2.0 * TABLE_HALF[1], 2.0 * TABLE_HALF[2]),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.52, 0.48)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_CENTER_X, 0.0, TABLE_HEIGHT / 2.0)),
    )
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=MultiSlotObjectCfg(),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(TABLE_CENTER_X, 0.0, TABLE_HEIGHT)),
    )
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        # Straight down. In the OpenGL convention a camera looks along its own
        # -Z, so the identity rotation already points it at the table.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(CAMERA_X, 0.0, TABLE_HEIGHT + CAMERA_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length_for_fovy(CAMERA_FOVY_DEG),
            horizontal_aperture=HORIZONTAL_APERTURE,
            clipping_range=(0.05, 3.0),
        ),
        width=224,
        height=224,
    )


def build_scene_cfg(num_envs: int, env_spacing: float, image_size: int) -> GraspSceneCfg:
    """A scene config at the requested parallelism and image resolution."""
    cfg = GraspSceneCfg(num_envs=int(num_envs), env_spacing=float(env_spacing))
    cfg.camera.width = int(image_size)
    cfg.camera.height = int(image_size)
    return cfg
