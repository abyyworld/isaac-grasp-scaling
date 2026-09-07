"""The object catalogue, translated from MuJoCo geoms to USD primitives.

The catalogue itself is **not** redefined here. Shapes, dimension ranges, the
seen/held-out split and the grasp hints are all sampled by
``simgrasp.objects.sample_object``, so both simulators draw from one
distribution and episode *i* means the same object in both. This module only
translates the resulting :class:`~simgrasp.objects.ObjectSpec` into the form
Isaac needs, and the translation is where the port can silently go wrong.

Three conventions differ between the engines, and each is a place a mistake
would be invisible until the success rates came out wrong:

**Size semantics.** MuJoCo geom sizes are half-extents: a ``box`` of size
``(hx, hy, hz)`` spans ``2hx`` by ``2hy`` by ``2hz``. USD ``Cube`` has a
``size`` attribute that is a full edge length, and ``Sphere``/``Cylinder``/
``Capsule`` take a radius with cylinder and capsule taking a full ``height``.
Everything is converted to full extents here, once.

**Capsule and cylinder length.** MuJoCo's capsule size is
``(radius, half_length)`` where ``half_length`` excludes the hemispherical
caps, so the total length is ``2 * half_length + 2 * radius``. USD ``Capsule``
takes ``height`` as the length of the cylindrical section only, which matches
``2 * half_length``. MuJoCo's cylinder is ``(radius, half_height)``, so USD
``height`` is ``2 * half_height``.

**Axis.** Both engines default a cylinder and a capsule to its local z axis, so
the catalogue's ``euler=(0, pi/2, 0)`` rotations carry over unchanged as a
quaternion about y.

Everything in this module is pure geometry with no simulator import, which is
deliberate: it is the part of the port that can be tested without a GPU, and
``tests/test_isaac_objects.py`` checks each conversion against the MuJoCo
definition it came from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from simgrasp.objects import ObjectSpec

# The catalogue's most complex object is the dumbbell: two balls and a shaft.
# Every object body is built with this many slots so one prim layout serves the
# whole catalogue and a scene never has to be rebuilt to change category.
MAX_PRIMS = 3

# USD primitive names this module emits. A slot carries one prim of each type
# and activates exactly one, because PhysX will not create a collision shape of
# a type that was not present when the scene was built.
PRIM_TYPES = ("Cube", "Sphere", "Cylinder", "Capsule")

_MUJOCO_TO_USD = {
    "box": "Cube",
    "sphere": "Sphere",
    "cylinder": "Cylinder",
    "capsule": "Capsule",
    "ellipsoid": "Sphere",  # a Sphere with a non-uniform scale
}


@dataclass(frozen=True)
class PrimPlacement:
    """One USD primitive in the object's local frame.

    ``scale`` is what the prim's ``xformOp:scale`` is set to, chosen so that a
    unit prim (a 1 m cube, a 1 m-radius sphere) reaches the extents the MuJoCo
    geom had. Driving everything through scale rather than through each prim
    type's own size attribute means one code path updates any shape at runtime.
    """

    prim_type: str
    scale: tuple[float, float, float]
    translate: tuple[float, float, float]
    orient_wxyz: tuple[float, float, float, float]
    # The extent the shape actually occupies, kept for the parity tests.
    half_extents: tuple[float, float, float]


def _quat_from_euler_y(angle: float) -> tuple[float, float, float, float]:
    """The catalogue only ever rotates about local y, by 0 or pi/2."""
    return (math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0)


def _aabb_half_extents(geom_type: str, size: tuple[float, ...],
                       euler: tuple[float, float, float]) -> tuple[float, float, float]:
    """Half-extents of the geom's axis-aligned bounding box after its rotation.

    Only the catalogue's own rotations (identity, or pi/2 about y) occur, so the
    axis swap is exact rather than a bound.
    """
    if geom_type == "box":
        hx, hy, hz = size[0], size[1], size[2]
    elif geom_type == "sphere":
        hx = hy = hz = size[0]
    elif geom_type == "ellipsoid":
        hx, hy, hz = size[0], size[1], size[2]
    elif geom_type == "cylinder":
        r, half_h = size[0], size[1]
        hx, hy, hz = r, r, half_h
    elif geom_type == "capsule":
        r, half_l = size[0], size[1]
        hx, hy, hz = r, r, half_l + r
    else:  # pragma: no cover - the catalogue is closed
        raise ValueError(f"unknown geom type {geom_type!r}")

    if abs(euler[1] - math.pi / 2.0) < 1e-9:
        hx, hz = hz, hx  # the pi/2 rotation about y swaps the x and z extents
    return (float(hx), float(hy), float(hz))


def prim_from_geom(geom) -> PrimPlacement:
    """Translate one :class:`~simgrasp.objects.GeomSpec` into a USD placement."""
    prim_type = _MUJOCO_TO_USD.get(geom.type)
    if prim_type is None:  # pragma: no cover - the catalogue is closed
        raise ValueError(f"no USD primitive for MuJoCo geom type {geom.type!r}")

    size = tuple(float(s) for s in geom.size)
    if geom.type == "box":
        # A unit USD Cube has size 1, i.e. half-extent 0.5, so a scale of
        # 2 * half_extent reaches the MuJoCo extents.
        scale = (2.0 * size[0], 2.0 * size[1], 2.0 * size[2])
    elif geom.type == "sphere":
        # A unit Sphere has radius 1.
        scale = (size[0], size[0], size[0])
    elif geom.type == "ellipsoid":
        scale = (size[0], size[1], size[2])
    elif geom.type == "cylinder":
        # A unit Cylinder has radius 1 and height 1, so z scales by the FULL
        # height while x and y scale by the radius.
        scale = (size[0], size[0], 2.0 * size[1])
    elif geom.type == "capsule":
        # A unit Capsule has radius 1 and a cylindrical section of height 1. The
        # caps scale with the radius, so the z scale is the cylindrical section
        # only: 2 * half_length, matching MuJoCo's exclusion of the caps.
        scale = (size[0], size[0], 2.0 * size[1])
    else:  # pragma: no cover
        raise ValueError(geom.type)

    return PrimPlacement(
        prim_type=prim_type,
        scale=tuple(float(s) for s in scale),
        translate=tuple(float(p) for p in geom.pos),
        orient_wxyz=_quat_from_euler_y(float(geom.euler[1])),
        half_extents=_aabb_half_extents(geom.type, size, geom.euler),
    )


def prims_for_spec(spec: ObjectSpec) -> list[PrimPlacement]:
    """Every primitive making up ``spec``, in catalogue order."""
    if len(spec.geoms) > MAX_PRIMS:  # pragma: no cover - the catalogue is closed
        raise ValueError(
            f"{spec.category} has {len(spec.geoms)} geoms but object bodies are built "
            f"with {MAX_PRIMS} slots")
    return [prim_from_geom(g) for g in spec.geoms]


def spec_top_z(spec: ObjectSpec) -> float:
    """Height of the object's highest point above the table, from the prims alone.

    Recomputed from the translated geometry rather than read off
    ``spec.top_z`` so that a translation error shows up as a mismatch instead of
    being papered over by the catalogue's own bookkeeping.
    """
    prims = prims_for_spec(spec)
    return max(p.translate[2] + p.half_extents[2] for p in prims)


def spec_footprint_radius(spec: ObjectSpec) -> float:
    """Largest horizontal reach from the body origin, from the prims alone."""
    prims = prims_for_spec(spec)
    return max(
        float(np.hypot(abs(p.translate[0]) + p.half_extents[0],
                       abs(p.translate[1]) + p.half_extents[1]))
        for p in prims
    )


def inactive_scale() -> tuple[float, float, float]:
    """Scale applied to a slot's unused prims.

    A slot carries one prim of every type and activates one. The unused ones are
    shrunk rather than deleted: PhysX builds its collision shapes when the scene
    is created, so a prim that was not there at build time cannot be introduced
    later without rebuilding, and rebuilding per episode is exactly the cost this
    port exists to avoid. They are also parked below the table so a residual
    contact cannot touch the object.
    """
    return (1e-5, 1e-5, 1e-5)


INACTIVE_TRANSLATE = (0.0, 0.0, -10.0)


def _geom_volume(geom_type: str, size: tuple[float, ...]) -> float:
    """Volume of one MuJoCo primitive, in cubic metres."""
    if geom_type == "box":
        return 8.0 * size[0] * size[1] * size[2]
    if geom_type == "sphere":
        return 4.0 / 3.0 * math.pi * size[0] ** 3
    if geom_type == "ellipsoid":
        return 4.0 / 3.0 * math.pi * size[0] * size[1] * size[2]
    if geom_type == "cylinder":
        return math.pi * size[0] ** 2 * (2.0 * size[1])
    if geom_type == "capsule":
        # Cylindrical section plus the two hemispherical caps, which together
        # make one sphere of the same radius.
        return math.pi * size[0] ** 2 * (2.0 * size[1]) + 4.0 / 3.0 * math.pi * size[0] ** 3
    raise ValueError(f"unknown geom type {geom_type!r}")  # pragma: no cover


def spec_volume(spec: ObjectSpec) -> float:
    """Total volume of the object's geoms.

    Overlapping geoms are counted twice, which is not an oversight: MuJoCo
    computes a body's mass as the sum of its geoms' independently-computed
    masses and does the same double counting. Matching the engine matters more
    here than matching the true solid, because the object's mass is what decides
    whether an off-centre grasp rotates out of the jaws during the lift.
    """
    return float(sum(_geom_volume(g.type, tuple(float(s) for s in g.size)) for g in spec.geoms))


def spec_mass(spec: ObjectSpec) -> float:
    """Mass in kilograms, from the catalogue's sampled density."""
    return float(spec.density * spec_volume(spec))
