"""The USD translation must reproduce the MuJoCo catalogue's geometry.

This is the half of the Isaac port that can be checked without a GPU, and it is
the half most likely to be silently wrong: half-extents against full extents,
a capsule's caps counted twice, a rotation dropped. Each of those produces a
scene that looks plausible in a viewport and grasps at the wrong height.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from simgrasp.objects import ALL_CATEGORIES, sample_object

from isaacgrasp.backends.isaac_objects import (
    MAX_PRIMS,
    PRIM_TYPES,
    prims_for_spec,
    spec_footprint_radius,
    spec_top_z,
)

N_SAMPLES = 40


def _specs(category: str):
    rng = np.random.default_rng(abs(hash(category)) % (2**32))
    return [sample_object(rng, category) for _ in range(N_SAMPLES)]


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_every_category_fits_the_slot_budget(category):
    for spec in _specs(category):
        prims = prims_for_spec(spec)
        assert 1 <= len(prims) <= MAX_PRIMS
        assert all(p.prim_type in PRIM_TYPES for p in prims)


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_top_height_survives_the_translation(category):
    """The object's height above the table is what sets every grasp's z."""
    for spec in _specs(category):
        assert spec_top_z(spec) == pytest.approx(spec.top_z, abs=1e-9), (
            f"{category}: translated top is {spec_top_z(spec):.6f} m, catalogue says "
            f"{spec.top_z:.6f} m")


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_footprint_bound_is_conservative(category):
    """The translated footprint may over-estimate but must never under-estimate.

    The catalogue mixes circumradius and radius between categories, so these are
    not expected to be equal. What matters for scene placement is that the
    translated value never claims the object is smaller than it is.
    """
    for spec in _specs(category):
        assert spec_footprint_radius(spec) >= spec.footprint_radius - 1e-9


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_scale_reproduces_the_mujoco_extents(category):
    """A unit prim under the emitted scale must occupy the MuJoCo geom's extents."""
    for spec in _specs(category):
        for geom, prim in zip(spec.geoms, prims_for_spec(spec), strict=True):
            sx, sy, sz = prim.scale
            if geom.type == "box":
                # Unit Cube spans -0.5..0.5, so the scaled half-extent is s / 2.
                got = (sx / 2.0, sy / 2.0, sz / 2.0)
                assert got == pytest.approx(tuple(geom.size[:3]), abs=1e-12)
            elif geom.type == "sphere":
                assert (sx, sy, sz) == pytest.approx((geom.size[0],) * 3, abs=1e-12)
            elif geom.type == "ellipsoid":
                assert (sx, sy, sz) == pytest.approx(tuple(geom.size[:3]), abs=1e-12)
            elif geom.type == "cylinder":
                # Unit Cylinder: radius 1, height 1 (half-height 0.5).
                assert sx == pytest.approx(geom.size[0], abs=1e-12)
                assert sz / 2.0 == pytest.approx(geom.size[1], abs=1e-12)
            elif geom.type == "capsule":
                # Unit Capsule: radius 1, cylindrical section of height 1. MuJoCo's
                # half_length also excludes the caps, so they correspond directly.
                assert sx == pytest.approx(geom.size[0], abs=1e-12)
                assert sz / 2.0 == pytest.approx(geom.size[1], abs=1e-12)


def test_lying_capsule_is_rotated_about_y():
    """The capsule and the dumbbell shaft lie along local x, not local z."""
    rng = np.random.default_rng(0)
    spec = sample_object(rng, "capsule")
    prim = prims_for_spec(spec)[0]
    w, x, y, z = prim.orient_wxyz
    assert (x, z) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert w == pytest.approx(math.cos(math.pi / 4), abs=1e-9)
    assert y == pytest.approx(math.sin(math.pi / 4), abs=1e-9)
    # After the rotation the long axis is x, so the x half-extent is the largest.
    hx, hy, hz = prim.half_extents
    assert hx > hy and hx > hz


def test_upright_shapes_are_unrotated():
    rng = np.random.default_rng(1)
    for category in ("box", "cylinder", "sphere", "ellipsoid"):
        prim = prims_for_spec(sample_object(rng, category))[0]
        assert prim.orient_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-12)


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_mass_matches_what_mujoco_computes(category):
    """Object mass sets the moment the jaws resist during the lift.

    MuJoCo computes a body's mass as the sum of its geoms' densities times their
    own volumes, double counting any overlap. The port has to reproduce that
    number, not the true solid volume, or the heavier categories grasp
    differently in the two simulators for reasons unrelated to the policy.
    """
    from isaacgrasp.backends.isaac_objects import spec_mass, spec_volume

    for spec in _specs(category):
        assert spec_volume(spec) > 0.0
        assert spec_mass(spec) == pytest.approx(spec.density * spec_volume(spec))
        # Sanity bound: nothing in a 20 to 100 mm catalogue at 300 to 1200 kg/m3
        # should weigh less than a gram or more than a kilogram.
        assert 1e-3 < spec_mass(spec) < 1.0


def test_volume_of_a_known_shape():
    """A capsule is a cylinder plus one sphere of the same radius."""
    import math

    from isaacgrasp.backends.isaac_objects import _geom_volume

    r, half_l = 0.02, 0.05
    expected = math.pi * r**2 * 2 * half_l + 4 / 3 * math.pi * r**3
    assert _geom_volume("capsule", (r, half_l)) == pytest.approx(expected)
    assert _geom_volume("box", (1.0, 2.0, 3.0)) == pytest.approx(48.0)
    assert _geom_volume("sphere", (1.0,)) == pytest.approx(4 / 3 * math.pi)
