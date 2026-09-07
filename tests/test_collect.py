"""The batched collector must reproduce the original collector exactly.

``isaacgrasp.collect`` exists so that one code path drives both simulators, which
is what makes the samples-per-hour comparison a comparison of simulators rather
than of two separately-tuned pipelines. That argument only holds if the batched
path produces the same dataset as the original when it is driven by the same
simulator, so this test runs both over the same episodes and compares the
output byte for byte: the same images, the same labels, the same outcomes.

A difference here would not look like a bug. It would look like the Isaac
dataset being slightly different from the MuJoCo one, and the scaling curve
would be measuring that difference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from isaacgrasp.collect import collect  # noqa: E402

EPISODES = 8
IMAGE_SIZE = 96
ANGLES = 3
LABEL_KEYS = ("episode", "category", "split", "u", "v", "angle", "width_px", "depth",
              "world_x", "world_y", "world_z", "world_yaw", "world_width",
              "success", "reason", "angle_index", "sampler_mode")


def _reference(out_dir: Path) -> None:
    """Collect the same episodes with the unmodified upstream collector."""
    from simgrasp.collect import collect_worker

    collect_worker(episodes=list(range(EPISODES)), out_dir=str(out_dir), worker_id=0,
                   base_seed=0, image_size=IMAGE_SIZE, split="seen", shard_size=256,
                   store_rgb=True, angles_per_scene=ANGLES, verbose=False)


def _labels(root: Path) -> list[dict]:
    out = []
    for meta_path in sorted(p for p in root.glob("*_meta.json")
                            if p.name != "dataset_meta.json"):
        out.extend(json.loads(meta_path.read_text())["labels"])
    return out


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    root = tmp_path_factory.mktemp("collect")
    reference, batched = root / "reference", root / "batched"
    _reference(reference)
    collect("mujoco", batched, n_scenes=EPISODES, split="seen", base_seed=0,
            image_size=IMAGE_SIZE, angles_per_scene=ANGLES, verbose=False)
    return reference, batched


def test_same_number_of_samples(datasets):
    reference, batched = datasets
    assert len(_labels(reference)) == len(_labels(batched)) > 0


def test_labels_are_identical(datasets):
    reference, batched = datasets
    for expected, got in zip(_labels(reference), _labels(batched), strict=True):
        for key in LABEL_KEYS:
            assert got[key] == pytest.approx(expected[key]) if isinstance(
                expected[key], float) else got[key] == expected[key], (
                f"label field {key!r} differs: {got[key]!r} against {expected[key]!r}")


def test_outcomes_are_identical(datasets):
    """The same grasps executed in the same scenes must succeed and fail alike."""
    reference, batched = datasets
    expected = [lab["success"] for lab in _labels(reference)]
    got = [lab["success"] for lab in _labels(batched)]
    assert got == expected
    assert 0 < sum(expected) < len(expected), "the episodes are not exercising both outcomes"


def test_images_are_identical(datasets):
    reference, batched = datasets
    for name in ("w00_00000_height.npy", "w00_00000_rgb.npy"):
        a = np.load(reference / name)
        b = np.load(batched / name)
        assert a.shape == b.shape
        assert np.array_equal(a, b), f"{name} differs between the two collectors"


def test_metadata_records_parity(datasets):
    _reference_dir, batched = datasets
    meta = json.loads((batched / "dataset_meta.json").read_text())
    assert meta["backend"] == "mujoco"
    assert meta["parity"]["parity_ok"] is True
    assert meta["stats"]["n"] == len(_labels(batched))
    assert meta["stats"]["samples_per_hour"] > 0
