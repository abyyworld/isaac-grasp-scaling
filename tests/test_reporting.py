"""The curve's outputs: the CSV, the figure and the throughput table.

These run on a synthetic result rather than a real one so they stay fast and so
they exercise the edge cases a real run would not reach for hours: a single
point, a missing control, a zero elapsed time.
"""

from __future__ import annotations

import csv
import json

import pytest

from isaacgrasp.plot import plot_scaling
from isaacgrasp.scaling import write_csv
from isaacgrasp.throughput import ThroughputRecord, compare, format_table, record_from_dataset


def _point(scenes: int, samples: int, seen: float, unseen: float, angle: float) -> dict:
    return {
        "train_scenes": scenes, "train_samples": samples, "epochs": 10,
        "best_val_ap": 0.5, "train_seconds": 12.0, "point_seconds": 20.0,
        "seen": {"rate": seen, "ci95": [seen - 0.05, seen + 0.05], "n": 200},
        "unseen": {"rate": unseen, "ci95": [unseen - 0.05, unseen + 0.05], "n": 200},
        "generalisation_gap_pp": 100.0 * (seen - unseen),
        "angle": {"angle_error_deg": angle, "angle_error_deg_seen": angle - 3,
                  "angle_error_deg_heldout": angle + 2, "bin_spread": 0.05},
    }


@pytest.fixture
def result() -> dict:
    return {
        "config": {"eval_episodes": 200},
        "dataset": {"backend": "mujoco", "n_scenes": 6000, "angles_per_scene": 3},
        "controls": {"heuristic": {"seen": {"rate": 0.88, "ci95": [0.8, 0.93], "n": 200},
                                   "unseen": {"rate": 0.75, "ci95": [0.65, 0.83], "n": 200}}},
        "points": [_point(128, 384, 0.55, 0.35, 52.0),
                   _point(512, 1536, 0.70, 0.45, 48.0),
                   _point(2048, 6144, 0.80, 0.55, 44.0)],
    }


def test_csv_has_one_row_per_point(result, tmp_path):
    path = tmp_path / "scaling.csv"
    write_csv(result, path)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 3
    assert [int(r["train_samples"]) for r in rows] == [384, 1536, 6144]
    assert float(rows[-1]["unseen_rate"]) == pytest.approx(0.55)
    assert float(rows[-1]["angle_error_deg"]) == pytest.approx(44.0)


def test_figure_is_written(result, tmp_path):
    path = plot_scaling(result, tmp_path / "curve.png")
    assert path.exists()
    # A PNG signature, so a silently empty file fails rather than passing.
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 10_000


def test_figure_survives_a_missing_control(result, tmp_path):
    result["controls"] = {}
    assert plot_scaling(result, tmp_path / "curve.png").exists()


def test_figure_survives_a_single_point(result, tmp_path):
    result["points"] = result["points"][:1]
    assert plot_scaling(result, tmp_path / "curve.png").exists()


def test_plot_rejects_an_empty_curve(result, tmp_path):
    result["points"] = []
    with pytest.raises(ValueError, match="no scaling points"):
        plot_scaling(result, tmp_path / "curve.png")


def test_throughput_rate_and_speedup():
    slow = ThroughputRecord("mujoco", "4 vCPU", n_samples=18_000, n_scenes=6000,
                            elapsed_seconds=3600.0, angles_per_scene=3, image_size=224,
                            workers=4)
    fast = ThroughputRecord("isaac", "RTX 4090", n_samples=180_000, n_scenes=60_000,
                            elapsed_seconds=3600.0, angles_per_scene=3, image_size=224,
                            batch_size=1024)
    assert slow.samples_per_hour == pytest.approx(18_000)
    comparison = compare([slow, fast])
    assert comparison["records"][1]["speedup_vs_slowest"] == pytest.approx(10.0)
    assert "hardware" in format_table(comparison)
    assert "caveat" in comparison


def test_throughput_handles_a_zero_duration():
    record = ThroughputRecord("mujoco", "x", 0, 0, 0.0, 3, 224)
    assert record.samples_per_hour == 0.0
    assert compare([record])["records"][0]["speedup_vs_slowest"] is None
    assert compare([]) == {"records": [], "note": "no runs measured"}


def test_record_is_read_from_dataset_metadata(tmp_path):
    (tmp_path / "dataset_meta.json").write_text(json.dumps({
        "backend": "mujoco", "angles_per_scene": 3, "image_size": 224,
        "stats": {"n": 1800, "n_scenes": 600, "elapsed_seconds": 360.0},
    }))
    record = record_from_dataset(tmp_path, hardware="4 vCPU", workers=4)
    assert record.n_samples == 1800
    assert record.samples_per_hour == pytest.approx(18_000)
    assert record.workers == 4
