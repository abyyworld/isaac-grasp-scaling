"""Samples per hour, measured rather than quoted.

The case for the Isaac Lab port is throughput: thousands of environments stepped
together on one GPU against a handful of MuJoCo processes on CPU cores. That
claim is worth exactly as much as the measurement behind it, so the number
reported here is read out of the dataset metadata that the collector wrote while
it ran, not from a benchmark run separately under friendlier conditions.

Both figures come from the same collector driving the same episode sequence, so
the ratio is a property of the simulators and not of two pipelines. What it is
**not** is a like-for-like hardware comparison: a rented RTX GPU against four
CPU cores says as much about the hardware as about the software. The hardware is
recorded alongside the number for exactly that reason, and the honest headline
is samples per hour per machine, which is what decides whether a dataset is
affordable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ThroughputRecord:
    """One collection run's measured rate."""

    backend: str
    hardware: str
    n_samples: int
    n_scenes: int
    elapsed_seconds: float
    angles_per_scene: int
    image_size: int
    batch_size: int = 1
    workers: int = 1

    @property
    def samples_per_hour(self) -> float:
        return 3600.0 * self.n_samples / self.elapsed_seconds if self.elapsed_seconds else 0.0

    @property
    def scenes_per_hour(self) -> float:
        return 3600.0 * self.n_scenes / self.elapsed_seconds if self.elapsed_seconds else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "hardware": self.hardware,
            "n_samples": self.n_samples,
            "n_scenes": self.n_scenes,
            "elapsed_seconds": self.elapsed_seconds,
            "elapsed_hours": self.elapsed_seconds / 3600.0,
            "angles_per_scene": self.angles_per_scene,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "workers": self.workers,
            "samples_per_hour": self.samples_per_hour,
            "scenes_per_hour": self.scenes_per_hour,
        }


def record_from_dataset(dataset_dir: Path | str, hardware: str,
                        workers: int = 1, batch_size: int = 1) -> ThroughputRecord:
    """Read a collection run's rate out of the metadata it wrote."""
    meta = json.loads((Path(dataset_dir) / "dataset_meta.json").read_text())
    stats = meta.get("stats", {})
    elapsed = float(stats.get("elapsed_seconds", meta.get("elapsed_seconds", 0.0)))
    return ThroughputRecord(
        backend=meta.get("backend", "mujoco"),
        hardware=hardware,
        n_samples=int(stats.get("n", 0)),
        n_scenes=int(stats.get("n_scenes", meta.get("n_scenes", meta.get("n_episodes", 0)))),
        elapsed_seconds=elapsed,
        angles_per_scene=int(meta.get("angles_per_scene", 1)),
        image_size=int(meta.get("image_size", 224)),
        batch_size=batch_size,
        workers=workers,
    )


def compare(records: list[ThroughputRecord]) -> dict[str, Any]:
    """Rates side by side, with the speedup relative to the slowest run."""
    if not records:
        return {"records": [], "note": "no runs measured"}
    rates = [r.samples_per_hour for r in records]
    baseline = min(r for r in rates if r > 0) if any(rates) else 0.0
    rows = []
    for record in records:
        row = record.as_dict()
        row["speedup_vs_slowest"] = (record.samples_per_hour / baseline) if baseline else None
        rows.append(row)
    return {
        "records": rows,
        "baseline_samples_per_hour": baseline,
        "caveat": (
            "Rates are per machine, not per unit of hardware. The two backends were "
            "measured on different machines because that is how each one is actually "
            "run, so this ratio mixes a software difference with a hardware one. The "
            "hardware is recorded per row."
        ),
    }


def format_table(comparison: dict[str, Any]) -> str:
    rows = comparison.get("records", [])
    if not rows:
        return "no throughput measurements recorded"
    lines = [
        f"  {'backend':<10}{'hardware':<34}{'samples':>10}{'hours':>8}"
        f"{'samples/h':>12}{'speedup':>9}",
        "  " + "-" * 83,
    ]
    for row in rows:
        speedup = row.get("speedup_vs_slowest")
        speedup_text = f"{speedup:.1f}x" if speedup else "-"
        lines.append(
            f"  {row['backend']:<10}{row['hardware'][:33]:<34}{row['n_samples']:>10}"
            f"{row['elapsed_hours']:>8.2f}{row['samples_per_hour']:>12,.0f}{speedup_text:>9}")
    lines.append("")
    lines.append("  " + comparison["caveat"])
    return "\n".join(lines)
