"""What must stay identical to the original study, and how that is enforced.

The headline of this project is a comparison against numbers produced by
``simgrasp``. A comparison is only worth publishing if the things being held
fixed are actually fixed, and "I was careful" is not a mechanism. So:

**Everything simulator-independent is imported, never reimplemented.** The
network, the training loop, the loss, the dataset format, the heuristic
baseline, the success criterion's thresholds and the evaluation statistics all
come from ``simgrasp``, pinned to an exact commit in ``pyproject.toml``. There
is no copy of them in this repository to drift.

**Everything simulator-dependent is checked numerically.** The Isaac Lab port
has to rebuild the scene, the robot and the object catalogue in a different
engine. Those cannot be imported, so this module records the values the port
must reproduce and provides :func:`check_parity`, which fails loudly if the
upstream definitions move underneath us.

The constants below are deliberately duplicated from ``simgrasp``: the point is
to notice when the two disagree. They are not an alternative source of truth,
and nothing in this package should read them instead of reading ``simgrasp``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Expected values, as measured from the pinned simgrasp commit.
# --------------------------------------------------------------------------- #

EXPECTED: dict[str, Any] = {
    # Scene geometry (simgrasp.scene)
    "TABLE_HEIGHT": 0.40,
    "WORKSPACE_X": (0.40, 0.68),
    "WORKSPACE_Y": (-0.20, 0.20),
    "CAMERA_HEIGHT": 0.55,
    "CAMERA_X": 0.54,
    "CAMERA_FOVY_DEG": 48.0,
    # Success criterion (simgrasp.env). Changing any of these changes what
    # "success" means and makes every number incomparable.
    "LIFT_SUCCESS_THRESHOLD": 0.08,
    "LIFT_DISTANCE": 0.20,
    "APPROACH_HEIGHT": 0.12,
    "SETTLE_TIME": 0.6,
    # Grasp geometry (simgrasp.grasp)
    "GRASP_DEPTH": 0.012,
    "MIN_TCP_HEIGHT": 0.011,
    "MAX_APPROACH_DEPTH": 0.035,
    "GRIP_BIAS": 0.005,
    "PRESHAPE_CLEARANCE": 0.022,
    # Gripper (simgrasp.objects)
    "MAX_GRIPPER_WIDTH": 0.08,
    "SAFE_GRASP_WIDTH": 0.068,
    # Label space (simgrasp.models)
    "ANGLE_BINS": 12,
    # Object catalogue (simgrasp.objects)
    "SEEN_CATEGORIES": ("box", "cylinder", "capsule", "sphere"),
    "UNSEEN_CATEGORIES": ("ellipsoid", "l_shape", "t_shape", "mug", "dumbbell"),
    # Evaluation (simgrasp.evaluation). Held-out scenes start here, far outside
    # any plausible collection range, so no evaluated scene was ever trained on.
    "EVAL_EPISODE_OFFSET": 1_000_000,
    # Height encoding (simgrasp.data.writer)
    "HEIGHT_SCALE": 10_000.0,
}


@dataclass
class ParityReport:
    """Outcome of comparing this package's expectations against ``simgrasp``."""

    matched: dict[str, Any] = field(default_factory=dict)
    mismatched: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatched and not self.missing

    def describe(self) -> str:
        if self.ok:
            return f"parity OK: {len(self.matched)} constants match the pinned simgrasp commit"
        lines = ["PARITY FAILURE: this port no longer matches the study it compares against."]
        for name, (expected, actual) in sorted(self.mismatched.items()):
            lines.append(f"  {name}: expected {expected!r}, simgrasp has {actual!r}")
        for name in sorted(self.missing):
            lines.append(f"  {name}: not found in simgrasp")
        lines.append("")
        lines.append("Either the simgrasp pin in pyproject.toml moved, or the upstream "
                     "definition changed. Do not 'fix' this by editing EXPECTED unless the "
                     "published numbers are being regenerated from scratch.")
        return "\n".join(lines)


def _upstream_values() -> dict[str, Any]:
    """Read every checked constant out of ``simgrasp``.

    Imported lazily and module by module so the failure message names the module
    that could not be imported, which is the common case on a fresh machine.
    """
    from simgrasp import env, evaluation, grasp, models, objects, scene
    from simgrasp.data import writer

    sources = (scene, env, grasp, objects, models, evaluation, writer)
    found: dict[str, Any] = {}
    for name in EXPECTED:
        for module in sources:
            if hasattr(module, name):
                found[name] = getattr(module, name)
                break
    return found


def _equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, tuple) and isinstance(actual, (tuple, list)):
        return len(expected) == len(actual) and all(_equal(a, b) for a, b in zip(expected, actual, strict=True))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and abs(float(actual) - expected) < 1e-12
    return bool(expected == actual)


def check_parity() -> ParityReport:
    """Compare :data:`EXPECTED` against the installed ``simgrasp``."""
    upstream = _upstream_values()
    report = ParityReport()
    for name, expected in EXPECTED.items():
        if name not in upstream:
            report.missing.append(name)
        elif _equal(expected, upstream[name]):
            report.matched[name] = upstream[name]
        else:
            report.mismatched[name] = (expected, upstream[name])
    return report


def require_parity() -> ParityReport:
    """Raise unless the port still matches the study it compares against."""
    report = check_parity()
    if not report.ok:
        raise RuntimeError(report.describe())
    return report


def manifest() -> dict[str, Any]:
    """Provenance recorded alongside every dataset and every result file.

    Written into the artefacts so a result can be traced to the exact upstream
    definitions it was produced under, without trusting the README.
    """
    import simgrasp

    report = check_parity()
    return {
        "simgrasp_version": getattr(simgrasp, "__version__", "unknown"),
        "parity_ok": report.ok,
        "constants": {k: list(v) if isinstance(v, tuple) else v
                      for k, v in sorted(report.matched.items())},
        "mismatched": {k: [list(a) if isinstance(a, tuple) else a,
                           list(b) if isinstance(b, tuple) else b]
                       for k, (a, b) in sorted(report.mismatched.items())},
        "missing": sorted(report.missing),
    }
