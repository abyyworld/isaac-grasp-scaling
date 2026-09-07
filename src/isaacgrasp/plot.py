"""The scaling curve, drawn so the answer is visible rather than asserted.

Two panels, never two y axes on one panel. Success rate and orientation error
are different measures on different scales, and overlaying them on a shared
frame invites a reader to see a crossing point that does not exist.

Panel A is the headline: does held-out success rise with data, and does it reach
the heuristic control. Panel B is the diagnosis: does orientation error fall,
and does it get below the 45 degrees that random guessing scores. The original
study's numbers are drawn on both so the follow-up can be read against what it
follows up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots 1 to 3 (light surface). Assigned by entity and
# never cycled: held-out is always blue, seen is always orange, the control is
# always aqua, whichever points happen to be present.
HELD_OUT = "#2a78d6"
SEEN = "#eb6834"
CONTROL = "#1baf7a"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedcd6"

# The predecessor study's published held-out numbers, for reference only.
ORIGINAL_CNN_HELD_OUT = 0.584
ORIGINAL_HEURISTIC_HELD_OUT = 0.753
CHANCE_ANGLE_DEG = 45.0


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, which="major", color=GRID, linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_MUTED, labelsize=9)


def _log_x(axis, x: list[int]) -> None:
    """Log x axis ticked at the measured sizes, not at powers of ten.

    Matplotlib's default log minor ticks render as "4 x 10^2 6 x 10^2 ..." and
    run into each other at this figure width. The sizes actually measured are
    the only x values that mean anything here, so they are the ticks, written as
    plain numbers.

    The padding matters too: without it the first and last markers sit on the
    spines, where they collide with the reference-line labels at the left and
    the direct labels at the right.
    """
    axis.set_xscale("log")
    axis.set_xlim(x[0] / 1.6, x[-1] * 1.6)
    axis.set_xticks(x)
    axis.set_xticklabels([f"{value:,}" for value in x])
    axis.tick_params(axis="x", which="minor", length=0)
    axis.set_xticks([], minor=True)


def _label_end(axis, x, y, text, colour, dy: float = 0.0) -> None:
    """Direct label just past the right end of a series.

    Required rather than decorative: the control colour sits below 3:1 contrast
    on this surface, so identity has to be carried by something other than hue.
    ``dy`` nudges labels apart when two series finish close together.
    """
    axis.annotate(text, xy=(x, y), xytext=(7, dy), textcoords="offset points",
                  color=colour, fontsize=9, fontweight="bold",
                  va="center", ha="left", clip_on=False)


def _label_reference(axis, x, y, text, colour, above: bool = True) -> None:
    """Label a horizontal reference line at the left edge.

    At the left, where no series has started yet, rather than at the right,
    where the series labels are. A reference line labelled on top of the marks
    it is a reference for is not a label, it is a collision.
    """
    axis.annotate(text, xy=(x, y), xytext=(2, 5 if above else -12),
                  textcoords="offset points", color=colour, fontsize=8.5,
                  fontweight="bold" if above else "normal", ha="left")


def _separate(a: float, b: float, span: float) -> tuple[float, float]:
    """Vertical nudges in points for two labels that may overlap."""
    if abs(a - b) > 0.06 * span:
        return 0.0, 0.0
    return (7.0, -7.0) if a >= b else (-7.0, 7.0)


def plot_scaling(result: dict[str, Any], out_path: Path | str,
                 title: str | None = None) -> Path:
    """Render the curve to ``out_path``."""
    points = sorted(result["points"], key=lambda p: p["train_samples"])
    if not points:
        raise ValueError("no scaling points to plot")

    x = [p["train_samples"] for p in points]
    seen = [100.0 * p["seen"]["rate"] for p in points]
    held = [100.0 * p["unseen"]["rate"] for p in points]
    seen_lo = [100.0 * p["seen"]["ci95"][0] for p in points]
    seen_hi = [100.0 * p["seen"]["ci95"][1] for p in points]
    held_lo = [100.0 * p["unseen"]["ci95"][0] for p in points]
    held_hi = [100.0 * p["unseen"]["ci95"][1] for p in points]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)
    figure.subplots_adjust(left=0.07, right=0.88, top=0.86, bottom=0.15, wspace=0.28)

    # -- Panel A: success rate --------------------------------------------- #
    _style(left)
    _log_x(left, x)
    left.fill_between(x, seen_lo, seen_hi, color=SEEN, alpha=0.13, linewidth=0)
    left.fill_between(x, held_lo, held_hi, color=HELD_OUT, alpha=0.13, linewidth=0)
    left.plot(x, seen, color=SEEN, linewidth=2, marker="o", markersize=5,
              markeredgecolor=SURFACE, markeredgewidth=1.2, label="CNN, seen categories")
    left.plot(x, held, color=HELD_OUT, linewidth=2, marker="o", markersize=5,
              markeredgecolor=SURFACE, markeredgewidth=1.2, label="CNN, held-out categories")

    control = _control_rate(result, "unseen")
    if control is not None:
        left.axhline(100.0 * control, color=CONTROL, linewidth=2, linestyle="--",
                     label="Heuristic control, held-out")
        _label_reference(left, x[0], 100.0 * control,
                         f"heuristic control {100 * control:.1f}%", CONTROL)

    left.axhline(100.0 * ORIGINAL_CNN_HELD_OUT, color=INK_MUTED, linewidth=1,
                 linestyle=":", alpha=0.8)
    _label_reference(left, x[0], 100.0 * ORIGINAL_CNN_HELD_OUT,
                     f"original study, held-out {100 * ORIGINAL_CNN_HELD_OUT:.1f}%",
                     INK_MUTED, above=False)

    dy_held, dy_seen = _separate(held[-1], seen[-1], 100.0)
    _label_end(left, x[-1], held[-1], f"{held[-1]:.1f}%", HELD_OUT, dy_held)
    _label_end(left, x[-1], seen[-1], f"{seen[-1]:.1f}%", SEEN, dy_seen)
    left.set_xlabel("training set size (labelled grasps, log scale)", color=INK_MUTED, fontsize=9)
    left.set_ylabel("grasp success rate (%)", color=INK_MUTED, fontsize=9)
    left.set_title("Does more data close the gap?", color=INK, fontsize=11,
                   fontweight="bold", loc="left")
    left.set_ylim(0, 100)
    left.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_MUTED)

    # -- Panel B: orientation error ---------------------------------------- #
    _style(right)
    _log_x(right, x)
    angle_seen = [p["angle"]["angle_error_deg_seen"] for p in points]
    angle_held = [p["angle"]["angle_error_deg_heldout"] for p in points]
    right.plot(x, angle_seen, color=SEEN, linewidth=2, marker="o", markersize=5,
               markeredgecolor=SURFACE, markeredgewidth=1.2, label="seen categories")
    right.plot(x, angle_held, color=HELD_OUT, linewidth=2, marker="o", markersize=5,
               markeredgecolor=SURFACE, markeredgewidth=1.2, label="held-out categories")
    right.axhline(CHANCE_ANGLE_DEG, color=INK_MUTED, linewidth=1.2, linestyle="--")
    # Below the line: the curves start above 45 degrees and end below it, so the
    # space under the line at the left is the only reliably empty corner.
    _label_reference(right, x[0], CHANCE_ANGLE_DEG, "random guessing, 45 deg",
                     INK_MUTED, above=False)
    ceiling = max(60.0, max(angle_seen + angle_held) + 8)
    dy_held, dy_seen = _separate(angle_held[-1], angle_seen[-1], ceiling)
    _label_end(right, x[-1], angle_held[-1], f"{angle_held[-1]:.0f} deg", HELD_OUT, dy_held)
    _label_end(right, x[-1], angle_seen[-1], f"{angle_seen[-1]:.0f} deg", SEEN, dy_seen)
    right.set_xlabel("training set size (labelled grasps, log scale)", color=INK_MUTED, fontsize=9)
    right.set_ylabel("mean orientation error (degrees)", color=INK_MUTED, fontsize=9)
    right.set_title("Does orientation improve with data?", color=INK, fontsize=11,
                    fontweight="bold", loc="left")
    right.set_ylim(0, ceiling)
    right.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK_MUTED)

    dataset = result.get("dataset", {})
    backend = {"mujoco": "MuJoCo", "isaac": "Isaac Lab"}.get(
        dataset.get("backend", ""), dataset.get("backend", "unknown"))
    heading = title or (
        f"Grasp success against training set size: {backend}-generated data, "
        f"{result['config']['eval_episodes']} evaluation episodes per split")
    figure.suptitle(heading, color=INK, fontsize=12, fontweight="bold", x=0.07, ha="left")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return out_path


def _control_rate(result: dict[str, Any], split: str) -> float | None:
    controls = result.get("controls", {})
    heuristic = controls.get("heuristic")
    if not heuristic or split not in heuristic:
        return None
    return float(heuristic[split]["rate"])


def plot_from_file(scaling_json: Path | str, out_path: Path | str) -> Path:
    return plot_scaling(json.loads(Path(scaling_json).read_text()), out_path)
