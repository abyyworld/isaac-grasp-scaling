"""The experiment: success rate against training set size.

The question is whether the original study's orientation failure was a data
problem. The answer is a curve, not an assertion, so this module trains the
**same architecture with the same training loop** at a series of dataset sizes
and evaluates every point on the **same held-out set** against the **same
heuristic control**.

What is held fixed
------------------
Everything except the number of training scenes. The model, the loss, the
optimiser schedule, the augmentation, the number of epochs, the evaluation
scenes and the baseline all come from ``simgrasp`` at a pinned commit. The only
thing that varies along the curve is ``TrainConfig.limit``, which caps how many
training episodes the loader is allowed to use.

Why the control is re-run rather than quoted
--------------------------------------------
The heuristic baseline does not depend on training data, so its success rate
should be identical to the original study's. Re-running it is how that is
checked. If it comes out different, something in the environment has moved and
every other number in this project is suspect, so the control is the first thing
to read in the output and not a formality.

Resumability
------------
Each point writes ``point.json`` when it finishes and is skipped on a re-run.
A curve is hours of compute; losing it to a disconnect would be a bad reason to
lose it.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from simgrasp.evaluation import EVAL_EPISODE_OFFSET, evaluate_policy, save_summary
from simgrasp.training import TrainConfig, train

from .angle import measure_angle_error
from .parity import manifest, require_parity


@dataclass
class ScalingConfig:
    """One scaling experiment."""

    data: str
    out: str = "results/scaling"
    # Training scenes per point. Sample count is scenes times angles-per-scene,
    # and the actual number used is recorded per point rather than assumed.
    sizes: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    epochs: int = 12
    batch_size: int = 32
    input_size: int | None = None
    pretrained: bool = False
    train_split: str = "seen"
    workers: int = 2
    device: str | None = None
    seed: int = 0
    # Evaluation. 200 episodes per split is what the original study reported, so
    # the confidence intervals are directly comparable.
    eval_episodes: int = 200
    eval_workers: int = 4
    eval_offset: int = EVAL_EPISODE_OFFSET
    # Orientation measurement. Episodes per category, so the total is nine times
    # this before the determinacy filter.
    angle_episodes: int = 10
    # The control. Re-run once, not per point, because it has no training data.
    controls: tuple[str, ...] = ("heuristic",)
    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _train_one(cfg: ScalingConfig, size: int, run_dir: Path) -> dict[str, Any]:
    train_cfg = TrainConfig(
        data=cfg.data,
        out=str(run_dir),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        input_size=cfg.input_size,
        pretrained=cfg.pretrained,
        train_split=cfg.train_split,
        device=cfg.device,
        seed=cfg.seed,
        limit=size,
    )
    return train(train_cfg)


def _evaluate_one(cfg: ScalingConfig, checkpoint: Path, run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("seen", "unseen"):
        summary = evaluate_policy(
            "cnn", n_episodes=cfg.eval_episodes, split=split, workers=cfg.eval_workers,
            base_seed=cfg.seed, episode_offset=cfg.eval_offset,
            policy_kwargs={"checkpoint": str(checkpoint)}, progress=False)
        save_summary(summary, run_dir / f"eval_{split}.json", keep_records=False)
        out[split] = {"rate": summary["success_rate"], "ci95": summary["ci95"],
                      "n": summary["n"], "by_category": summary["by_category"],
                      "failure_reasons": summary["failure_reasons"]}
    return out


def run_point(cfg: ScalingConfig, size: int) -> dict[str, Any]:
    """Train, evaluate and measure orientation at one training-set size."""
    run_dir = Path(cfg.out) / f"n{size:06d}"
    point_file = run_dir / "point.json"
    if point_file.exists():
        print(f"[scaling] n={size}: already done, skipping", flush=True)
        return json.loads(point_file.read_text())

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[scaling] === training scenes = {size} ===", flush=True)
    started = time.perf_counter()
    summary = _train_one(cfg, size, run_dir)

    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        raise RuntimeError(
            f"training at n={size} produced no checkpoint. Validation AP was NaN for "
            "every epoch, which happens when the validation split contains only one "
            "class. Raise the size or the validation fraction.")

    evaluation = _evaluate_one(cfg, checkpoint, run_dir)
    angle = measure_angle_error(checkpoint, episodes=cfg.angle_episodes,
                                device=cfg.device or "cpu")

    point = {
        "train_scenes": size,
        # The number of labelled grasps actually used, which is what the curve's
        # x axis means. Scenes times angles-per-scene is an upper bound: unstable
        # episodes are dropped at collection time.
        "train_samples": int(summary["n_train"]),
        "epochs": cfg.epochs,
        "best_val_ap": summary["best"]["ap"],
        "best_epoch": summary["best"]["epoch"],
        "train_seconds": summary["elapsed_seconds"],
        "point_seconds": time.perf_counter() - started,
        "device": summary["device"],
        "seen": evaluation["seen"],
        "unseen": evaluation["unseen"],
        "generalisation_gap_pp": 100.0 * (evaluation["seen"]["rate"]
                                          - evaluation["unseen"]["rate"]),
        "angle": angle,
    }
    point_file.write_text(json.dumps(point, indent=2, default=float))
    print(f"[scaling] n={size}: seen {evaluation['seen']['rate']:.1%}, "
          f"held-out {evaluation['unseen']['rate']:.1%}, "
          f"angle error {angle['angle_error_deg']:.1f} deg", flush=True)
    return point


def run_controls(cfg: ScalingConfig) -> dict[str, Any]:
    """Re-run the training-free baselines on the same evaluation scenes."""
    out_dir = Path(cfg.out) / "controls"
    control_file = out_dir / "controls.json"
    if control_file.exists():
        return json.loads(control_file.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    controls: dict[str, Any] = {}
    for name in cfg.controls:
        controls[name] = {}
        for split in ("seen", "unseen"):
            print(f"[scaling] control {name} / {split}", flush=True)
            summary = evaluate_policy(
                name, n_episodes=cfg.eval_episodes, split=split, workers=cfg.eval_workers,
                base_seed=cfg.seed, episode_offset=cfg.eval_offset, progress=False)
            save_summary(summary, out_dir / f"{name}_{split}.json", keep_records=False)
            controls[name][split] = {"rate": summary["success_rate"],
                                     "ci95": summary["ci95"], "n": summary["n"],
                                     "by_category": summary["by_category"]}
    control_file.write_text(json.dumps(controls, indent=2, default=float))
    return controls


def run_scaling(cfg: ScalingConfig) -> dict[str, Any]:
    """Run the whole curve, resuming any points already finished."""
    require_parity()
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_meta = json.loads((Path(cfg.data) / "dataset_meta.json").read_text())
    controls = run_controls(cfg)
    points = [run_point(cfg, size) for size in cfg.sizes]

    result = {
        "config": asdict(cfg),
        "dataset": {
            "path": cfg.data,
            "backend": dataset_meta.get("backend", "unknown"),
            "n_scenes": dataset_meta.get("n_scenes", dataset_meta.get("n_episodes")),
            "angles_per_scene": dataset_meta.get("angles_per_scene"),
            "image_size": dataset_meta.get("image_size"),
            "stats": dataset_meta.get("stats", {}),
        },
        "controls": controls,
        "points": points,
        "parity": manifest(),
    }
    result["reading"] = read_curve(result)
    (out_dir / "scaling.json").write_text(json.dumps(result, indent=2, default=float))
    write_csv(result, out_dir / "scaling.csv")
    return result


def write_csv(result: dict[str, Any], path: Path) -> None:
    """The curve as a flat table, so it can be read without parsing JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["train_scenes", "train_samples", "epochs", "best_val_ap",
               "seen_rate", "seen_ci_lo", "seen_ci_hi",
               "unseen_rate", "unseen_ci_lo", "unseen_ci_hi",
               "generalisation_gap_pp", "angle_error_deg", "angle_error_deg_seen",
               "angle_error_deg_heldout", "bin_spread", "train_seconds"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for point in result["points"]:
            writer.writerow([
                point["train_scenes"], point["train_samples"], point["epochs"],
                point["best_val_ap"],
                point["seen"]["rate"], point["seen"]["ci95"][0], point["seen"]["ci95"][1],
                point["unseen"]["rate"], point["unseen"]["ci95"][0], point["unseen"]["ci95"][1],
                point["generalisation_gap_pp"],
                point["angle"]["angle_error_deg"], point["angle"]["angle_error_deg_seen"],
                point["angle"]["angle_error_deg_heldout"], point["angle"]["bin_spread"],
                point["train_seconds"],
            ])


# --------------------------------------------------------------------------- #
# Reading the curve.
# --------------------------------------------------------------------------- #


def fit_log_trend(samples: list[int], rates: list[float]) -> dict[str, float]:
    """Least-squares fit of ``rate = a + b * log2(samples)``.

    ``b`` is the slope in success-rate points per doubling of the training set,
    and it is a measured property of the points that were run. It is reported
    because "the curve is still rising" is an adjective and this is a number.

    ``r_squared`` is reported next to it so the slope is not read as more solid
    than the points supporting it. With five points and a 95% interval of about
    seven points on each, a low value here means the slope is a summary of noise.
    """
    import numpy as np

    if len(samples) < 2:
        return {"slope_per_doubling_pp": float("nan"), "intercept_pp": float("nan"),
                "r_squared": float("nan"), "n_points": len(samples)}

    x = np.log2(np.asarray(samples, dtype=float))
    y = 100.0 * np.asarray(rates, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_residual = float(np.sum((y - predicted) ** 2))
    ss_total = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope_per_doubling_pp": float(slope),
        "intercept_pp": float(intercept),
        "r_squared": 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan"),
        "n_points": len(samples),
    }


def samples_to_reach(trend: dict[str, float], target_rate: float) -> float:
    """Training set size at which the fitted trend would reach ``target_rate``.

    **This is an extrapolation, not a measurement.** Projecting a line fitted
    over one decade of data out to two or three decades assumes the trend holds
    where nothing was measured, and success rates are bounded above so it cannot
    hold indefinitely. It is reported because it converts "would need a lot more
    data" into a quantity someone can decide about, and it should never be
    quoted without the word extrapolation attached.

    Returns ``inf`` when the fitted slope is flat or negative.
    """
    slope = trend["slope_per_doubling_pp"]
    if not slope or slope <= 0:
        return float("inf")
    doublings = (100.0 * target_rate - trend["intercept_pp"]) / slope
    return float(2.0 ** doublings)


def read_curve(result: dict[str, Any]) -> dict[str, Any]:
    """Fitted trends for the curve, and what they project about the control."""
    points = sorted(result["points"], key=lambda p: p["train_samples"])
    samples = [p["train_samples"] for p in points]
    trends = {
        "held_out_success": fit_log_trend(samples, [p["unseen"]["rate"] for p in points]),
        "seen_success": fit_log_trend(samples, [p["seen"]["rate"] for p in points]),
        "held_out_angle_error": fit_log_trend(
            samples, [p["angle"]["angle_error_deg_heldout"] / 100.0 for p in points]),
    }
    out: dict[str, Any] = {"trends": trends, "measured_range": [samples[0], samples[-1]]}

    control = result.get("controls", {}).get("heuristic", {}).get("unseen")
    if control:
        projected = samples_to_reach(trends["held_out_success"], control["rate"])
        out["extrapolation"] = {
            "note": ("Extrapolated from a log-linear fit over the measured range. "
                     "Not a measurement, and success rates are bounded above so the "
                     "fit cannot hold indefinitely."),
            "target": "heuristic control, held-out",
            "target_rate": control["rate"],
            "projected_samples": projected,
            "times_the_measured_maximum": (projected / samples[-1]
                                           if projected != float("inf") else None),
        }
    return out
