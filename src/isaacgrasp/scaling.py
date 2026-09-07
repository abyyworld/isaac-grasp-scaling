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
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from simgrasp.evaluation import EVAL_EPISODE_OFFSET, evaluate_policy, save_summary
from simgrasp.training import TrainConfig, train

from .angle import measure_angle_error
from .parity import manifest, require_parity


@contextmanager
def single_threaded_torch():
    """Pin torch to one thread per process for the duration of an evaluation.

    Evaluation runs one MuJoCo environment per worker process, and each worker
    imports torch, which by default sizes its thread pool to the whole machine.
    Four workers each claiming four cores is 16 threads fighting over 4, and the
    contention costs more than the parallelism buys: measured at 1.12 seconds per
    episode against 0.25 with one thread per worker, a 4.5x difference, for
    byte-identical results.

    The variables are read by OpenMP and MKL at import, and evaluation workers are
    spawned rather than forked, so setting them here reaches the children.

    Training is deliberately left alone: there the batch is large enough that
    intra-op parallelism in one process is worth having.
    """
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
    previous = {k: os.environ.get(k) for k in keys}
    for key in keys:
        os.environ[key] = "1"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
    # Scene seed. Fixes which objects appear in every evaluation and every angle
    # measurement, so all points are scored on identical scenes.
    seed: int = 0
    # Training seed, deliberately separate. Changing it re-runs the training
    # pipeline (initialisation, data order, augmentation draws, and the
    # train/validation split) without touching the evaluation scenes. Varying the
    # two together would confound run-to-run variance with a different test set,
    # which is exactly the measurement this separation exists to make possible.
    train_seed: int | None = None
    # Evaluation. 200 episodes per split is what the original study reported, so
    # the confidence intervals are directly comparable.
    eval_episodes: int = 200
    eval_workers: int = 4
    eval_offset: int = EVAL_EPISODE_OFFSET
    # Orientation measurement. Episodes per category, so the total is nine times
    # this before the determinacy filter removes the rotationally symmetric ones.
    #
    # 40 rather than the upstream script's 10. At 10, the filter leaves 15 scored
    # samples on the seen split, and the first pass of this curve produced a seen
    # orientation error that wandered by three degrees between adjacent points on
    # that basis alone. Orientation is the question this project exists to answer,
    # so it is not the number to economise on: 40 gives 59 seen and 167 held-out,
    # at roughly 570 seconds per point on a CPU.
    angle_episodes: int = 40
    # The control. Re-run once, not per point, because it has no training data.
    controls: tuple[str, ...] = ("heuristic",)
    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def effective_train_seed(cfg: ScalingConfig) -> int:
    """The training seed, defaulting to the scene seed when unset."""
    return cfg.seed if cfg.train_seed is None else cfg.train_seed


def point_dir(cfg: ScalingConfig, size: int) -> Path:
    """Where a point's artefacts live.

    Replicates get their own directory so a seed sweep never overwrites the
    published curve, and so the seed is visible in the path rather than only
    inside the JSON.
    """
    seed = effective_train_seed(cfg)
    name = f"n{size:06d}" if seed == cfg.seed else f"n{size:06d}_seed{seed}"
    return Path(cfg.out) / name


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
        seed=effective_train_seed(cfg),
        limit=size,
    )
    return train(train_cfg)


def _evaluate_one(cfg: ScalingConfig, checkpoint: Path, run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("seen", "unseen"):
        with single_threaded_torch():
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
    run_dir = point_dir(cfg, size)
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
        "train_seed": effective_train_seed(cfg),
        "scene_seed": cfg.seed,
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
            with single_threaded_torch():
                summary = evaluate_policy(
                    name, n_episodes=cfg.eval_episodes, split=split,
                    workers=cfg.eval_workers, base_seed=cfg.seed,
                    episode_offset=cfg.eval_offset, progress=False)
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
            # A dataset with no backend key was written by simgrasp's own
            # collector, which is the MuJoCo environment by construction. The
            # batched collector here records the key explicitly.
            "backend": dataset_meta.get("backend", "mujoco"),
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
    # A seed sweep is not the curve. Writing it to scaling.json would overwrite
    # the published result with a partial one that merely looks like it.
    seed = effective_train_seed(cfg)
    stem = "scaling" if seed == cfg.seed else f"scaling_seed{seed}"
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2, default=float))
    write_csv(result, out_dir / f"{stem}.csv")
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
    residuals = y - predicted
    ss_residual = float(np.sum(residuals ** 2))
    ss_total = float(np.sum((y - y.mean()) ** 2))

    # Standard error of the slope, and a 95% interval from the t distribution.
    #
    # This is what turns "the trend did not resolve" into a bound. A slope whose
    # interval is [-0.4, +2.0] says the data are consistent with flat AND rule
    # out anything steeper than 2 points per doubling, which is a far more useful
    # statement than an r-squared alone, and it is the one this study can
    # actually support.
    degrees_of_freedom = len(samples) - 2
    slope_stderr = float("nan")
    slope_ci: list[float] = [float("nan"), float("nan")]
    if degrees_of_freedom > 0:
        variance_x = float(np.sum((x - x.mean()) ** 2))
        if variance_x > 0:
            slope_stderr = float(np.sqrt(ss_residual / degrees_of_freedom / variance_x))
            slope_ci = [float(slope - _t95(degrees_of_freedom) * slope_stderr),
                        float(slope + _t95(degrees_of_freedom) * slope_stderr)]

    residual_sd = (float(np.sqrt(ss_residual / degrees_of_freedom))
                   if degrees_of_freedom > 0 else float("nan"))
    return {
        "slope_per_doubling_pp": float(slope),
        "slope_stderr_pp": slope_stderr,
        "slope_ci95_pp": slope_ci,
        "intercept_pp": float(intercept),
        "residual_sd_pp": residual_sd,
        "r_squared": 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan"),
        "n_points": len(samples),
        "degrees_of_freedom": degrees_of_freedom,
    }


def decompose_scatter(points: list[dict[str, Any]], trend: dict[str, float],
                      split: str = "unseen") -> dict[str, float]:
    """Split the points' scatter about the fit into evaluation and training noise.

    Two independent things move a point off the line. The evaluation is a finite
    sample of scenes, and its contribution is known exactly from the binomial:
    ``sqrt(p(1-p)/n)``. Whatever is left over is the training run itself, which
    at a fixed dataset size still varies with initialisation, data order and
    augmentation draws.

    Knowing which one dominates decides what to spend the next hour on. While
    evaluation dominates, more episodes help. Once training dominates, more
    episodes are wasted and the money goes on repeated runs or more sizes. This
    study crossed that line: at 200 episodes per split evaluation dominated, and
    at 1,500 it no longer does.

    Variances add, so the training term is the difference of squares, floored at
    zero because sampling noise can make the estimate come out negative.
    """
    import numpy as np

    residual_sd = trend.get("residual_sd_pp", float("nan"))
    evaluation_variances = []
    for point in points:
        measurement = point.get(split, {})
        # A point written before the episode count was recorded contributes
        # nothing rather than crashing the whole reading.
        if "rate" not in measurement or not measurement.get("n"):
            continue
        rate = float(measurement["rate"])
        n = int(measurement["n"])
        evaluation_variances.append(1e4 * rate * (1.0 - rate) / n)
    evaluation_sd = float(np.sqrt(np.mean(evaluation_variances))) if evaluation_variances         else float("nan")

    training_variance = residual_sd ** 2 - evaluation_sd ** 2
    training_sd = float(np.sqrt(training_variance)) if training_variance > 0 else 0.0
    dominant = ("evaluation" if evaluation_sd > training_sd else "training")
    return {
        "residual_sd_pp": residual_sd,
        "evaluation_sd_pp": evaluation_sd,
        "training_sd_pp": training_sd,
        "dominant_noise_source": dominant,
        "note": (
            "Evaluation noise is the binomial term for the episode count used. "
            "Training noise is what is left after removing it, and covers "
            "initialisation, data order and augmentation draws at a fixed dataset "
            "size. More evaluation episodes only help while evaluation dominates."
        ),
    }


# Two-sided 95% critical values of Student's t, by degrees of freedom. Tabulated
# rather than pulled from scipy: this is the only statistic the package needs and
# scipy is a heavy dependency to add for one lookup. Values beyond 30 are close
# enough to the normal limit that 1.96 is used.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
       15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def _t95(degrees_of_freedom: int) -> float:
    return T95.get(degrees_of_freedom, 1.96)


# Below this, the fitted slope is not distinguishable from flat given the scatter,
# and inverting it produces a number with no meaning. Five points each carrying a
# 95% interval of about seven points can easily fit a slope of the wrong sign.
MIN_R_SQUARED_FOR_PROJECTION = 0.5


def samples_to_reach(trend: dict[str, float], target_rate: float) -> float:
    """Training set size at which the fitted trend would reach ``target_rate``.

    **This is an extrapolation, not a measurement.** Projecting a line fitted
    over one decade of data out to two or three decades assumes the trend holds
    where nothing was measured, and success rates are bounded above so it cannot
    hold indefinitely. It is reported because it converts "would need a lot more
    data" into a quantity someone can decide about, and it should never be
    quoted without the word extrapolation attached.

    Returns ``inf`` when the fitted slope is flat, negative, or too poorly
    supported to invert. A slope that explains almost none of the scatter is
    noise, and extrapolating noise across ten orders of magnitude produces a
    confident-looking number that means nothing at all.
    """
    slope = trend["slope_per_doubling_pp"]
    r_squared = trend.get("r_squared")
    if not slope or slope <= 0:
        return float("inf")
    if r_squared is None or r_squared != r_squared or r_squared < MIN_R_SQUARED_FOR_PROJECTION:
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
    out: dict[str, Any] = {
        "trends": trends,
        "measured_range": [samples[0], samples[-1]],
        "scatter": decompose_scatter(points, trends["held_out_success"], "unseen"),
    }

    control = result.get("controls", {}).get("heuristic", {}).get("unseen")
    if control:
        trend = trends["held_out_success"]
        projected = samples_to_reach(trend, control["rate"])
        reachable = projected != float("inf")
        out["extrapolation"] = {
            "target": "heuristic control, held-out",
            "target_rate": control["rate"],
            "projected_samples": projected if reachable else None,
            "times_the_measured_maximum": projected / samples[-1] if reachable else None,
            "note": (
                "Extrapolated from a log-linear fit over the measured range. Not a "
                "measurement, and success rates are bounded above so the fit cannot "
                "hold indefinitely."
                if reachable else
                f"No projection. The fitted held-out slope explains "
                f"{trend['r_squared']:.0%} of the scatter over {trend['n_points']} "
                f"points, which is not enough to invert: within this range the trend "
                f"is not distinguishable from flat. That is a statement about what "
                f"this arm can resolve, not evidence that the curve is flat."
            ),
        }
    return out

# --------------------------------------------------------------------------- #
# Repeated runs at the same size.
# --------------------------------------------------------------------------- #


def measure_seed_variance(points: list[dict[str, Any]], split: str = "unseen") -> dict[str, Any]:
    """Directly measure how much re-running training moves a point.

    ``decompose_scatter`` infers training variance by subtracting the known
    binomial term from the scatter about a fitted line. That inference assumes
    the true curve is log-linear, so anything curved is charged to training
    noise. Repeated runs at one size make no such assumption: hold the data size
    and the evaluation scenes fixed, change only the training seed, and whatever
    the number does is what re-running does.

    The observed spread still contains evaluation noise, because each replicate
    is scored on a finite sample, so the binomial term is subtracted again here.
    Variances add and the estimate is floored at zero.

    Returns per-size statistics and a pooled estimate. Pooling is by degrees of
    freedom, so a size with three seeds counts twice as much as one with two.
    """
    import numpy as np

    by_size: dict[int, list[dict[str, Any]]] = {}
    for point in points:
        measurement = point.get(split, {})
        if "rate" not in measurement or not measurement.get("n"):
            continue
        by_size.setdefault(int(point["train_samples"]), []).append(point)

    per_size = []
    pooled_numerator = 0.0
    pooled_df = 0
    for samples in sorted(by_size):
        group = by_size[samples]
        if len(group) < 2:
            continue
        rates = np.array([100.0 * p[split]["rate"] for p in group])
        seeds = sorted(p.get("train_seed", 0) for p in group)
        # Sample standard deviation, so ddof=1.
        observed_sd = float(np.std(rates, ddof=1))
        evaluation_variance = float(np.mean([
            1e4 * p[split]["rate"] * (1.0 - p[split]["rate"]) / p[split]["n"]
            for p in group]))
        training_variance = observed_sd ** 2 - evaluation_variance
        per_size.append({
            "train_samples": samples,
            "n_seeds": len(group),
            "seeds": seeds,
            "rates_pp": [float(r) for r in rates],
            "mean_pp": float(np.mean(rates)),
            "range_pp": float(rates.max() - rates.min()),
            "observed_sd_pp": observed_sd,
            "evaluation_sd_pp": float(np.sqrt(evaluation_variance)),
            "training_sd_pp": float(np.sqrt(training_variance)) if training_variance > 0 else 0.0,
        })
        pooled_numerator += (len(group) - 1) * observed_sd ** 2
        pooled_df += len(group) - 1

    out: dict[str, Any] = {"per_size": per_size, "split": split}
    if pooled_df:
        pooled_observed = float(np.sqrt(pooled_numerator / pooled_df))
        mean_evaluation_variance = float(np.mean(
            [row["evaluation_sd_pp"] ** 2 for row in per_size]))
        pooled_training_variance = pooled_observed ** 2 - mean_evaluation_variance
        out["pooled"] = {
            "degrees_of_freedom": pooled_df,
            "observed_sd_pp": pooled_observed,
            "evaluation_sd_pp": float(np.sqrt(mean_evaluation_variance)),
            "training_sd_pp": (float(np.sqrt(pooled_training_variance))
                               if pooled_training_variance > 0 else 0.0),
        }
    return out


def load_points(root: Path | str) -> list[dict[str, Any]]:
    """Every point under ``root``, replicates included."""
    return [json.loads(path.read_text())
            for path in sorted(Path(root).glob("n*/point.json"))]
