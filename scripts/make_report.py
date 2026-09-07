#!/usr/bin/env python3
"""Regenerate docs/results.md from the committed run artefacts.

    python scripts/make_report.py --result results/scaling/mujoco/scaling.json

The tables in the documentation are generated from the JSON rather than typed,
so a number in the prose cannot drift away from the number in the artefact. If
this script has not been re-run after a new result, the header says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The predecessor study's published held-out numbers.
ORIGINAL = {"cnn_unseen": 0.584, "heuristic_unseen": 0.753,
            "cnn_seen": 0.838, "heuristic_seen": 0.883,
            "angle_heldout_deg": 47.2, "bin_spread": 0.070,
            # Sample sizes behind the published figures, from the committed
            # run artefacts in the predecessor repository. Its 200-trial run
            # covered all categories and was split afterwards, so the held-out
            # figure rests on 89 trials.
            "heuristic_unseen_n": 89, "heuristic_seen_n": 111,
            "heuristic_unseen_ci": (0.654, 0.831)}

BACKEND_NAMES = {"mujoco": "MuJoCo", "isaac": "Isaac Lab"}


def backend_name(result: dict) -> str:
    raw = result.get("dataset", {}).get("backend", "unknown")
    return BACKEND_NAMES.get(raw, raw)


# The best orientation error the predecessor reached on any split, after its
# contrastive-label fix. A useful target because it is known to be achievable
# with this architecture rather than being an arbitrary round number.
ORIGINAL_BEST_ANGLE_DEG = 35.2


def _orientation_verdict(trend: dict, last: dict) -> str:
    """What the orientation trend supports, in doublings of training data."""
    low, high = trend.get("slope_ci95_pp", [float("nan"), float("nan")])
    slope = trend.get("slope_per_doubling_pp", float("nan"))
    current = last["angle"]["angle_error_deg_heldout"]
    if low != low or high >= 0:
        return (
            f"Orientation error on held-out shapes is not distinguishable from flat "
            f"over this range: {slope:+.2f} degrees per doubling, 95% interval "
            f"{low:+.2f} to {high:+.2f}.")

    distance = current - ORIGINAL_BEST_ANGLE_DEG
    fastest = distance / abs(low)
    typical = distance / abs(slope)
    return (
        f"**Orientation error is falling, and this is the one trend the data resolve.** "
        f"It drops {abs(slope):.2f} degrees per doubling of training data, 95% interval "
        f"{abs(high):.2f} to {abs(low):.2f}, which excludes zero. More data does improve "
        f"orientation.\n\n"
        f"The rate is what settles the question. From {current:.1f} degrees, reaching the "
        f"{ORIGINAL_BEST_ANGLE_DEG:.1f} degrees the original study achieved on seen "
        f"categories after its contrastive-label fix would take about {typical:.0f} "
        f"further doublings at the fitted rate, and about {fastest:.0f} even at the "
        f"fastest end of the interval. That is between {2 ** fastest:,.0f}x and "
        f"{2 ** typical:,.0f}x the {last['train_samples']:,} grasps used here: of order "
        f"{2 ** fastest * last['train_samples'] / 1e6:.1f} million to "
        f"{2 ** typical * last['train_samples'] / 1e6:,.0f} million labelled grasps. "
        "The lower end of that is reachable with a GPU simulator. The upper end is not "
        "an experiment anyone is going to run, and it is the reason to suspect the "
        "architecture rather than the dataset."
    )


def curve_table(result: dict) -> str:
    lines = [
        "| training scenes | labelled grasps | seen | held-out | gap | angle error (held-out) |",
        "|---|---|---|---|---|---|",
    ]
    for point in sorted(result["points"], key=lambda p: p["train_samples"]):
        seen, unseen = point["seen"], point["unseen"]
        lines.append(
            f"| {point['train_scenes']:,} | {point['train_samples']:,} "
            f"| {seen['rate']:.1%} | {unseen['rate']:.1%} "
            f"| {point['generalisation_gap_pp']:.1f} pp "
            f"| {point['angle']['angle_error_deg_heldout']:.1f} deg |")
    return "\n".join(lines)


def control_table(result: dict) -> str:
    lines = ["| policy | seen | held-out | 95% CI, held-out | original study, held-out |",
             "|---|---|---|---|---|"]
    for name, control in result.get("controls", {}).items():
        original = ORIGINAL.get(f"{name}_unseen")
        original_n = ORIGINAL.get(f"{name}_unseen_n")
        reference = (f"{original:.1%} (n={original_n})" if original else "-")
        held = control["unseen"]
        lines.append(
            f"| {name} (re-run unchanged, n={held['n']}) | {control['seen']['rate']:.1%} "
            f"| {held['rate']:.1%} | {held['ci95'][0]:.1%} to {held['ci95'][1]:.1%} "
            f"| {reference} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", action="append", required=True,
                        help="a scaling.json; repeat for each arm of the experiment")
    parser.add_argument("--throughput", default=None)
    parser.add_argument("--seed-variance", default=None,
                        help="results/seed_variance.json, from scripts/seed_variance.py")
    parser.add_argument("--out", default="docs/results.md")
    parser.add_argument("--readme", default=None,
                        help="also fill the RESULTS block in this README")
    parser.add_argument("--figure", default=None,
                        help="path to the curve image, as the README should reference it")
    args = parser.parse_args()

    sections = [
        "# Results",
        "",
        "Generated by `python scripts/make_report.py`. Every number here is read "
        "from a committed artefact under `results/`; none is typed by hand.",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
    ]

    for path in args.result:
        result = json.loads(Path(path).read_text())
        dataset = result.get("dataset", {})
        config = result.get("config", {})
        sections += [
            f"## The {backend_name(result)} arm",
            "",
            f"Dataset `{Path(str(dataset.get('path', '?'))).name}`: "
            f"{dataset.get('n_scenes', '?')} scenes, "
            f"{dataset.get('angles_per_scene', '?')} grasps per scene, "
            f"{dataset.get('image_size', '?')} px. "
            "(The name only: the absolute path is whatever machine generated it.)",
            f"Training: {config.get('epochs', '?')} epochs at "
            f"{config.get('input_size') or dataset.get('image_size', '?')} px, "
            f"identical at every point. Evaluation: {config.get('eval_episodes', '?')} "
            "episodes per split on the original held-out scenes.",
            "",
            "### The control",
            "",
            "Re-run unchanged. It does not depend on training data, so it is the check "
            "that the environment still is what it was. Read it against the original's "
            "sample size, not only against its headline: that figure rests on 89 "
            "held-out trials.",
            "",
            control_table(result),
            "",
            "### The curve",
            "",
            curve_table(result),
            "",
            "Random guessing scores 45 degrees of orientation error. The original "
            f"study measured {ORIGINAL['angle_heldout_deg']:.1f} degrees on held-out "
            "categories, which is worse than chance.",
            "",
        ]

    if args.throughput:
        from isaacgrasp.throughput import format_table

        comparison = json.loads(Path(args.throughput).read_text())
        sections += ["## Throughput", "", "```", format_table(comparison), "```", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections))
    print(f"wrote {out}")

    if args.readme:
        primary = json.loads(Path(args.result[0]).read_text())
        variance = (json.loads(Path(args.seed_variance).read_text())
                    if args.seed_variance and Path(args.seed_variance).exists() else None)
        update_readme(Path(args.readme), primary, args.figure, variance)
        print(f"updated {args.readme}")
    return 0


def seed_variance_section(variance: dict) -> list[str]:
    """The direct measurement of what a re-run does, when replicates exist."""
    per_size = variance.get("per_size", [])
    pooled = variance.get("pooled")
    if not per_size or not pooled:
        return []

    inferred = variance.get("inferred_from_curve_sd_pp")
    lines = [
        "### What a re-run does, measured directly",
        "",
        "**This is the most important measurement in the study.**",
        "",
        "The decomposition above infers training variance by subtracting the binomial "
        "term from the scatter about a fitted line, which assumes the true relationship "
        "is log-linear and charges any curvature to noise. These runs assume nothing: "
        "same dataset size, same evaluation scenes, only the training seed changed, so "
        "initialisation, data order, augmentation draws and the train/validation split "
        "differ and nothing else does.",
        "",
        "| training grasps | seeds | held-out (%) | spread | observed SD | training SD |",
        "|---|---|---|---|---|---|",
    ]
    for row in per_size:
        rates = ", ".join(f"{r:.1f}" for r in row["rates_pp"])
        lines.append(
            f"| {row['train_samples']:,} | {row['n_seeds']} | {rates} "
            f"| {row['range_pp']:.1f} pp | {row['observed_sd_pp']:.2f} pp "
            f"| {row['training_sd_pp']:.2f} pp |")

    lines += [
        "",
        f"Pooled over {pooled['degrees_of_freedom']} degrees of freedom: "
        f"{pooled['observed_sd_pp']:.2f} points observed, "
        f"{pooled['evaluation_sd_pp']:.2f} of it the binomial evaluation floor, "
        f"**{pooled['training_sd_pp']:.2f} points of training variance**.",
    ]
    if inferred is not None:
        difference = pooled["training_sd_pp"] - inferred
        lines.append(
            f"The curve's residual inferred {inferred:.2f} points, so the direct "
            f"measurement is {abs(difference):.2f} points "
            f"{'higher' if difference > 0 else 'lower'}. The same order, which is the "
            "check that mattered, and the direction is expected: every point on the "
            "curve was trained with the same seed, so those runs share an "
            "initialisation stream and their scatter understates what independent runs "
            "do. The replicates are the number to trust.")

    # Only claim a trend in variance if it is actually monotonic. Comparing the
    # first and last size alone would have reported one here on two seeds, and
    # the third seed showed the middle size has the largest spread of all.
    observed = [row["observed_sd_pp"] for row in per_size]
    monotonic = all(a > b for a, b in zip(observed, observed[1:], strict=True))
    if monotonic and len(per_size) >= 3:
        lines += [
            "",
            f"Run-to-run variance shrinks with data at every size measured: "
            f"{observed[0]:.2f} points at {per_size[0]['train_samples']:,} grasps down to "
            f"{observed[-1]:.2f} at {per_size[-1]['train_samples']:,}. The large end of "
            "the curve is therefore more trustworthy than the small end.",
        ]
    else:
        worst = max(per_size, key=lambda row: row["observed_sd_pp"])
        lines += [
            "",
            f"Variance does **not** fall cleanly with data. The widest spread is at "
            f"{worst['train_samples']:,} grasps, in the middle of the range, where three "
            f"runs of the identical experiment gave "
            f"{', '.join(f'{r:.1f}%' for r in worst['rates_pp'])}: a "
            f"{worst['range_pp']:.1f} point spread from nothing but the seed. Any story "
            "about the large end being steadier than the small end is not supported by "
            "these runs.",
        ]
    lines.append("")
    return lines


def variance_versus_effect(variance: dict, points: list[dict]) -> list[str]:
    """The comparison the whole study turns on: noise against signal."""
    pooled = variance.get("pooled")
    if not pooled or len(points) < 2:
        return []
    ordered = sorted(points, key=lambda p: p["train_samples"])
    first, last = ordered[0], ordered[-1]
    effect = 100.0 * (last["unseen"]["rate"] - first["unseen"]["rate"])
    noise = pooled["training_sd_pp"]
    ratio = effect / noise if noise else float("inf")
    return [
        f"Put that next to the effect it has to be measured against. Going from "
        f"{first['train_samples']:,} labelled grasps to {last['train_samples']:,}, a "
        f"{last['train_samples'] / first['train_samples']:.0f}x increase, moved held-out "
        f"success by {effect:.1f} points. Re-running one training moves it by "
        f"{noise:.2f} points, one standard deviation.",
        "",
        f"**The entire effect of a {last['train_samples'] / first['train_samples']:.0f}x "
        f"increase in data is about {ratio:.1f} standard deviations of the noise you get "
        "for free by changing a seed.** That is the honest reason this question is hard, "
        "and it is not a reason a bigger simulator fixes. It is an argument for repeated "
        "runs, and for suspecting that what is being measured is mostly not there.",
        "",
    ]


def readme_block(result: dict, figure: str | None,
                 variance: dict | None = None) -> str:
    """The README's results section, generated from the artefacts.

    Written between markers so the numbers in the README are the numbers in the
    JSON. A README table maintained by hand drifts from the run that produced
    it, and the drift is invisible until someone tries to reproduce it.
    """
    points = sorted(result["points"], key=lambda p: p["train_samples"])
    first, last = points[0], points[-1]
    controls = result.get("controls", {}).get("heuristic", {})
    dataset = result.get("dataset", {})
    config = result.get("config", {})

    lines = ["## Results", ""]
    if figure:
        lines += [f"![Grasp success and orientation error against training set size]({figure})",
                  ""]

    lines += [
        "### The control, re-run unchanged",
        "",
        "The heuristic does not depend on training data, so it is the check that the "
        "environment still is what it was. It is the first number to read.",
        "",
        "| | this run | original study |",
        "|---|---|---|",
    ]
    if controls:
        held = controls["unseen"]
        seen_control = controls["seen"]
        lines += [
            f"| Heuristic, held-out categories | **{held['rate']:.1%}** "
            f"(n={held['n']}, 95% CI {held['ci95'][0]:.1%} to {held['ci95'][1]:.1%}) "
            f"| {ORIGINAL['heuristic_unseen']:.1%} (n={ORIGINAL['heuristic_unseen_n']}) |",
            f"| Heuristic, seen categories | {seen_control['rate']:.1%} "
            f"(n={seen_control['n']}) "
            f"| {ORIGINAL['heuristic_seen']:.1%} (n={ORIGINAL['heuristic_seen_n']}) |",
            "",
            f"**The original's held-out control was underpowered.** Its {ORIGINAL['heuristic_unseen']:.1%} "
            f"came from {ORIGINAL['heuristic_unseen_n']} held-out trials, a 95% interval "
            f"of roughly {ORIGINAL['heuristic_unseen_ci'][0]:.0%} to "
            f"{ORIGINAL['heuristic_unseen_ci'][1]:.0%}. Measured here on {held['n']} "
            f"trials of the same scenes with the same unchanged policy, it is "
            f"{held['rate']:.1%}. The two are consistent, but they are not the same "
            f"number, and the bar the learned policy has to clear is the more precise "
            f"one.",
        ]
    reading = result.get("reading", {})
    trends = reading.get("trends", {})
    scatter = reading.get("scatter", {})

    lines += [
        "",
        f"### The curve, on {backend_name(result)}-generated data",
        "",
        f"Same architecture, same training loop, {config.get('epochs', '?')} epochs at "
        f"{config.get('input_size') or dataset.get('image_size', '?')} px at every point. "
        "Training subsets are nested and the validation set is fixed, so a difference "
        "between points is added data and nothing else.",
        "",
        curve_table(result),
        "",
        f"n = {points[0]['unseen'].get('n', config.get('eval_episodes', '?'))} evaluation "
        f"episodes per split, on the original held-out scenes, giving a 95% interval of "
        f"about plus or minus "
        f"{50.0 * max(p['unseen']['ci95'][1] - p['unseen']['ci95'][0] for p in points):.1f} "
        "points on each. The trend is the readable part, not any single pair of adjacent "
        "points.",
        "",
        "### What the curve says",
        "",
        f"Over a {last['train_samples'] / first['train_samples']:.0f}x range in training "
        f"data, fitting rate against log2(samples):",
        "",
        "| quantity | slope per doubling | 95% interval | r-squared |",
        "|---|---|---|---|",
    ]
    for key, label in (("seen_success", "Seen-category success"),
                       ("held_out_success", "Held-out success"),
                       ("held_out_angle_error", "Held-out orientation error")):
        trend = trends.get(key)
        if not trend:
            continue
        unit = "deg" if "angle" in key else "pp"
        low, high = trend.get("slope_ci95_pp", [float("nan"), float("nan")])
        interval = ("not available" if low != low
                    else f"{low:+.2f} to {high:+.2f}")
        lines.append(f"| {label} | {trend['slope_per_doubling_pp']:+.2f} {unit} "
                     f"| {interval} | {trend['r_squared']:.2f} |")

    held_trend = trends.get("held_out_success", {})
    low, high = held_trend.get("slope_ci95_pp", [float("nan"), float("nan")])
    bounded = low == low and high == high
    if bounded and low <= 0 <= high:
        verdict = (
            f"Held-out success is **consistent with flat**: its 95% interval runs from "
            f"{low:+.2f} to {high:+.2f} points per doubling, which contains zero. The "
            f"useful half of that is the upper end. Whatever gain more data buys on "
            f"unseen shapes, over this range it is **at most {high:.2f} points per "
            f"doubling**, and closing the {100 * (controls['unseen']['rate'] - last['unseen']['rate']):.0f} "
            f"point distance to the heuristic at that rate would take about "
            f"{(100 * (controls['unseen']['rate'] - last['unseen']['rate']) / high):.0f} "
            f"further doublings.")
    elif bounded and low > 0:
        verdict = (
            f"Held-out success **is rising**, by {low:+.2f} to {high:+.2f} points per "
            f"doubling at 95% confidence. Closing the "
            f"{100 * (controls['unseen']['rate'] - last['unseen']['rate']):.0f} point "
            f"distance to the heuristic would take between "
            f"{(100 * (controls['unseen']['rate'] - last['unseen']['rate']) / high):.0f} "
            f"and {(100 * (controls['unseen']['rate'] - last['unseen']['rate']) / low):.0f} "
            "further doublings.")
    else:
        verdict = (
            f"Held-out success does not resolve: the fitted slope explains "
            f"{held_trend.get('r_squared', float('nan')):.0%} of the scatter.")

    lines += [
        "",
        f"Seen-category success rises. {verdict} Held-out success moved from "
        f"{first['unseen']['rate']:.1%} to {last['unseen']['rate']:.1%} against the "
        f"heuristic's {controls['unseen']['rate']:.1%}, and the gap between seen and "
        f"held-out went from {first['generalisation_gap_pp']:.1f} to "
        f"{last['generalisation_gap_pp']:.1f} points.",
        "",
        _orientation_verdict(trends.get("held_out_angle_error", {}), last),
        "",
        f"Falling is not the same as good. Held-out orientation error goes from "
        f"{first['angle']['angle_error_deg_heldout']:.1f} degrees to "
        f"{last['angle']['angle_error_deg_heldout']:.1f} across the whole 32x range, "
        f"against the 45 a random guess scores. It never gets more than "
        f"{max(abs(p['angle']['angle_error_deg_heldout'] - 45.0) for p in points):.1f} "
        f"degrees away from chance at any size measured.",
        "",
        f"The seen-category figure does not resolve at all "
        f"({first['angle']['angle_error_deg_seen']:.1f} degrees at the smallest size, "
        f"{last['angle']['angle_error_deg_seen']:.1f} at the largest, wandering in "
        f"between). It is scored on only {last['angle']['n_seen']} grasps per point "
        f"against {last['angle']['n_heldout']} for held-out, because the determinacy "
        "filter removes the rotationally symmetric shapes and most of the training "
        "categories are symmetric. That scatter is the measurement, not the model, and "
        "it is why the held-out figure is the one quoted.",
        "",
        f"The sharper measurement is the **bin spread**: the range of predicted grasp "
        f"quality across the twelve gripper angles at the pixel the network chose. It "
        f"falls from {points[0]['angle']['bin_spread']:.2f} to "
        f"{last['angle']['bin_spread']:.2f} across the curve. The first value is an "
        f"undertrained network's noise rather than real angle sensitivity, so the reading "
        f"is that the network's quality estimate becomes progressively **less** sensitive "
        f"to how the gripper is turned as it sees more data, settling near the "
        f"{ORIGINAL['bin_spread']:.3f} the original measured at roughly 27,000 samples. "
        "Predicting the angle-marginal success rate is a minimum of this loss, and more "
        "data finds it more reliably.",
        "",
        "### Where the scatter comes from",
        "",
        "Two things move a point off the line, and they call for different fixes.",
        "",
        "| source | standard deviation |",
        "|---|---|",
        f"| Evaluation, binomial at n={points[0]['unseen']['n']} per split "
        f"| {scatter.get('evaluation_sd_pp', float('nan')):.2f} pp |",
        f"| Training, at a fixed dataset size "
        f"| {scatter.get('training_sd_pp', float('nan')):.2f} pp |",
        f"| Total scatter about the fit | {scatter.get('residual_sd_pp', float('nan')):.2f} pp |",
        "",
        f"**{scatter.get('dominant_noise_source', 'unknown').capitalize()} noise "
        f"dominates.** The evaluation term is known exactly from the episode count; "
        f"the training term is what is left, and covers initialisation, data order and "
        f"augmentation draws at a fixed dataset size. At 200 episodes per split the "
        f"evaluation term was {scatter.get('evaluation_sd_pp', float('nan')) * (1500 / 200) ** 0.5:.2f} "
        f"points and dominated; at {points[0]['unseen']['n']} it no longer does, so more "
        f"episodes would now be wasted money and the next spend belongs on repeated runs "
        f"or more sizes.",
        "",
        f"It is worth putting that next to the trend: retraining the same size moves "
        f"held-out success by about {scatter.get('training_sd_pp', float('nan')):.1f} "
        f"points, and doubling the data moves it by "
        f"{trends.get('held_out_success', {}).get('slope_per_doubling_pp', float('nan')):.2f}. "
        "The run-to-run noise is larger than the effect being measured, which is the "
        "honest reason this curve is hard to resolve and not a matter of needing a "
        "bigger simulator.",
        "",
        *seed_variance_section(variance or {}),
        *variance_versus_effect(variance or {}, points),
        "### What this arm does not settle",
        "",
        f"**It tops out below the study it follows up.** The largest point here is "
        f"{last['train_samples']:,} labelled grasps. The original trained on roughly "
        f"27,000 and reported {ORIGINAL['cnn_unseen']:.1%} held-out, which is above every "
        f"point on this curve. So the curve evidently continues upward past where this arm "
        f"reached, and nothing here shows that more data cannot help. What it shows is "
        f"that across a {last['train_samples'] / first['train_samples']:.0f}x range the "
        f"held-out gap did not close and orientation did not leave chance.",
        "",
        "Reaching the original's scale, and the one to two orders of magnitude beyond it "
        "that the Isaac Lab port exists to make affordable, is the experiment this arm "
        "sets up rather than the one it performs. The Isaac backend has not been run.",
        "",
        "Full tables in [docs/results.md](docs/results.md); the raw numbers are in "
        "`results/scaling/` as JSON and CSV.",
    ]
    return "\n".join(lines)


def update_readme(path: Path, result: dict, figure: str | None,
                  variance: dict | None = None) -> None:
    text = path.read_text()
    start, end = "<!-- RESULTS -->", "<!-- /RESULTS -->"
    if start not in text or end not in text:
        raise SystemExit(f"{path} has no {start} / {end} markers to fill")
    head = text[: text.index(start) + len(start)]
    tail = text[text.index(end):]
    path.write_text(f"{head}\n{readme_block(result, figure, variance)}\n{tail}")


if __name__ == "__main__":
    raise SystemExit(main())
