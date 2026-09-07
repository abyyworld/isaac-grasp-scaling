# Design notes

## 1. The question, and what makes it answerable

The predecessor study trained a fully-convolutional grasp-quality network on
self-supervised grasp attempts in MuJoCo and lost to a hand-written depth
heuristic on unseen objects, 58.4% against 75.3%. An ablation traced the cause
to the network learning grasp position but not orientation: mean angle error
against the oracle was 47.2 degrees on held-out categories, worse than the 45
degrees of a random guess.

The obvious objection is that 15,000 attempts is simply too few. That objection
is testable, and the test is a curve: success rate and orientation error against
training set size, everything else held fixed.

"Everything else held fixed" is the load-bearing phrase, so it is enforced
mechanically rather than promised. See section 3.

## 2. What is actually being varied

One thing: the number of training scenes the loader may use
(`TrainConfig.limit`). Not the epochs, not the architecture, not the
augmentation, not the evaluation scenes, not the baseline.

That has a consequence worth stating plainly. Holding epochs fixed while the
dataset grows means the larger points see more gradient steps, so the curve
measures "more data at a fixed epoch budget", which is the practically
interesting question but not the only one. Holding *steps* fixed instead would
answer a different question. The epoch count is recorded per point so the
distinction stays visible.

## 3. How "unchanged" is enforced

Two mechanisms, because a comparison whose invariants are maintained by care
alone is not a comparison.

**Import, do not reimplement.** The network, the training loop, the loss, the
dataset format, the augmentation, the grasp sampler, the heuristic baseline and
the evaluation statistics are all imported from `simgrasp` at a commit pinned in
`pyproject.toml`. There is no copy of any of them in this repository, so none of
them can drift.

**Check numerically what cannot be imported.** The Isaac port has to rebuild the
scene, the robot and the object catalogue in a different engine. Those cannot be
imported, so `isaacgrasp.parity` records the 22 constants the port must
reproduce and fails loudly if the upstream definitions move. The manifest is
written into every dataset and every result file, so a number can be traced to
the definitions it was produced under without trusting this document.

Three further pieces are pinned by test rather than by import:

| piece | test | why it matters |
|---|---|---|
| the oracle grasp restated from episode state | `test_oracle_view.py` | it is the centre of the label distribution |
| the success criterion | `test_outcome_classification.py` | it defines what every number counts |
| the batched collector | `test_collect.py` | it must produce byte-identical output to the original |

`test_collect.py` is the strongest of these: it runs the original collector and
the batched one over the same episodes and compares the images, the labels and
the outcomes byte for byte.

## 4. The confound, and the control point

Training on Isaac-generated data and evaluating in MuJoCo confounds two
variables: more data, and a different simulator. A curve that rises could be
either.

The design separates them:

* **Evaluation always happens in MuJoCo**, on the same held-out scenes
  (`EVAL_EPISODE_OFFSET = 1_000_000`) with the same unchanged heuristic control,
  so every number is directly comparable to the original study's.
* **A MuJoCo arm of the same curve** is run with the same collector and the same
  driver. Scale varies within one simulator, with no sim-to-sim gap at all.
* **A control point at the original scale.** If the Isaac arm at roughly 15,000
  samples reproduces the MuJoCo arm at 15,000 samples within the confidence
  interval, the simulator is not the confound and the rest of the Isaac curve is
  interpretable. If it does not, the Isaac curve measures scale plus a
  sim-to-sim gap, and it has to be reported that way.

The heuristic control is re-run rather than quoted, for the same reason. It does
not depend on training data, so it should reproduce the original 75.3% on
held-out categories. If it does not, something in the environment has moved and
every other number here is suspect. It is the first thing to read in the output,
not a formality.

## 5. Porting the object catalogue

The catalogue is not redefined. Shapes, dimension ranges, the seen and held-out
split and the grasp hints are sampled by `simgrasp.objects.sample_object`, so
episode *i* means the same object in both simulators and only the placement into
the engine is new.

The translation is where a port goes quietly wrong, so each convention that
differs is converted in one place and tested against the MuJoCo definition it
came from:

* **Size semantics.** MuJoCo geom sizes are half-extents. USD `Cube` takes a
  full edge length; `Cylinder` and `Capsule` take a full height.
* **Capsule length.** MuJoCo's capsule size is `(radius, half_length)` with the
  hemispherical caps excluded, which happens to match USD `Capsule`'s `height`
  being the cylindrical section only.
* **Mass.** MuJoCo derives a body's mass from its geoms' volumes and the sampled
  density, double counting any overlap. `spec_mass` reproduces that number
  rather than the true solid volume, because matching the engine matters more
  here: mass sets the moment the jaws resist during the 0.20 m lift, which is
  what makes the dumbbell the hardest category.
* **Friction.** The catalogue samples a sliding coefficient per object, which
  does not carry over through geometry and decides whether a curved surface
  slips out during closing.

`tests/test_isaac_objects.py` checks all of this without a GPU, which is the
point of keeping the translation in a module with no simulator import.

## 6. The multi-slot object body, and its fallback

PhysX creates collision shapes when the scene is built, so a primitive that was
not present at build time cannot appear later without rebuilding, and rebuilding
per episode costs more than the episode. Each object body therefore carries
three slots, each holding one primitive of every type the catalogue uses, and an
episode activates one per slot and shrinks the rest.

**The risk.** Whether PhysX updates an analytic shape's collision geometry when
its `xformOp:scale` changes at runtime has not been verified. If it does not,
objects settle at the wrong height and every label is wrong in a way that still
produces a plausible-looking dataset.

**The check.** `scripts/check_setup.py --isaac`, stage `isaac:object geometry`,
resets twice with different objects and asserts the observed heights change.

**The fallback, if it fails.** Fix the category per environment and rebuild the
scene every few thousand episodes rather than rewriting per episode. Dimensions
still randomise per rebuild; only the category becomes blocky in the sampling
order, which is corrected by shuffling environments across rebuilds. This costs
a rebuild every few thousand episodes against a rewrite every episode, and it is
a change to `isaac_backend.py` alone.

## 7. Two bugs found by running, one inherited

**The predecessor repository could not import itself.** Its `.gitignore` carried
an unanchored `data/` pattern intended for the generated dataset directory at
the repository root. Git applies an unanchored pattern at every level, so it
also matched `src/simgrasp/data/`, and that package was excluded from all 31
commits. `collect.py` does `from .data.writer import ShardWriter` and
`training.py` does `from .data import GraspDataset`, so a fresh clone failed at
import: no dataset, no training, no test suite. Local runs and CI both passed,
because both ran against a working tree that still had the files.

The package is gone from history, so it was reconstructed from the committed
callers and the committed tests, which specify it in unusual detail. All 205
upstream tests pass against the reconstruction. It is a reconstruction and not a
recovery, and the commit says so.

`tests/test_repo_integrity.py` exists in both repositories now: it fails if any
file under `src`, `scripts` or `tests` is excluded by `.gitignore`, or if a
package `__init__.py` is untracked. A normal suite cannot catch this, because it
runs against the working tree rather than against what was committed.

**This repository segfaulted on its first real test run.** MuJoCo's OSMesa
software renderer and the Triton compiler bundled in the CUDA PyTorch wheel each
load their own LLVM, and whichever loads second kills the process: no traceback,
just a stack dump. `test_collect.py` hit it the first time it ran. The
predecessor documents the same failure and guards against it in its `scripts/`
directory, which pip does not install, so the guard was rebuilt here as
`isaacgrasp.bootstrap` and applied from `tests/conftest.py`. Importing torch
alone is not enough: torch loads Triton lazily, so the clash just moves.

Both were found by executing, not by reading. That is the habit this project
inherited on purpose.

## 8. What the scatter is made of, and what that changed

The first pass of this curve evaluated each point on 200 episodes per split. The
fitted held-out slope came out at +0.80 points per doubling with an r-squared of
0.09 and a 95% interval running from -4.0 to +5.6. That bounds nothing: it is
consistent with the curve falling, and with it rising fast enough to reach the
control almost immediately.

An r-squared says the fit is bad. It does not say what the data rule out, and
with a null result that is the only interesting part, so the fit now carries the
slope's standard error and a t interval. The report reads the interval.

**Then the points were re-measured rather than the conclusion rewritten.**
Evaluation is cheap next to training, because it reuses checkpoints that already
exist, so `scripts/reevaluate.py` re-scored every point at 1,500 episodes per
split. That took the interval on each point from about plus or minus 7 points to
plus or minus 2.5.

Two things came out of that, and neither was the expected one.

**The control moved.** The heuristic on 1,500 held-out episodes gives 79.5%,
not the 75.5% measured on 200. The earlier agreement with the predecessor's
published 75.3% was a coincidence of two small samples. They are consistent, and
75.3% rests on 89 trials whose interval spans roughly 65% to 83%, but they are
not the same number and the more precise one is the bar. The claim that the
control "reproduced the original" was the headline reassurance of this
repository and it was withdrawn rather than reconciled.

**The bottleneck moved.** With the evaluation term shrunk, the points still
scatter about the fit by 2.99 points. The binomial term at n=1500 accounts for
1.29 of that. Variances add, so the remainder, 2.70 points, is the training run
itself: initialisation, data order, augmentation draws, at a fixed dataset size.

That single number reorganises the experiment:

* Retraining the same size moves held-out success by about 2.7 points. Doubling
  the data moves it by 1.3. **The run-to-run noise is larger than the effect
  being measured.**
* More evaluation episodes are now wasted money. The next spend belongs on
  repeated seeds at the same sizes, or on more sizes.
* It is a property of this training setup, not of the simulator. A larger
  simulator does not fix it, which matters because the Isaac Lab port is
  precisely an argument for a larger simulator.

`isaacgrasp.scaling.decompose_scatter` computes the split rather than asserting
it, and reports which term dominates.

## 9. A 4.5x speedup that was pure waste

Evaluation runs one MuJoCo environment per worker process. Each worker imports
torch, and torch sizes its intra-op thread pool to the whole machine, so four
workers each claimed four cores: sixteen threads over four cores, all of them
context-switching.

Pinning one thread per worker for the duration of an evaluation takes it from
1.12 seconds per episode to 0.25, measured over the same 40 episodes with
byte-identical results (42.5% both ways). That is the only reason 1,500-episode
evaluation was affordable at all.

Training is deliberately left alone. There the work is one process with a large
batch, and intra-op parallelism is worth having.

The measurement itself needed two attempts. The first benchmark ran the
evaluation under `python -c`, which has no `__main__` guard, and multiprocessing
with the spawn start method hung rather than failing. That is worth writing down
because the symptom, a process at zero CPU with no output, looks nothing like
its cause.

## 10. Executing the unverifiable

The Isaac Lab backend cannot be run without an RTX GPU, and until
`tests/fakes/isaac.py` existed, not one line of it had been executed by
anything. The stub modules do not fix that. They cannot: they were written from
the same understanding of the Isaac Lab API that wrote the backend, so a shared
misunderstanding survives both, and `scripts/check_setup.py --isaac` on a real
machine remains the only thing that settles it.

What they do is separate two kinds of doubt that were previously tangled. The
Isaac API binding is still unverified. Everything on this side of the boundary
is not: the TCP-to-hand-body offset, the world-to-base-frame conversion, the
per-environment ordering of a batch, the prim scaling, the mass and friction,
the joint addressing, the depth-to-height path, the outcome rule.

The harness earned itself on its first run, by failing. The stub was the side
that was wrong: `root_state_w` is a world pose and carries the environment
origin, which is why the backend recovers the mount as `root_state_w` minus
`env_origins`. Being forced to write that convention down explicitly, and to
decide which side was wrong, is worth more than the test having passed.

## 11. Known limitations

* The Isaac backend has never been executed. See the status banner at the top of
  `isaac_backend.py` and the gate in `scripts/check_setup.py`.
* The evaluation is single-object on a clean table with a noiseless depth
  camera, exactly as in the original, so absolute success rates are optimistic
  relative to a real robot. The seen against held-out gap is the number that
  carries meaning.
* Success rates at n = 200 per split carry a 95% interval of roughly plus or
  minus 7 points. Small differences between adjacent points on the curve are
  noise, and the intervals are plotted for that reason.
* Orientation error is scored only on objects with a determinate grasp axis. A
  cylinder grasps equally well at every angle, so scoring it against "the"
  oracle angle measures nothing; an earlier version of the upstream script
  averaged those in and reported no effect where there was a large one.
