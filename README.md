# Isaac grasp scaling

**Does a grasping network's orientation failure survive more data, or was 15,000
samples simply too few?**

A follow-up to [Simulated-grasping](https://github.com/abyyworld/Simulated-grasping),
where a fully-convolutional grasp-quality network trained on self-supervised
grasp attempts in MuJoCo **lost to a hand-written depth heuristic** on unseen
objects, 58.4% against 75.3%. An ablation traced the cause to the network
learning grasp *position* but not grasp *orientation*: mean angle error against
the oracle was 47.2 degrees on held-out shapes, worse than the 45 degrees of a
random guess.

The obvious objection is that the dataset was too small. That objection is
testable. This repository ports the task to NVIDIA Isaac Lab, which runs
thousands of environments in parallel on one GPU, regenerates the dataset at
much larger scale, retrains the same architecture, and plots success against
training set size.

Either answer publishes. If more data closes the gap, the original negative
result is explained. If it does not, the failure is architectural, which is a
sharper claim than the original made.

---

<!-- RESULTS -->
<!-- /RESULTS -->

---

## Status: what has been run, and what has not

This matters more than usual here, so it is at the top rather than buried.

| | status |
|---|---|
| MuJoCo control arm: collect, train, evaluate, curve | **run, numbers below** |
| Heuristic baseline, re-run unchanged | **run, reproduces the original** |
| Object catalogue translation to USD | **tested** (48 tests, no GPU needed) |
| Oracle grasp, success criterion, batched collector | **tested against the originals** |
| Isaac Lab backend, in contact with the simulator | **never executed** |

The Isaac Lab backend was written on a machine with no NVIDIA GPU, where Isaac
Sim cannot be installed. Everything about the port that can be checked without
one is checked and passes; what remains unverified is the code's contact with
the simulator. `src/isaacgrasp/backends/isaac_backend.py` carries that status
banner in its own docstring, and a test fails if the banner is removed.

`scripts/check_setup.py --isaac` is the gate. It exercises the backend in six
named stages and reports which one fails, because the useful question on a fresh
GPU box is not "does it work" but "which of the six things it needs is missing".

---

## Quick start

```bash
make install
make assets         # Franka Panda MJCF, ~33 MB, from a pinned Menagerie commit
make check          # verifies the install by running it, not by reading it
make test
```

Then, on CPU, the control arm:

```bash
make collect  BACKEND=mujoco SCENES=6000 DATA=data/mj18k
make scaling  DATA=data/mj18k OUT=results/scaling/mujoco
```

On a rented RTX box, the Isaac arm (see [docs/setup-cloud.md](docs/setup-cloud.md)):

```bash
bash scripts/setup_cloud.sh --check     # refuses a GPU with no RT cores
bash scripts/setup_cloud.sh
python scripts/check_setup.py --isaac   # the gate. Read its output.
make collect BACKEND=isaac SCENES=200000 NUM_ENVS=1024 DATA=data/isaac600k
```

### Where to get a GPU

The two stages want different machines, and splitting them is what keeps this
affordable:

* **Dataset generation** needs an RTX-class GPU with working Vulkan and about
  30 GB of disk. A spot RTX 4090 on Vast.ai or RunPod is a few dollars for the
  whole run. GCP's free trial and the GitHub Student Pack's Azure credit both
  cover it.
* **Training** is ordinary PyTorch and runs free on Kaggle's T4 x2, which is 30
  hours a week. [`notebooks/kaggle_scaling.ipynb`](notebooks/kaggle_scaling.ipynb)
  is ready to run: attach your dataset, set the accelerator, go. That is where the
  settings the original study said it needed and could not afford belong, namely
  30 epochs at 224 px with ImageNet initialisation.
* **Kaggle cannot run Isaac Sim.** Its P100 has no RT cores at all; its T4 has
  them but is not on NVIDIA's supported list, and the session disk and 12-hour
  limit sit badly against a 30 GB install that does not persist.
* **The MuJoCo control arm and every evaluation** are CPU only.

Full detail, including the exact provisioning steps, in
[docs/setup-cloud.md](docs/setup-cloud.md).

---

## How the comparison is kept honest

The entire value of this project is that only one thing changed. That is
enforced mechanically, not promised.

**Nothing simulator-independent is reimplemented.** The network, the training
loop, the loss, the dataset format, the augmentation, the grasp sampler, the
heuristic baseline and the evaluation statistics are imported from `simgrasp` at
a commit pinned in `pyproject.toml`. There is no copy of any of them here, so
none of them can drift.

**What cannot be imported is checked numerically.** `isaacgrasp.parity` records
the 22 constants the port must reproduce, from the table height and the camera
field of view to the 0.08 m lift that defines success, and fails loudly if the
upstream definitions move. The manifest is written into every dataset and every
result file, so a number can be traced to the definitions it was produced under
without trusting this README.

**Three pieces are pinned by test rather than by import:**

| piece | test | why |
|---|---|---|
| the oracle grasp restated from episode state | `test_oracle_view.py` | it is the centre of the label distribution |
| the success criterion | `test_outcome_classification.py` | it defines what every number counts |
| the batched collector | `test_collect.py` | it must reproduce the original byte for byte |

The last one is the strongest: it runs the original collector and the batched one
over the same episodes and compares the images, the labels and the outcomes byte
for byte. That is what makes the throughput comparison a comparison of
simulators rather than of two separately-tuned pipelines.

**The control is re-run, never quoted.** The heuristic does not depend on
training data, so it should reproduce the original study exactly. It is the
first number in the results below for that reason: if it had come out different,
something in the environment had moved and every other number here would be
suspect.

**Training subsets are nested and the validation set is fixed.** Every point on
the curve validates on the same held-out 600 scenes, and the training set at
each size is a prefix of the next, so the curve measures added data rather than
a different draw.

---

## Separating "more data" from "different simulator"

Training on Isaac-generated data and evaluating in MuJoCo would confound two
variables. The design keeps them apart:

* **Evaluation always happens in MuJoCo**, on the original held-out scenes
  (`EVAL_EPISODE_OFFSET = 1_000_000`), against the original unchanged heuristic.
  Every number is directly comparable to the original study's.
* **A MuJoCo arm of the same curve** varies scale inside one simulator, with no
  sim-to-sim gap at all. That is the arm reported below.
* **A control point at the original scale.** If the Isaac arm at the same sample
  count reproduces the MuJoCo arm within the confidence interval, the simulator
  is not the confound and the rest of the Isaac curve is interpretable. If it
  does not, the Isaac curve measures scale plus a sim-to-sim gap and has to be
  reported that way.

---

## Repository layout

```
src/isaacgrasp/
  parity.py            the 22 constants the port must reproduce, and the check
  bootstrap.py         render backend selection, and the OSMesa/Triton guard
  collect.py           backend-agnostic dataset generation
  scaling.py           the experiment: train at each size, evaluate, measure angle
  angle.py             orientation error, the metric the question turns on
  plot.py              the committed curve
  throughput.py        samples per hour, read from what the collector recorded
  backends/
    base.py            the interface, the oracle, the success criterion
    mujoco_backend.py  batch of one over the original environment
    isaac_backend.py   NEVER EXECUTED. Read its status banner.
    isaac_scene.py     NEVER EXECUTED. The USD scene.
    isaac_objects.py   catalogue translation; pure geometry, fully tested

scripts/               one CLI per stage; all take --help
docs/design.md         decisions, the confound, the fallback plan
docs/setup-cloud.md    which GPU to rent, and what runs free
docs/results.md        generated from the artefacts by scripts/make_report.py
results/               committed JSON and CSV
```

---

## Two bugs found by running rather than reading

**The predecessor repository could not import itself.** Its `.gitignore` carried
an unanchored `data/` pattern for the generated dataset directory at the
repository root. Git applies an unanchored pattern at every level, so it also
matched `src/simgrasp/data/`, and that package was excluded from all 31 commits.
A fresh clone failed at import: no dataset, no training, no test suite. Local
runs and CI both passed, because both ran against a working tree that still had
the files. The package is gone from history, so it was reconstructed from the
committed callers and the committed tests; all 205 upstream tests pass against
the reconstruction. Fixed in
[abyyworld/Simulated-grasping#claude/restore-data-package](https://github.com/abyyworld/Simulated-grasping/tree/claude/restore-data-package).

**This repository segfaulted on its first real test run.** MuJoCo's OSMesa
software renderer and the Triton compiler bundled in the CUDA PyTorch wheel each
load their own LLVM, and whichever loads second kills the process with no
traceback. `isaacgrasp.bootstrap` loads torch and Triton before any GL context
exists. Importing torch alone is not enough, because torch loads Triton lazily
and the clash just moves.

Both repositories now carry `tests/test_repo_integrity.py`, which fails if any
source file becomes gitignored or any package `__init__.py` goes untracked. A
normal test suite cannot catch that, because it runs against the working tree
rather than against what was committed.

---

## Credits and licence

MIT licensed, see [LICENSE](LICENSE).

Built on [Simulated-grasping](https://github.com/abyyworld/Simulated-grasping),
which supplies the environment, the network and the evaluation. The Franka Emika
Panda model comes from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
Apache-2.0, fetched at a pinned commit and not vendored.
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) is BSD-3-Clause.
