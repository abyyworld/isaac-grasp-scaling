# Orientation for anyone picking this up

## What this project is

A follow-up to [Simulated-grasping](https://github.com/abyyworld/Simulated-grasping),
not a rewrite of it. That study's network lost to a depth heuristic on unseen
objects, 58.4% against 75.3%, and the cause was traced to orientation. This one
asks whether that was a data problem, by porting the task to Isaac Lab,
regenerating the dataset at much larger scale, and plotting success against
training set size.

Either answer publishes. If more data closes the gap, the original negative
result is explained. If it does not, the failure is architectural, which is a
sharper claim than the original made.

## The one rule

**The model, the training loop, the dataset format, the heuristic baseline and
the evaluation are imported from `simgrasp` at a pinned commit. Do not
reimplement any of them here.** The entire value of the comparison is that only
the training set size changed. If you find yourself copying a function out of
`simgrasp`, stop: either import it, or add it to `isaacgrasp.parity.EXPECTED`
and pin it with a test that compares against the original.

`pyproject.toml` pins the commit. Moving that pin invalidates every committed
number unless they are regenerated.

## Layout

```
src/isaacgrasp/
  parity.py          the 22 constants the port must reproduce, and the check
  bootstrap.py       render backend selection, and the OSMesa/Triton guard
  collect.py         backend-agnostic dataset generation
  scaling.py         the experiment: train at each size, evaluate, measure angle
  angle.py           orientation error, the metric the question turns on
  plot.py            the committed curve
  throughput.py      samples per hour, read from what the collector recorded
  backends/
    base.py          the interface, plus the oracle and the success criterion
    mujoco_backend.py  batch of one over the original environment
    isaac_backend.py   NEVER EXECUTED. Read its status banner first.
    isaac_scene.py     NEVER EXECUTED. The USD scene.
    isaac_objects.py   catalogue translation; pure geometry, fully tested

scripts/           one CLI per stage; all take --help
docs/design.md     decisions, the confound, the fallback plan
docs/setup-cloud.md  which GPU to rent, and what runs free
results/           committed JSON and CSV
```

## Before you change anything

```bash
make install
make check        # verifies the install by running it
make test
```

`make check` is not a formality. It renders a frame, executes a grasp and runs
the collector. On a GPU box, `make check ISAAC=1` adds six Isaac stages and
names the one that fails.

## The habits this project inherited

1. **Verify by running, never by reading.** Two real bugs were found this way
   while building it, both invisible to inspection. See `docs/design.md` section 7.
2. **Re-measure after every fix.** A correctness change is not free until it has
   been re-benchmarked.
3. **When a headline number will not reproduce, withdraw it.** Do not reconcile
   it with a story. The heuristic control exists to catch exactly this: it does
   not depend on training data, so if it stops reproducing the original 75.3%,
   something moved and every other number here is suspect.

## What is unfinished

The Isaac Lab backend has never been executed, because it was written on a
machine with no NVIDIA GPU. Everything that can be checked without one is
checked and passes. `scripts/check_setup.py --isaac` is the gate; run it before
anything long. The README says which numbers exist and which do not, and that
should stay accurate.
