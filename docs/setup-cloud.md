# Where to run this, and what it costs

Two stages, two very different machines. Splitting them is what keeps this
project affordable on a student budget.

| stage | needs | where |
|---|---|---|
| Isaac Lab dataset generation | RTX GPU, Vulkan, ~30 GB disk | rented, hours |
| Training the network | any CUDA GPU | free on Kaggle |
| MuJoCo control arm and evaluation | CPU only | anywhere, including free tiers |

The dataset is the handoff artefact between stages. Generate it once on a rented
box, download it, and everything after that is free.

---

## What Isaac Sim actually requires

* **An RTX-class GPU.** Isaac Sim renders through NVIDIA's RTX path and needs
  ray-tracing cores. This project consumes camera output, so a physics-only
  configuration is not enough.
* **Vulkan**, with a driver that exposes it. A container that ships `libvulkan`
  without a working ICD fails at the renderer rather than at install.
* **About 30 GB of disk** for Isaac Sim plus the extension cache, before any
  dataset.
* **Linux**, headless. `AppLauncher(headless=True, enable_cameras=True)`. The
  `enable_cameras` half is easy to miss and fails silently: every array has the
  right shape and every depth value is zero.

## Kaggle and Colab

Kaggle offers **T4 x2**, **P100** and **TPU v5e-8**. For this project:

* **P100: no.** Pascal has no RT cores at all.
* **TPU: no.** Isaac Sim is CUDA software.
* **T4: not for Isaac.** Turing does have RT cores, so a T4 is not
  categorically excluded the way a P100 is, but it is not on NVIDIA's supported
  list, the session disk and the 12-hour session limit sit awkwardly against a
  30 GB install that does not persist, and a Kaggle image is not built to expose
  Vulkan. **This has not been tested here**, so treat it as unpromising rather
  than as proven impossible. If you want to try it, the gate is
  `python scripts/check_setup.py --isaac --num-envs 4`, and it will tell you
  within a few minutes which stage fails.
* **T4 x2 for training: yes, and this is the useful part.** There is a ready notebook at
  [`notebooks/kaggle_scaling.ipynb`](../notebooks/kaggle_scaling.ipynb): attach your dataset,
  set the accelerator to T4 x2, run it. Training the network
  is ordinary PyTorch. 30 hours a week of free T4 covers the 30 epochs at
  224x224 with `--pretrained` that the original study said it needed and could
  not afford on a laptop CPU. Upload the dataset, run `scripts/run_scaling.py`,
  download the results.

Colab's free tier is the same picture with a shorter leash.

## Renting a GPU for the generation stage

Order of magnitude only. **Check current prices before booking**, these move.

| provider | card | roughly | notes |
|---|---|---|---|
| Vast.ai | RTX 4090 | $0.25 to $0.40 / hr | cheapest; interruptible, verify the disk allowance |
| RunPod | RTX 4090 / A40 | $0.35 to $0.50 / hr | community and secure tiers; Isaac templates exist |
| Lambda | A10 | around $0.75 / hr | simpler, pricier |
| GCP | L4 | on-demand | covered by the $300 free trial |
| Azure | NCasT4_v3 | on-demand | GitHub Student Pack includes credit |

Student routes worth claiming first: the **GitHub Student Developer Pack**
(Azure credit and more), and **Google Cloud's $300 free trial**, which covers L4
instances. Between them the generation stage can plausibly cost nothing.

**Budget.** The generation run is hours, not days. Even at $0.40/hr with a
generous allowance for failed first attempts, this is single-digit dollars. Rent
by the hour, generate, download the dataset, stop the instance.

## Provisioning

```bash
bash scripts/setup_cloud.sh --check    # report GPU, disk and Vulkan, change nothing
bash scripts/setup_cloud.sh            # full install
```

`--check` first, always. It refuses a card with no RT cores and warns on a T4,
before you have spent 30 GB and twenty minutes finding out the slow way.

Then the gate:

```bash
conda activate env_isaaclab
python scripts/check_setup.py --isaac --num-envs 16
```

This is the most important command in the repository. It exercises the Isaac
backend in six stages and names the one that fails. **Do not start a long
collection run until it passes.** The `isaac:object geometry` stage checks
whether PhysX picks up a runtime rescale of an analytic collision shape, which
has never been verified anywhere; if it does not, the failure is silent in the
data rather than loud at runtime, and every label in the dataset is wrong in a
way that still looks like a plausible dataset.

## Then

```bash
# on the rented box
python scripts/collect.py --backend isaac --scenes 200000 --num-envs 1024 \
    --angles-per-scene 3 --split seen --out data/isaac600k

# download data/isaac600k, stop the instance, and continue for free
python scripts/run_scaling.py --data data/isaac600k --out results/scaling/isaac \
    --sizes 2000 6000 20000 60000 200000 --epochs 12
```

`--num-envs` is the throughput knob. Start at 256, confirm a full collection
cycle, then raise it until GPU memory or the render budget stops you, and record
what you used: the throughput comparison is only meaningful next to the batch
size that produced it.

---

## Getting results off a throwaway machine

Both Kaggle and a rented instance disappear when you stop paying attention to
them, so results have to leave before the machine does.
`scripts/push_results.py` commits the artefacts and pushes them to a branch:

```bash
GITHUB_TOKEN=... python scripts/push_results.py \
    --paths results/scaling/kaggle --branch kaggle-results
```

**The token.** Create a **fine-grained** token at
<https://github.com/settings/personal-access-tokens>, scoped to only this
repository with **Contents: Read and write**, and give it a short expiry. On
Kaggle, store it under **Add-ons, Secrets** as `GITHUB_TOKEN` and attach it to
the notebook. Never paste it into a cell: a public notebook publishes its own
source and its output.

The script reads the token from the environment rather than from an argument, so
it stays out of process listings and shell history; passes it to git through a
credential helper rather than a remote URL, so it is never written into
`.git/config`; and scrubs anything token-shaped out of git's output before
printing, so an error message that echoes a rewritten URL does not leak it
either. `tests/test_push_results.py` covers all three, and sweeps every tracked
file for credential-shaped strings.

It pushes to a branch rather than the default branch. A notebook should propose
results, not rewrite history.

**If a token is ever exposed**, including by pasting it into a chat or a
terminal someone else can read, revoke it at the link above. Rotation is cheap;
a leaked write-scoped token on a public repository is not.
