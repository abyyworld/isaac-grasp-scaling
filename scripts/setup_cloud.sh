#!/usr/bin/env bash
# Provision a rented Ubuntu 22.04 box with an RTX GPU for Isaac Lab data
# generation. See docs/setup-cloud.md for which box to rent and why.
#
#   bash scripts/setup_cloud.sh            # full install
#   bash scripts/setup_cloud.sh --check    # report what is present, change nothing
#
# This script has been shellcheck-clean and dry-run traced, but it has NOT been
# executed end to end on a real GPU instance: the machine it was written on has
# no NVIDIA hardware. Treat the first run as part of the experiment and read the
# output rather than backgrounding it.

set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
# Isaac Sim 4.5 / Isaac Lab 2.x is the pairing this port was written against.
# Newer pairs may work; the API moved between 1.x and 2.x, so 1.x will not.
ISAACSIM_VERSION="${ISAACSIM_VERSION:-4.5.0}"
ISAACLAB_REF="${ISAACLAB_REF:-v2.0.2}"
PYTHON_VERSION="3.10"
REQUIRED_DISK_GB=40

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail ]\033[0m %s\n' "$*" >&2; exit 1; }

check_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || die \
    "nvidia-smi not found. This box has no usable NVIDIA driver, so Isaac Sim
     cannot run. Rent an instance that ships the driver, or install it first."

  local name driver
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  log "GPU: ${name} (driver ${driver})"

  # Isaac Sim renders through RTX, so it needs ray-tracing cores. Pascal (P100,
  # GTX 10xx) and Volta (V100) have none and will fail at the renderer, not at
  # install time, which is a slow and confusing way to find out.
  case "${name}" in
    *P100*|*K80*|*V100*|*GTX\ 10*|*GTX\ 9*)
      die "${name} has no RT cores. Isaac Sim's renderer requires them, and this
           project needs camera output, so physics-only is not enough. Rent an
           RTX card instead: see docs/setup-cloud.md." ;;
    *T4*)
      warn "T4 has RT cores but is not on NVIDIA's supported list for Isaac Sim.
            It may work. Run scripts/check_setup.py --isaac before committing to
            a long collection run." ;;
  esac
}

check_disk() {
  local avail
  avail="$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')"
  log "free disk in \$HOME: ${avail} GB"
  [ "${avail}" -ge "${REQUIRED_DISK_GB}" ] || die \
    "Isaac Sim needs about 30 GB installed plus room for the dataset;
     ${REQUIRED_DISK_GB} GB free is the minimum. Found ${avail} GB.
     Many rented boxes put the big volume somewhere other than \$HOME:
     set ISAACLAB_DIR and the dataset --out to that volume."
}

check_vulkan() {
  if command -v vulkaninfo >/dev/null 2>&1; then
    if vulkaninfo --summary >/dev/null 2>&1; then
      log "Vulkan: OK"
    else
      warn "vulkaninfo failed. Isaac Sim's renderer needs Vulkan; if
            check_setup.py --isaac fails at the camera stage, this is why."
    fi
  else
    warn "vulkaninfo not installed, so Vulkan could not be verified."
  fi
}

install_system_packages() {
  log "installing system packages"
  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    build-essential git curl wget unzip \
    libglu1-mesa libxi6 libxrandr2 libxcursor1 libxinerama1 \
    libgl1 libegl1 libxkbcommon-x11-0 vulkan-tools
}

install_conda() {
  if command -v conda >/dev/null 2>&1; then
    log "conda already present"
    return
  fi
  log "installing miniconda"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  rm -f /tmp/miniconda.sh
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
}

install_isaac() {
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if ! conda env list | grep -q '^env_isaaclab'; then
    log "creating the env_isaaclab environment (python ${PYTHON_VERSION})"
    conda create -y -n env_isaaclab "python=${PYTHON_VERSION}"
  fi
  conda activate env_isaaclab

  log "installing Isaac Sim ${ISAACSIM_VERSION}"
  pip install --upgrade pip
  pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
  pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
    --extra-index-url https://pypi.nvidia.com

  if [ ! -d "${ISAACLAB_DIR}" ]; then
    log "cloning Isaac Lab ${ISAACLAB_REF} into ${ISAACLAB_DIR}"
    git clone --depth 1 --branch "${ISAACLAB_REF}" \
      https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
  fi
  log "installing Isaac Lab"
  (cd "${ISAACLAB_DIR}" && ./isaaclab.sh --install)
}

install_project() {
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate env_isaaclab
  log "installing isaacgrasp"
  pip install -e "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)[dev]"
}

main() {
  check_gpu
  check_disk
  check_vulkan

  if [ "${1:-}" = "--check" ]; then
    log "check only, nothing installed"
    return 0
  fi

  install_system_packages
  install_conda
  install_isaac
  install_project

  cat <<'NEXT'

[setup] Done. Next, in order, and read the output of each:

  conda activate env_isaaclab
  python scripts/check_setup.py --isaac --num-envs 16

The last command is the gate. It exercises the Isaac backend in six stages and
names the one that fails. Do not start a long collection run until it passes:
the object geometry stage in particular checks something that has never been
verified anywhere, and a failure there is silent in the data rather than loud.

NEXT
}

main "$@"
