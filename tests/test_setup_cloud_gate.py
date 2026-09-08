"""The provisioning gate, which is what stands between a student and a bad rental.

``scripts/setup_cloud.sh --check`` is the very first command run on a rented
box. It changes nothing and reports whether the machine is worth continuing
with. Its job is to fail in seconds, for a stated reason, instead of failing
twenty minutes and thirty gigabytes later inside Isaac Sim's renderer.

The one decision that actually costs money is the GPU check. Isaac Sim renders
through RTX and needs ray-tracing cores, and several of the cards a marketplace
will happily rent you do not have any. The A100 is the trap: it is expensive,
it is everywhere, and being a compute die it will install perfectly and then
fail at the renderer.

Nothing here validates Isaac Sim. What it validates is that the gate reaches
its verdict, and reaches the right one, by running the script rather than by
reading it. The GPU, disk and Vulkan probes are stubbed on PATH so the result
does not depend on whatever machine the tests happen to run on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "setup_cloud.sh"

# Cards with no RT cores. Pascal and Volta never had them; A100 and A30 are the
# GA100 compute die and H100 and H200 the GH100, and neither carries any either.
NO_RT_CORES = [
    "Tesla P100-PCIE-16GB",
    "Tesla K80",
    "Tesla V100-SXM2-16GB",
    "NVIDIA GeForce GTX 1080 Ti",
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A30",
    "NVIDIA H100 PCIe",
    "NVIDIA H200",
]

# Cards that should get through the GPU check.
HAS_RT_CORES = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
]


def _sandbox(tmp_path: Path, gpu_name: str | None = "NVIDIA GeForce RTX 4090",
             free_gb: int = 100, vulkan: bool = True) -> dict[str, str]:
    """A PATH holding nothing but what the script is allowed to see.

    Sealed rather than merely prepended: a real nvidia-smi further down PATH
    would otherwise decide the answer, and on a GPU machine the test would quietly
    stop testing anything. ``gpu_name=None`` leaves no nvidia-smi at all.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)

    # The real utilities the script calls, and nothing else.
    for tool in ("bash", "head", "tail", "tr"):
        real = shutil.which(tool)
        assert real, f"{tool} is needed to run the script under test"
        (bin_dir / tool).symlink_to(real)

    # The three probes whose answers this test is choosing.
    if gpu_name is not None:
        (bin_dir / "nvidia-smi").write_text(
            f'#!/bin/sh\ncase "$*" in\n'
            f'  *name*) echo "{gpu_name}" ;;\n'
            f'  *driver_version*) echo "550.54.15" ;;\n'
            f'esac\n')
    (bin_dir / "df").write_text(f'#!/bin/sh\necho avail\necho {free_gb}G\n')
    if vulkan:
        (bin_dir / "vulkaninfo").write_text("#!/bin/sh\nexit 0\n")
    else:
        (bin_dir / "vulkaninfo").write_text("#!/bin/sh\nexit 1\n")

    for stub in bin_dir.iterdir():
        if not stub.is_symlink():
            stub.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = str(bin_dir)
    env["HOME"] = str(tmp_path)
    return env


def _run_check(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(Path(env["PATH"]) / "bash"), str(SCRIPT), "--check"],
        env=env, capture_output=True, text=True, timeout=60)


# --------------------------------------------------------------------------- #
# The GPU verdict
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gpu", NO_RT_CORES)
def test_a_card_without_rt_cores_is_refused(tmp_path, gpu):
    """Refused in seconds, by name, rather than at the renderer hours later."""
    result = _run_check(_sandbox(tmp_path, gpu))

    assert result.returncode != 0, f"{gpu} was accepted"
    assert "no RT cores" in result.stderr
    assert gpu in result.stderr


@pytest.mark.parametrize("gpu", HAS_RT_CORES)
def test_a_card_with_rt_cores_gets_through(tmp_path, gpu):
    """The whole point of --check is that it says yes as well as no."""
    result = _run_check(_sandbox(tmp_path, gpu))

    assert result.returncode == 0, result.stderr
    assert "no RT cores" not in result.stderr
    assert "check only, nothing installed" in result.stdout


def test_a_t4_is_warned_about_but_not_refused(tmp_path):
    """Turing has RT cores, so a T4 may work. NVIDIA does not list it, so say so."""
    result = _run_check(_sandbox(tmp_path, "Tesla T4"))

    assert result.returncode == 0, result.stderr
    assert "T4" in result.stdout + result.stderr
    assert "check_setup.py --isaac" in result.stdout + result.stderr


def test_the_a100_is_refused_even_though_it_is_the_expensive_one(tmp_path):
    """Named separately because it is the mistake this check exists to stop.

    An A100 passes every intuition a renter has: it is the most costly card on
    the page, it installs Isaac Sim without complaint, and it fails only once
    the renderer asks for ray tracing.
    """
    result = _run_check(_sandbox(tmp_path, "NVIDIA A100-SXM4-40GB"))

    assert result.returncode != 0
    assert "no RT cores" in result.stderr


# --------------------------------------------------------------------------- #
# Disk and Vulkan
# --------------------------------------------------------------------------- #


def test_too_little_disk_is_refused_with_the_number_found(tmp_path):
    """Isaac Sim is about 30 GB, and running out mid-download wastes the rental."""
    result = _run_check(_sandbox(tmp_path, "NVIDIA GeForce RTX 4090", free_gb=12))

    assert result.returncode != 0
    assert "12" in result.stderr


def test_the_gpu_is_checked_before_the_disk(tmp_path):
    """A refused card should not also be told about its disk: one reason, first."""
    result = _run_check(
        _sandbox(tmp_path, "NVIDIA A100-SXM4-40GB", free_gb=12))

    assert "no RT cores" in result.stderr
    assert "Isaac Sim needs about 30 GB" not in result.stderr


def test_a_broken_vulkan_warns_rather_than_refusing(tmp_path):
    """It is often a headless driver detail, and fixable without a new rental."""
    result = _run_check(
        _sandbox(tmp_path, "NVIDIA GeForce RTX 4090", vulkan=False))

    assert result.returncode == 0, result.stderr
    assert "Vulkan" in result.stdout + result.stderr


def test_no_driver_at_all_is_refused(tmp_path):
    """nvidia-smi missing means the box cannot run Isaac Sim at all."""
    result = _run_check(_sandbox(tmp_path, gpu_name=None))

    assert result.returncode != 0
    assert "nvidia-smi not found" in result.stderr


# --------------------------------------------------------------------------- #
# The check does what it says
# --------------------------------------------------------------------------- #


def test_check_installs_nothing(tmp_path):
    """--check is advertised as changing nothing, so it must not reach an install."""
    result = _run_check(_sandbox(tmp_path, "NVIDIA GeForce RTX 4090"))

    combined = result.stdout + result.stderr
    assert "installing" not in combined
    assert "conda" not in combined.lower()


def test_the_documented_refusal_list_matches_the_script():
    """docs/setup-cloud.md tells the reader which cards are refused outright.

    The doc and the case statement drifting apart is exactly the kind of thing
    nobody notices until someone rents the wrong card on the doc's advice.
    """
    script = SCRIPT.read_text()
    for card in ("A100", "V100", "P100", "H100"):
        assert f"*{card}*" in script, f"{card} is documented as refused but is not"
