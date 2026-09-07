"""The setup gate, which is the thing that protects a GPU rental.

``scripts/check_setup.py --isaac`` is the first command anyone runs on a rented
machine, and it decides whether hours of collection are worth starting. It had
never been executed with its Isaac stages enabled, so a bug in the gate would
have been discovered by paying for an instance to find it.

The Isaac stages here run against the stub API from ``tests/fakes``. As
everywhere else, that does not validate the real Isaac Lab. What it validates is
that each stage detects the failure it exists to detect, and reports it rather
than raising something unreadable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fakes.isaac import install_fake_isaac, uninstall_fake_isaac

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "check_setup", ROOT / "scripts" / "check_setup.py")
check_setup = importlib.util.module_from_spec(spec)
sys.modules["check_setup"] = check_setup
spec.loader.exec_module(check_setup)

NUM_ENVS = 3


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #


def test_stages_stop_at_the_first_failure():
    """Later stages depend on earlier ones, so continuing would report noise."""
    stages = check_setup.Stages()
    stages.run("first", lambda: "fine")
    stages.run("second", lambda: (_ for _ in ()).throw(RuntimeError("the real problem")))
    stages.run("third", lambda: "never reached")

    names = [name for name, _status, _detail in stages.results]
    statuses = [status for _name, status, _detail in stages.results]
    assert names == ["first", "second", "third"]
    assert statuses == [check_setup.PASS, check_setup.FAIL, check_setup.SKIP]
    assert stages.failed is True
    assert "the real problem" in stages.results[1][2]


def test_a_clean_run_reports_no_failure():
    stages = check_setup.Stages()
    for name in ("a", "b", "c"):
        stages.run(name, lambda: "ok")
    assert stages.failed is False
    assert all(status == check_setup.PASS for _n, status, _d in stages.results)


def test_parity_stage_reads_the_real_manifest():
    assert "constants match" in check_setup.check_parity()


def test_imports_stage_names_the_versions():
    detail = check_setup.check_imports()
    assert "isaacgrasp" in detail and "simgrasp" in detail and "torch" in detail


# --------------------------------------------------------------------------- #
# The Isaac stages
# --------------------------------------------------------------------------- #


@pytest.fixture
def isaac():
    """Fake Isaac installed, and the gate's module-level handle cleaned up after."""
    install_fake_isaac(NUM_ENVS)
    try:
        yield
    finally:
        backend = check_setup._ISAAC.pop("backend", None)
        if backend is not None:
            backend.close()
        check_setup._ISAAC.clear()
        uninstall_fake_isaac()


def _launch():
    # cpu, because the gate defaults to cuda:0 and there is no GPU here. That the
    # device is settable at all is the point of the --device flag.
    check_setup.check_isaac_launch(NUM_ENVS, device="cpu")
    return check_setup._ISAAC["backend"]


def test_import_and_launch_stages(isaac):
    assert "isaaclab" in check_setup.check_isaac_import()
    detail = check_setup.check_isaac_launch(NUM_ENVS, device="cpu")
    assert f"{NUM_ENVS} environments" in detail
    assert check_setup._ISAAC["backend"].batch_size == NUM_ENVS


def test_reset_stage_reports_the_categories(isaac):
    _launch()
    detail = check_setup.check_isaac_reset()
    assert f"{NUM_ENVS} scenes" in detail
    assert "categories" in detail


def test_camera_stage_catches_a_flat_height_map(isaac):
    """The signature of headless rendering being off: right shapes, wrong values."""
    _launch()
    check_setup.check_isaac_reset()
    with pytest.raises(RuntimeError, match="flat"):
        check_setup.check_isaac_camera()


def test_camera_stage_passes_when_an_object_is_visible(isaac):
    backend = _launch()
    backend._scene["camera"].height_source = lambda: [
        view.state.spec.top_z if view.state else 0.0 for view in backend.views()]
    check_setup.check_isaac_reset()
    detail = check_setup.check_isaac_camera()
    assert "above the table" in detail


def test_geometry_stage_catches_a_collision_shape_that_never_updates(isaac):
    """The highest-risk stage. A flat camera means no shape ever changes."""
    _launch()
    with pytest.raises(RuntimeError, match="did not change"):
        check_setup.check_isaac_geometry()


def test_geometry_stage_passes_when_shapes_actually_change(isaac):
    backend = _launch()
    backend._scene["camera"].height_source = lambda: [
        view.state.spec.top_z if view.state else 0.0 for view in backend.views()]
    detail = check_setup.check_isaac_geometry()
    assert "changed shape between resets" in detail


def test_geometry_stage_names_the_documented_fallback(isaac):
    """A failure here has to point somewhere, or it costs a rental to diagnose."""
    _launch()
    with pytest.raises(RuntimeError) as excinfo:
        check_setup.check_isaac_geometry()
    message = str(excinfo.value)
    assert "docs/design.md" in message
    assert "PhysX" in message


def test_execute_stage_runs_a_batch_of_heuristic_grasps(isaac):
    backend = _launch()
    backend._scene["camera"].height_source = lambda: [
        view.state.spec.top_z if view.state else 0.0 for view in backend.views()]
    detail = check_setup.check_isaac_execute()
    assert "heuristic grasps succeeded" in detail
    assert "outcomes" in detail


def test_execute_stage_catches_an_ik_controller_that_never_converges(isaac, monkeypatch):
    backend = _launch()
    backend._scene["camera"].height_source = lambda: [
        view.state.spec.top_z if view.state else 0.0 for view in backend.views()]
    import numpy as np

    monkeypatch.setattr(backend, "_move_to_pose",
                        lambda command, seconds: np.ones(backend.batch_size, dtype=bool))
    with pytest.raises(RuntimeError, match="ik_failed"):
        check_setup.check_isaac_execute()
