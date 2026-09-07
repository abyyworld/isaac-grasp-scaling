"""The parity mechanism has to actually fail when the upstream definitions move.

A check that cannot fail is worse than no check: it reads as reassurance in the
README while guaranteeing nothing.
"""

from __future__ import annotations

import pytest

from isaacgrasp import parity


def test_the_installed_simgrasp_matches():
    report = parity.check_parity()
    assert report.ok, report.describe()
    assert len(report.matched) == len(parity.EXPECTED)


def test_require_parity_returns_the_report():
    assert parity.require_parity().ok


def test_a_changed_constant_is_detected(monkeypatch):
    monkeypatch.setitem(parity.EXPECTED, "LIFT_SUCCESS_THRESHOLD", 0.05)
    report = parity.check_parity()
    assert not report.ok
    assert "LIFT_SUCCESS_THRESHOLD" in report.mismatched
    assert "0.05" in report.describe()
    with pytest.raises(RuntimeError, match="PARITY FAILURE"):
        parity.require_parity()


def test_a_missing_constant_is_detected(monkeypatch):
    monkeypatch.setitem(parity.EXPECTED, "NOT_A_REAL_CONSTANT", 1)
    report = parity.check_parity()
    assert not report.ok
    assert "NOT_A_REAL_CONSTANT" in report.missing


def test_manifest_is_json_serialisable():
    import json

    payload = parity.manifest()
    assert payload["parity_ok"] is True
    assert json.loads(json.dumps(payload))["constants"]["ANGLE_BINS"] == 12


def test_the_success_criterion_is_covered():
    """The thresholds that define success must be among the checked constants."""
    for name in ("LIFT_SUCCESS_THRESHOLD", "LIFT_DISTANCE", "SAFE_GRASP_WIDTH",
                 "ANGLE_BINS", "EVAL_EPISODE_OFFSET"):
        assert name in parity.EXPECTED


def test_the_arm_base_is_level_with_the_table_top():
    """The plinth exists so the base and the table top sit at the same height.

    The Isaac scene builds a plinth from PLINTH_HALF and mounts the arm at
    TABLE_HEIGHT. If those two ever disagree the arm floats or sinks, and every
    grasp is offset in the robot's own frame by the difference: reachability
    changes, and nothing raises.
    """
    from simgrasp.scene import PLINTH_HALF, TABLE_HEIGHT

    assert 2.0 * PLINTH_HALF[2] == pytest.approx(TABLE_HEIGHT), (
        "the plinth is no longer the height of the table top")


def test_the_tcp_offset_is_inside_the_fingertip_pads():
    """Grasps are commanded at the pad centre, not the hand body origin.

    103.4 mm is the pad centre along the hand's +z. Commanding the hand body to
    the grasp position instead would put the gripper a hand's length too low.
    """
    from simgrasp.scene import TCP_OFFSET_Z

    assert 0.096 <= TCP_OFFSET_Z <= 0.108
