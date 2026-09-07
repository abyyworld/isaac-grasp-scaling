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
