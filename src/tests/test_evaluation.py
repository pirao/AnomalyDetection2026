"""
End-to-end pipeline evaluation — DS evaluation track.

Each test runs the full fit → batched-predict pipeline against a real parquet
scenario and compares alert timestamps against the ground-truth incident windows
from labels/incidents.yaml.

Scoring intent
--------------
* No-incident scenarios (TN): alert must NOT fire  → precision signal
* Incident scenarios (TP):    at least one alert must land inside the window
                               → recall signal
* Multi-incident scenarios:   the engine must re-arm after normal periods
                               so EVERY incident window gets at least one alert
                               → advanced recall (requires fixing AlertEngine)

The baseline implementation is expected to pass some but not all tests.
A strong candidate will pass all tests by improving the model and alert engine.
"""

import pytest

from tests.conftest import INCIDENT_IDS
from tests.conftest import MULTI_INCIDENT_IDS as _MULTI_INCIDENT_IDS
from tests.conftest import NO_INCIDENT_IDS
from tests.conftest import SINGLE_INCIDENT_IDS as _SINGLE_INCIDENT_IDS
from tests.conftest import alert_hits_window, any_alert_in_incidents

# ── Helper ────────────────────────────────────────────────────────────────────


def _incident_windows(incidents: dict, sid: int) -> list[dict]:
    """Return valid incident windows for scenario *sid* (skips inverted ranges)."""
    return [
        inc
        for inc in (incidents.get(sid) or [])
        if inc.get("start") and inc.get("end") and inc["start"] < inc["end"]
    ]


# ── No-incident scenarios: no false positives ─────────────────────────────────


@pytest.mark.parametrize("scenario_id", NO_INCIDENT_IDS)
def test_no_alert_when_no_incident(scenario_alerts, scenario_id):
    """
    Scenarios with zero incidents must produce zero alerts.
    Any alert here is a false positive — degrading precision.
    """
    alerts = scenario_alerts[scenario_id]
    assert (
        alerts == []
    ), f"Scenario {scenario_id} has no incidents but got alerts at: {alerts}"


# ── Single-incident scenarios: alert fires within window ─────────────────────


@pytest.mark.parametrize("scenario_id", _SINGLE_INCIDENT_IDS)
def test_alert_fires_in_incident_window_single(scenario_alerts, incidents, scenario_id):
    """
    Scenarios with exactly one incident must produce at least one alert that
    falls within (or near) that incident window.
    """
    windows = _incident_windows(incidents, scenario_id)
    if not windows:
        pytest.skip(f"Scenario {scenario_id} has no valid incident windows in labels")

    alerts = scenario_alerts[scenario_id]
    assert any_alert_in_incidents(
        alerts, windows
    ), f"Scenario {scenario_id}: expected alert inside {windows}, got {alerts}"


# ── Multi-incident scenarios ──────────────────────────────────────────────────


@pytest.mark.parametrize("scenario_id", _MULTI_INCIDENT_IDS)
def test_at_least_one_alert_in_any_incident_window_multi(
    scenario_alerts, incidents, scenario_id
):
    """
    Scenarios with multiple incidents must produce at least one alert that
    falls within any of the incident windows.

    Passes even with the baseline engine (one alert is enough).
    """
    windows = _incident_windows(incidents, scenario_id)
    if not windows:
        pytest.skip(f"Scenario {scenario_id} has no valid incident windows")

    alerts = scenario_alerts[scenario_id]
    assert any_alert_in_incidents(
        alerts, windows
    ), f"Scenario {scenario_id}: expected alert in one of {windows}, got {alerts}"


@pytest.mark.parametrize("scenario_id", _MULTI_INCIDENT_IDS)
def test_every_incident_window_gets_an_alert(scenario_alerts, incidents, scenario_id):
    """
    Each distinct incident window should receive at least one alert.

    This test requires a properly re-arming AlertEngine (the baseline fails it).
    Candidates must fix the engine so it re-arms after a normal period and
    can detect subsequent incidents.
    """
    windows = _incident_windows(incidents, scenario_id)
    if not windows:
        pytest.skip(f"Scenario {scenario_id} has no valid incident windows")

    alerts = scenario_alerts[scenario_id]
    missed = [
        inc
        for inc in windows
        if not any(alert_hits_window(a, inc["start"], inc["end"]) for a in alerts)
    ]
    assert (
        not missed
    ), f"Scenario {scenario_id}: the following incident windows had no alert: {missed}"


# ── Aggregate quality metrics ─────────────────────────────────────────────────


def test_precision_above_threshold(scenario_alerts, incidents):
    """
    Precision = TP / (TP + FP) across all scenarios must exceed 50 %.

    A model that fires alerts indiscriminately will fail this test.
    """
    tp = fp = 0
    for sid, alerts in scenario_alerts.items():
        windows = _incident_windows(incidents, sid)
        if alerts:
            if any_alert_in_incidents(alerts, windows):
                tp += 1
            else:
                fp += 1

    total = tp + fp
    precision = tp / total if total > 0 else 0.0
    assert (
        precision >= 0.50
    ), f"Precision {precision:.0%} is below 50% (TP={tp}, FP={fp})"


def test_recall_above_threshold(scenario_alerts, incidents):
    """
    Recall = TP / (TP + FN) across incident scenarios must exceed 30 %.

    The baseline recall is ~27 %; candidates are expected to beat it.
    Scoring weight: important — a strong candidate will push this above 70 %.
    """
    tp = fn = 0
    for sid in INCIDENT_IDS:
        windows = _incident_windows(incidents, sid)
        if not windows:
            continue
        alerts = scenario_alerts[sid]
        if any_alert_in_incidents(alerts, windows):
            tp += 1
        else:
            fn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    assert recall >= 0.30, f"Recall {recall:.0%} is below 30% (TP={tp}, FN={fn})"


def test_f1_above_threshold(scenario_alerts, incidents):
    """
    F1 = harmonic mean of precision and recall must exceed 0.35.

    Balances avoiding false positives with detecting real incidents.
    Candidates improving both model quality and alert logic will exceed 0.70.
    """
    tp = fp = fn = 0
    for sid, alerts in scenario_alerts.items():
        windows = _incident_windows(incidents, sid)
        has_incident = bool(windows)
        has_alert_in_window = any_alert_in_incidents(alerts, windows)

        if has_alert_in_window:
            tp += 1
        elif alerts and not has_incident:
            fp += 1
        elif has_incident and not has_alert_in_window:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    assert f1 >= 0.35, (
        f"F1 {f1:.2f} is below 0.35 "
        f"(precision={precision:.0%}, recall={recall:.0%}, TP={tp}, FP={fp}, FN={fn})"
    )
