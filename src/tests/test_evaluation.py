"""
End-to-end pipeline evaluation.

Each test runs the full fit -> batched-predict pipeline against a real parquet
scenario and compares alert timestamps against the ground-truth incident windows
from the private incident-label YAML.

Scoring intent
--------------
* No-incident scenarios (TN): alert must NOT fire  -> precision signal
* Incident scenarios (TP):    at least one alert must land inside the window
                               -> recall signal
* Multi-incident scenarios:   the engine must re-arm after normal periods
                               so EVERY incident window gets at least one alert
                               -> advanced recall
"""

import pytest

from analysis.evaluation import summarize_inference_test_metrics
from tests.conftest import (
    DATA_AVAILABLE,
    NO_INCIDENT_IDS,
    alert_hits_window,
    any_alert_in_incidents,
)
from tests.conftest import MULTI_INCIDENT_IDS as _MULTI_INCIDENT_IDS
from tests.conftest import SINGLE_INCIDENT_IDS as _SINGLE_INCIDENT_IDS

pytestmark = pytest.mark.skipif(
    not DATA_AVAILABLE, reason="private benchmark data not present"
)

# -- Helper --------------------------------------------------------------------


def _incident_windows(incidents: dict, sid: int) -> list[dict]:
    """Return valid incident windows for scenario *sid* (skips inverted ranges)."""
    return [
        inc
        for inc in (incidents.get(sid) or [])
        if inc.get("start") and inc.get("end") and inc["start"] < inc["end"]
    ]


# -- No-incident scenarios: no false positives ---------------------------------


@pytest.mark.parametrize("scenario_id", NO_INCIDENT_IDS)
def test_no_alert_when_no_incident(scenario_alerts, scenario_id):
    """
    Scenarios with zero incidents must produce zero alerts.
    Any alert here is a false positive - degrading precision.
    """
    alerts = scenario_alerts[scenario_id]
    assert (
        alerts == []
    ), f"Scenario {scenario_id} has no incidents but got alerts at: {alerts}"


# -- Single-incident scenarios: alert fires within window ---------------------


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


# -- Multi-incident scenarios --------------------------------------------------


@pytest.mark.parametrize("scenario_id", _MULTI_INCIDENT_IDS)
def test_at_least_one_alert_in_any_incident_window_multi(
    scenario_alerts, incidents, scenario_id
):
    """
    Scenarios with multiple incidents must produce at least one alert that
    falls within any of the incident windows.

    Passes when at least one alert lands in any incident window.
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

    Requires a properly re-arming AlertEngine that resets after a normal period
    and can detect subsequent incidents.
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


# -- Aggregate quality metrics -------------------------------------------------


_METRIC_THRESHOLD = 0.85


@pytest.fixture(scope="session")
def metrics_summary(scenario_alerts, incidents) -> dict:
    """Scenario-level precision/recall/F1 from the canonical evaluator.

    Delegates to ``summarize_inference_test_metrics`` - the same function the
    analysis notebooks use - so the test and the notebooks can never disagree
    on how precision / recall / F1 are scored.
    """
    report = summarize_inference_test_metrics(scenario_alerts, incidents)
    return report["summary"]


def test_precision_above_threshold(metrics_summary):
    """Precision = TP / (TP + FP) across all scenarios must clear the gate.

    A model that fires alerts indiscriminately will fail this test.
    """
    precision = metrics_summary["precision"]
    assert precision >= _METRIC_THRESHOLD, (
        f"Precision {precision:.0%} is below {_METRIC_THRESHOLD:.0%} "
        f"(TP={metrics_summary['tp']}, FP={metrics_summary['fp']})"
    )


def test_recall_above_threshold(metrics_summary):
    """Recall = TP / (TP + FN) across incident scenarios must clear the gate."""
    recall = metrics_summary["recall"]
    assert recall >= _METRIC_THRESHOLD, (
        f"Recall {recall:.0%} is below {_METRIC_THRESHOLD:.0%} "
        f"(TP={metrics_summary['tp']}, FN={metrics_summary['fn']})"
    )


def test_f1_above_threshold(metrics_summary):
    """F1 = harmonic mean of precision and recall must clear the gate.

    Balances avoiding false positives with detecting real incidents.
    """
    f1 = metrics_summary["f1"]
    assert f1 >= _METRIC_THRESHOLD, (
        f"F1 {f1:.2f} is below {_METRIC_THRESHOLD:.2f} "
        f"(precision={metrics_summary['precision']:.0%}, "
        f"recall={metrics_summary['recall']:.0%}, "
        f"TP={metrics_summary['tp']}, FP={metrics_summary['fp']}, FN={metrics_summary['fn']})"
    )
