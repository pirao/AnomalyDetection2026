"""
API contract tests — MLE evaluation track.

These tests verify that the HTTP API exposes the correct contracts:
correct status codes, response shapes, field types, and state semantics.
A passing candidate must not alter the endpoint signatures or response models.

All fit/predict payloads are drawn from the real evaluation dataset (data/).
Reference scenario: 8 (confirmed TP baseline — training on healthy data then
presenting incident-window data reliably triggers an alert).
"""

from __future__ import annotations

import pytest

from tests.conftest import (
    _REF_ANOMALOUS,
    _REF_FIT,
    _REF_NORMAL,
    df_to_data_points,
    load_scenario,
)

# ── /health ───────────────────────────────────────────────────────────────────


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── /fit ──────────────────────────────────────────────────────────────────────


def test_fit_returns_200_and_trained_status(client):
    r = client.post("/fit", json={"sensor_id": "s1", "data": _REF_FIT})
    assert r.status_code == 200
    body = r.json()
    assert body["sensor_id"] == "s1"
    assert body["status"] == "trained"


def test_fit_empty_data_returns_422(client):
    r = client.post("/fit", json={"sensor_id": "s1", "data": []})
    assert r.status_code == 422


def test_fit_missing_sensor_id_returns_422(client):
    r = client.post("/fit", json={"data": _REF_FIT[:10]})
    assert r.status_code == 422


def test_fit_missing_data_field_returns_422(client):
    r = client.post("/fit", json={"sensor_id": "s1"})
    assert r.status_code == 422


# ── /predict ──────────────────────────────────────────────────────────────────


def test_predict_before_fit_returns_404(client):
    r = client.post("/predict", json={"sensor_id": "not_trained", "data": _REF_NORMAL})
    assert r.status_code == 404


def test_predict_empty_data_returns_422(client):
    client.post("/fit", json={"sensor_id": "s2", "data": _REF_FIT})
    r = client.post("/predict", json={"sensor_id": "s2", "data": []})
    assert r.status_code == 422


def test_predict_returns_200(client):
    client.post("/fit", json={"sensor_id": "s3", "data": _REF_FIT})
    r = client.post("/predict", json={"sensor_id": "s3", "data": _REF_NORMAL})
    assert r.status_code == 200


def test_predict_response_has_required_fields(client):
    client.post("/fit", json={"sensor_id": "s4", "data": _REF_FIT})
    r = client.post("/predict", json={"sensor_id": "s4", "data": _REF_NORMAL})
    body = r.json()
    assert "sensor_id" in body
    assert "anomaly" in body
    assert "alert" in body
    assert "timestamp" in body


def test_predict_response_field_types(client):
    client.post("/fit", json={"sensor_id": "s5", "data": _REF_FIT})
    r = client.post("/predict", json={"sensor_id": "s5", "data": _REF_NORMAL})
    body = r.json()
    assert isinstance(body["sensor_id"], str)
    assert isinstance(body["anomaly"], bool)
    assert isinstance(body["alert"], bool)
    assert isinstance(body["timestamp"], str)


def test_predict_sensor_id_echoed_in_response(client):
    sid = "unique_sensor_xyz"
    client.post("/fit", json={"sensor_id": sid, "data": _REF_FIT})
    r = client.post("/predict", json={"sensor_id": sid, "data": _REF_NORMAL})
    assert r.json()["sensor_id"] == sid


def test_predict_timestamp_equals_last_datapoint(client):
    """The response `timestamp` must be the last DataPoint's timestamp."""
    client.post("/fit", json={"sensor_id": "s6", "data": _REF_FIT})
    batch = _REF_NORMAL[:5]
    expected_ts = batch[-1]["timestamp"]
    r = client.post("/predict", json={"sensor_id": "s6", "data": batch})
    assert r.json()["timestamp"] == expected_ts


def test_predict_single_point_batch(client):
    """A single-point batch is valid and should return a result."""
    client.post("/fit", json={"sensor_id": "s7", "data": _REF_FIT})
    r = client.post("/predict", json={"sensor_id": "s7", "data": _REF_NORMAL[:1]})
    assert r.status_code == 200


def test_predict_missing_sensor_id_returns_422(client):
    r = client.post("/predict", json={"data": _REF_NORMAL})
    assert r.status_code == 422


# ── State semantics ───────────────────────────────────────────────────────────


def test_retrain_resets_alert_state(client):
    """Retraining must reset the alert engine so an alert can fire again."""
    sid = "retrain_test"
    client.post("/fit", json={"sensor_id": sid, "data": _REF_FIT})

    r1 = client.post("/predict", json={"sensor_id": sid, "data": _REF_ANOMALOUS})
    assert r1.json()["alert"] is True, "Expected alert on first anomalous batch"

    # Retrain — must reset alert state
    client.post("/fit", json={"sensor_id": sid, "data": _REF_FIT})

    r2 = client.post("/predict", json={"sensor_id": sid, "data": _REF_ANOMALOUS})
    assert r2.json()["alert"] is True, "Expected alert to fire again after retrain"


def test_multiple_sensors_are_independent(client):
    """Alerting on sensor A must not affect sensor B."""
    for sid in ["sA", "sB"]:
        client.post("/fit", json={"sensor_id": sid, "data": _REF_FIT})

    # Fire alert on sA
    client.post("/predict", json={"sensor_id": "sA", "data": _REF_ANOMALOUS})

    # sB should still fire independently
    r = client.post("/predict", json={"sensor_id": "sB", "data": _REF_ANOMALOUS})
    assert r.json()["alert"] is True, "Sensor B should alert independently from A"


def test_sensor_a_state_not_leaked_to_new_sensor(client):
    """Predictions for a trained sensor must not create state for a different sensor."""
    client.post("/fit", json={"sensor_id": "existing", "data": _REF_FIT})
    client.post("/predict", json={"sensor_id": "existing", "data": _REF_NORMAL})

    r = client.post("/predict", json={"sensor_id": "ghost", "data": _REF_NORMAL})
    assert r.status_code == 404


def test_anomaly_flag_and_alert_flag_are_consistent(client):
    """When alert=True the anomaly flag must also be True."""
    client.post("/fit", json={"sensor_id": "consistency", "data": _REF_FIT})
    r = client.post(
        "/predict", json={"sensor_id": "consistency", "data": _REF_ANOMALOUS}
    )
    body = r.json()
    if body["alert"]:
        assert body["anomaly"] is True, "alert=True requires anomaly=True"


def test_normal_batch_does_not_alert(client):
    """Normal data (pre-incident predict data) must not trigger an alert."""
    client.post("/fit", json={"sensor_id": "s_normal", "data": _REF_FIT})
    r = client.post("/predict", json={"sensor_id": "s_normal", "data": _REF_NORMAL})
    assert r.json()["alert"] is False


def test_anomalous_batch_triggers_alert(client):
    """Data from within the reference incident window must trigger an alert."""
    client.post("/fit", json={"sensor_id": "s_anomaly", "data": _REF_FIT})
    r = client.post("/predict", json={"sensor_id": "s_anomaly", "data": _REF_ANOMALOUS})
    assert r.json()["alert"] is True


def test_second_anomalous_batch_does_not_re_alert_without_reset(client):
    """
    After an alert fires, a second anomalous batch on the same sensor should
    NOT fire another alert until the state is explicitly reset.
    """
    client.post("/fit", json={"sensor_id": "s_lock", "data": _REF_FIT})

    r1 = client.post("/predict", json={"sensor_id": "s_lock", "data": _REF_ANOMALOUS})
    assert r1.json()["alert"] is True, "First anomaly should alert"

    r2 = client.post("/predict", json={"sensor_id": "s_lock", "data": _REF_ANOMALOUS})
    assert r2.json()["alert"] is False, "Immediate second anomaly must not re-alert"
