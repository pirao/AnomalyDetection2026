"""
Performance and concurrency tests — MLE evaluation track.

Verifies that the API correctly handles:
- Concurrent requests from multiple threads (shared in-memory state safety)
- High-throughput sequential batches without degrading response
- Large payload handling

All payloads are drawn from the real evaluation dataset (data/).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi.testclient import TestClient

from sample_processing.api.main import app
from tests.conftest import (
    _REF_ANOMALOUS,
    _REF_FIT,
    _REF_NORMAL,
    BATCH_SIZE,
    df_to_data_points,
    load_scenario,
)

# ── Throughput ────────────────────────────────────────────────────────────────


def test_fit_then_many_predict_batches(client):
    """API should handle 50 sequential predict calls without error."""
    client.post("/fit", json={"sensor_id": "perf_s1", "data": _REF_FIT})
    for _ in range(50):
        r = client.post("/predict", json={"sensor_id": "perf_s1", "data": _REF_NORMAL})
        assert r.status_code == 200


def test_large_fit_payload(client):
    """Fitting on a full real dataset (~700 rows) should succeed."""
    # scenario 8 fit has 706 rows — a realistic fit payload
    r = client.post("/fit", json={"sensor_id": "perf_large", "data": _REF_FIT})
    assert r.status_code == 200
    assert r.json()["status"] == "trained"


def test_large_predict_payload(client):
    """Predicting on a large batch (200 rows) must return 200."""
    client.post("/fit", json={"sensor_id": "perf_large_pred", "data": _REF_FIT})
    fit_df, pred_df = load_scenario(8)
    large_batch = df_to_data_points(pred_df.iloc[:200])
    r = client.post(
        "/predict", json={"sensor_id": "perf_large_pred", "data": large_batch}
    )
    assert r.status_code == 200


def test_sequential_batches_across_full_scenario(client):
    """Running the full batched-predict pipeline on one scenario must not error."""
    fit_df, pred_df = load_scenario(8)
    client.post(
        "/fit", json={"sensor_id": "perf_full", "data": df_to_data_points(fit_df)}
    )
    points = df_to_data_points(pred_df)
    for start in range(0, len(points), BATCH_SIZE):
        r = client.post(
            "/predict",
            json={"sensor_id": "perf_full", "data": points[start : start + BATCH_SIZE]},
        )
        assert r.status_code == 200


# ── Concurrent access ─────────────────────────────────────────────────────────


def _fit_and_predict_real(sensor_id: str) -> tuple[str, bool]:
    """Fit with real data and predict an anomalous batch; return (sensor_id, alert_fired)."""

    with TestClient(app) as c:
        r = c.post("/fit", json={"sensor_id": sensor_id, "data": _REF_FIT})
        assert r.status_code == 200, f"Fit failed for {sensor_id}"
        r = c.post("/predict", json={"sensor_id": sensor_id, "data": _REF_ANOMALOUS})
        assert r.status_code == 200, f"Predict failed for {sensor_id}"
        return sensor_id, r.json()["alert"]


def test_concurrent_sensors_do_not_interfere():
    """
    Multiple threads operating on different sensor IDs must not cross-contaminate
    each other's state (models or alert engines).
    """
    from sample_processing.api import main as m

    m._models.clear()
    m._engines.clear()

    sensor_ids = [f"concurrent_sensor_{i}" for i in range(10)]
    results: dict[str, bool] = {}

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fit_and_predict_real, sid): sid for sid in sensor_ids}
        for fut in as_completed(futures):
            sid, alerted = fut.result()
            results[sid] = alerted

    for sid, alerted in results.items():
        assert alerted is True, f"Sensor {sid} did not alert (concurrency issue?)"


def test_concurrent_predict_same_sensor_no_crash():
    """
    Multiple threads predicting on the same sensor concurrently must not
    raise exceptions (state reads should be thread-safe).
    """
    with TestClient(app) as c:
        c.post("/fit", json={"sensor_id": "shared_sensor", "data": _REF_FIT})

        def _predict():
            return c.post(
                "/predict", json={"sensor_id": "shared_sensor", "data": _REF_NORMAL}
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_predict) for _ in range(20)]
            for fut in as_completed(futures):
                r = fut.result()
                assert r.status_code == 200
