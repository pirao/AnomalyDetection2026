"""Serving-layer contract tests. These need no private data and run in CI."""
from __future__ import annotations

# pytest src/tests/test_serving.py -v --pdb for breakpoint debugging

def test_bundle_class_lives_in_deployable_package():
    import sample_processing.serving.registry as reg
    assert reg.AnomalyDetectorBundle.__module__ == "sample_processing.serving.registry"


def test_scenario_id_parsing_handles_known_forms():
    from sample_processing.serving.registry import AnomalyDetectorBundle
    f = AnomalyDetectorBundle._scenario_id
    assert f("sensor_9") == 9
    assert f("analysis_sensor_12") == 12
    assert f("13") == 13
    assert f("not_a_sensor") is None


def test_resolve_tracking_uri_prefers_env(monkeypatch):
    from sample_processing.serving import registry
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert registry.resolve_tracking_uri() == "http://mlflow:5000"


def test_resolve_tracking_uri_falls_back_to_sqlite(monkeypatch):
    from sample_processing.serving import registry
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert registry.resolve_tracking_uri().startswith("sqlite:///")

def test_metadata_reports_runtime_fit_when_no_bundle():
    from fastapi.testclient import TestClient

    from sample_processing.api import main as m
    m._bundle.clear()
    m._serving_meta.clear()
    with TestClient(m.app) as c:
        body = c.get("/metadata").json()
    assert body["bundle_loaded"] is False
    assert body["model_source"] == "runtime_fit"
    assert body["n_bundle_models"] == 0
    for k in ("registry_alias", "registry_version", "model_fingerprint", "git_sha", "eval_f1"):
        assert k in body


def test_ready_returns_503_without_bundle():
    from fastapi.testclient import TestClient

    from sample_processing.api import main as m
    m._bundle.clear()
    with TestClient(m.app) as c:
        r = c.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_metrics_endpoint_exposes_prometheus_text():
    from fastapi.testclient import TestClient

    from sample_processing.api import main as m
    with TestClient(m.app) as c:
        c.get("/health")
        r = c.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "http_requests_total" in r.text