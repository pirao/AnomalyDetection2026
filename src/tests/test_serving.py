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
