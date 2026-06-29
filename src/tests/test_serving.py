"""Serving-layer contract tests. These need no private data and run in CI."""
from __future__ import annotations


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
