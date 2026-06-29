"""Register the fitted anomaly-detector bundle in the MLflow Model Registry.

This is the *deployment-versioning* counterpart to ``mlflow_experiments.py``
(which only does experiment tracking / comparison). It packages the per-scenario
fitted models from a model-cache version into ONE pyfunc model, registers it
under a single name, copies the cache's change-detection fingerprints onto the
registry version as tags, and promotes a version via an **alias** (MLflow 3 - the
deprecated Staging/Production *stages* are not used).

Why one registered model (not 29): the 29 sensors are calibrations of the same
model, and the 4 scenario-group hyperparameter sets are branches *inside* it. The
served unit routes by ``sensor_id`` -> scenario group internally, so a single
version (a bundle of all fitted weights + configs) is the deployable artifact.

Typical use (from a notebook in the repo root)::

    from analysis.mlflow.mlflow_registry import register_bundle, load_for_serving
    v = register_bundle(1, description="initial calibration", alias="production")
    model = load_for_serving("production")          # what FastAPI would call
    model.predict(batch_df)                          # batch_df has a sensor_id col

View it with::

    mlflow ui --backend-store-uri sqlite:///mlflow.db   # -> Models / Model registry
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature

from sample_processing.serving.registry import (
    REGISTERED_MODEL_NAME,
    AnomalyDetectorBundle,
    load_for_serving as load_for_serving,  # re-exported for notebooks/back-compat
)

# This module lives at src/analysis/mlflow/mlflow_registry.py -> repo root is 3 up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRACKING_URI = f"sqlite:///{(_REPO_ROOT / 'mlflow.db').as_posix()}"
DEFAULT_CACHE_ROOT = _REPO_ROOT / "cache/models"

# REGISTERED_MODEL_NAME is owned by sample_processing.serving.registry (single
# source of truth) and re-exported here for register_bundle and notebooks.
EXPERIMENT_NAME = "anomaly-detector-release"

# The wire-format reading columns the served model expects (same names the
# FastAPI service receives), plus the routing/identity columns.
READING_COLUMNS = [
    "vel_rms_x", "vel_rms_y", "vel_rms_z",
    "accel_rms_x", "accel_rms_y", "accel_rms_z",
]
INPUT_COLUMNS = ["sensor_id", "timestamp", "uptime", *READING_COLUMNS]




def _read_cache_meta(version: int, cache_root: Path = DEFAULT_CACHE_ROOT) -> dict:
    """Read a cache version's ``meta.json`` (written by model_cache.fit_and_save)."""
    return json.loads((Path(cache_root) / f"v{version}" / "meta.json").read_text())


def _input_example():
    """A single-row request example, reused for the signature and log_model."""
    return pd.DataFrame(
        [
            {
                "sensor_id": "sensor_1",
                "timestamp": "2024-01-01T00:00:00+00:00",
                "uptime": True,
                "vel_rms_x": 0.0, "vel_rms_y": 0.0, "vel_rms_z": 0.0,
                "accel_rms_x": 0.0, "accel_rms_y": 0.0, "accel_rms_z": 0.0,
            }
        ]
    )


def _signature():
    """Schema-only signature (does not invoke the model)."""
    example_in = _input_example()
    example_out = pd.DataFrame(
        [
            {
                "sensor_id": "sensor_1",
                "scenario_group": "group_1",
                "anomaly_status": False,
                "alert_score": 0.0,
                "occupancy_score": 0.0,
            }
        ]
    )
    return infer_signature(example_in, example_out)


_EVAL_EXPERIMENT = "baseline-vs-current"
_EVAL_METRIC_KEYS = (
    "precision", "recall", "f1",
    "event_precision", "event_recall", "event_f1",
    "total_alerts", "alert_efficiency",
)


def fetch_evaluation_metrics(
    fingerprint: str,
    *,
    model_tag: str = "current",
    experiment_name: str = _EVAL_EXPERIMENT,
) -> dict[str, float]:
    """Return metrics from the evaluation run whose fingerprint matches this artifact.

    The fingerprint (data_digest + config_hash + git_sha) is the join key that
    proves the evaluation was run against the exact same data and code that
    produced the registered bundle. If no matching run exists, raises ValueError
    rather than returning stale or fabricated numbers.

    Metrics are prefixed ``eval.`` so they are distinguishable from fingerprint
    tags on the registry version (e.g. ``eval.f1``, ``eval.total_alerts``).
    """
    if not fingerprint:
        raise ValueError("Cannot link metrics: bundle has no fingerprint (re-run fit_and_save first).")

    mlflow.set_tracking_uri(_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. Run compare_baseline_vs_current() first."
        )

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.fingerprint = '{fingerprint}' and tags.model = '{model_tag}'",
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"No '{model_tag}' run with fingerprint='{fingerprint}' found in "
            f"'{experiment_name}'. Run compare_baseline_vs_current() with the "
            "same data and config before registering."
        )

    run_metrics = runs[0].data.metrics
    return {
        f"eval.{k}": run_metrics[k]
        for k in _EVAL_METRIC_KEYS
        if k in run_metrics
    }


def register_bundle(
    version: int,
    *,
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    description: str = "",
    alias: str | None = None,
    link_metrics: bool = True,
) -> int:
    """Log cache ``v{version}`` as a pyfunc bundle and register a new registry version.

    The cache meta's fingerprints (``data_digest``/``config_hash``/``git_sha``/
    ``fingerprint``) are copied onto the registry version as tags - the cache
    meta is the single source of truth, so nothing is recomputed here.

    When ``link_metrics=True`` (default), the function looks up the
    ``baseline-vs-current`` experiment for a run whose fingerprint matches this
    bundle's fingerprint and copies its metrics onto the registry version as
    ``eval.*`` tags. This proves the displayed metrics were produced by the exact
    same data and code. Raises ``ValueError`` if no matching evaluation run exists
    (run ``compare_baseline_vs_current()`` first, then re-register).

    Returns the new registry version number.
    """
    cache_root = Path(cache_root)
    bundle_dir = cache_root / f"v{version}"
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Cache version not found: {bundle_dir}")
    meta = _read_cache_meta(version, cache_root)
    fingerprint = meta.get("fingerprint", "")

    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Fetch metrics before logging the model so a missing evaluation aborts early.
    eval_metrics: dict[str, float] = {}
    if link_metrics:
        eval_metrics = fetch_evaluation_metrics(fingerprint)

    tag_keys = ("data_digest", "config_hash", "git_sha", "fingerprint")
    with mlflow.start_run(run_name=f"register-current-cache-v{version}"):
        mlflow.set_tags({"model": "current", "cache_version": f"v{version}",
                         **{k: meta.get(k, "") for k in tag_keys}})
        # NOTE: we intentionally do NOT ship the source via ``code_paths``. The
        # serving environment (notebook, FastAPI container, inference test) always
        # has ``sample_processing`` importable, and copying the package triggers a
        # Windows temp-dir cleanup bug in mlflow. For fully isolated serving you
        # would add ``code_paths=[str(_REPO_ROOT / "src" / "sample_processing")]``.
        mlflow.pyfunc.log_model(
            name="model",
            python_model=AnomalyDetectorBundle(),
            artifacts={"bundle": str(bundle_dir)},
            signature=_signature(),
            input_example=_input_example(),
            registered_model_name=REGISTERED_MODEL_NAME,
            pip_requirements=["mlflow", "pandas", "numpy", "pydantic", "pyyaml"],
        )

        mlflow.log_metrics(eval_metrics)
        
        param_keys = ("version", "data_digest", "config_hash", "git_sha", "git_dirty", "fingerprint")
        mlflow.log_params({k: meta.get(k, "") for k in param_keys})
        mlflow.log_param("cache_version", f"v{version}")
        
    client = mlflow.tracking.MlflowClient()
    new_version = max(
        (int(mv.version) for mv in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")),
        default=0,
    )

    # Stamp the audit trail onto the registry version.
    for key in tag_keys:
        client.set_model_version_tag(REGISTERED_MODEL_NAME, new_version, key, meta.get(key, ""))
    client.set_model_version_tag(REGISTERED_MODEL_NAME, new_version, "cache_version", f"v{version}")

    # Stamp verified evaluation metrics (prefixed eval.*) onto the registry version.
    for metric_key, metric_val in eval_metrics.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, new_version, metric_key, str(round(metric_val, 4)))

    if description:
        client.update_model_version(REGISTERED_MODEL_NAME, new_version, description=description)
    if alias:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, new_version)

    return new_version
