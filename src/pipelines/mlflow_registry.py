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

Typical use (from a notebook, run *inside* the ``notebooks`` container - see the
platform guard in ``register_bundle`` for why this must not run on Windows)::

    from pipelines.mlflow_registry import register_bundle
    from anomaly_detection.registry.bundle import load_for_serving
    v = register_bundle(1, description="initial calibration", alias="production")
    model = load_for_serving("production")          # what FastAPI calls at startup
    model.predict(batch_df)                          # batch_df has a sensor_id col

View it with::

    docker compose up mlflow   # -> http://localhost:5000 -> Models
"""
import json
import platform
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import infer_signature

from anomaly_detection.registry.bundle import REGISTERED_MODEL_NAME, resolve_tracking_uri

# This module lives at src/pipelines/mlflow_registry.py -> repo root is 2 up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = _REPO_ROOT / "cache/models"
_MODEL_FROM_CODE_PATH = _REPO_ROOT / "src" / "anomaly_detection" / "registry" / "_model_from_code.py"

# REGISTERED_MODEL_NAME is owned by anomaly_detection.registry.bundle (single
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

    The fingerprint (data_digest + config_hash + code_version) is the join key
    that proves the evaluation was run against the exact same data and code that
    produced the registered bundle. If no matching run exists, raises ValueError
    rather than returning stale or fabricated numbers.

    Metrics are prefixed ``eval.`` so they are distinguishable from fingerprint
    tags on the registry version (e.g. ``eval.f1``, ``eval.total_alerts``).
    """
    if not fingerprint:
        raise ValueError("Cannot link metrics: bundle has no fingerprint (re-run fit_and_save first).")

    mlflow.set_tracking_uri(resolve_tracking_uri())
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


def _registry_version_by_fingerprint(client, fingerprint: str) -> int | None:
    """Return an existing registry version whose ``fingerprint`` tag matches, if any.

    This is what makes registration idempotent: an unchanged model (same data +
    config + code version, so the same ``fingerprint``) is already registered, and
    re-running should reuse that version rather than mint a duplicate. Returns the
    highest matching version number, or ``None`` when there is no match (or when
    the bundle carries no fingerprint).
    """
    if not fingerprint:
        return None
    matches: list[int] = []
    for mv in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"):
        tags = mv.tags or {}
        if not tags:  # some backends omit tags on search results; read them directly
            tags = client.get_model_version(REGISTERED_MODEL_NAME, mv.version).tags or {}
        if tags.get("fingerprint") == fingerprint:
            matches.append(int(mv.version))
    return max(matches) if matches else None


def register_bundle(
    version: int,
    *,
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    description: str = "",
    alias: str | None = None,
    link_metrics: bool = True,
    force: bool = False,
) -> int:
    """Register cache ``v{version}`` as a pyfunc bundle, idempotently.

    The cache meta's fingerprints (``data_digest``/``config_hash``/``git_sha``/
    ``fingerprint``) are copied onto the registry version as tags - the cache
    meta is the single source of truth, so nothing is recomputed here.

    **Idempotent by fingerprint.** If a registry version already carries this
    bundle's ``fingerprint`` (same data + config + ``MODEL_CODE_VERSION``), that
    version is reused: no duplicate is logged, its ``eval.*`` tags / description
    are refreshed, and the alias is re-pointed at it. This stops identical
    re-runs from minting v9, v10, ... Pass ``force=True`` to always create a new
    version regardless of fingerprint. Note that the fingerprint match only proves
    the *inputs* were identical - it does not verify the previously-registered
    version's artifacts are intact. If an earlier registration was broken by a
    since-fixed bug, re-running with the same fingerprint will just reuse the
    broken version; pass ``force=True`` once to mint a clean replacement.

    When ``link_metrics=True`` (default), the function looks up the
    ``baseline-vs-current`` experiment for a run whose fingerprint matches this
    bundle's fingerprint and copies its metrics onto the registry version as
    ``eval.*`` tags. This proves the displayed metrics were produced by the exact
    same data and code. Raises ``ValueError`` if no matching evaluation run exists
    (run ``compare_baseline_vs_current()`` first, then re-register).

    Returns the registry version number (new, or the reused existing one).
    """
    if platform.system() == "Windows":
        # mlflow bakes the calling OS's native path separators into the model;
        # the Linux api container can't resolve a Windows-style path at load time.
        raise RuntimeError(
            "register_bundle() must not run on Windows - it bakes an OS-native "
            "code path into the MLflow model that the Linux api container can't "
            "resolve. Run it inside the `notebooks` Docker service instead."
        )
    cache_root = Path(cache_root)
    bundle_dir = cache_root / f"v{version}"
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Cache version not found: {bundle_dir}")
    meta = _read_cache_meta(version, cache_root)
    fingerprint = meta.get("fingerprint", "")

    mlflow.set_tracking_uri(resolve_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()

    # Fetch metrics before logging the model so a missing evaluation aborts early.
    eval_metrics: dict[str, float] = {}
    if link_metrics:
        eval_metrics = fetch_evaluation_metrics(fingerprint)

    tag_keys = ("data_digest", "config_hash", "git_sha", "fingerprint")

    # Idempotency: reuse an existing version with the same fingerprint instead of
    # logging a duplicate. force=True skips the check and always logs a new one.
    target_version = None if force else _registry_version_by_fingerprint(client, fingerprint)
    if target_version is not None:
        print(
            f"Fingerprint {fingerprint} already registered as version "
            f"{target_version}; reusing it (no new version logged). "
            "Pass force=True to override."
        )
    else:
        with mlflow.start_run(run_name=f"register-current-cache-v{version}"):
            mlflow.set_tags({"model": "current", "cache_version": f"v{version}",
                             **{k: meta.get(k, "") for k in tag_keys}})
            # python_model is a path string (models-from-code), not an
            # AnomalyDetectorBundle() instance, so mlflow packages this one script
            # instead of CloudPickle-serializing the object. We intentionally do NOT
            # also ship the source via `code_paths`: the serving environment (notebook,
            # FastAPI container, inference test) always has `anomaly_detection`
            # importable, and copying the whole package triggers a Windows temp-dir
            # cleanup bug in mlflow. For fully isolated serving you would add
            # `code_paths=[str(_REPO_ROOT / "src" / "anomaly_detection")]`.
            # `artifacts=` needs a proper "file://" URI (as_uri()), not a raw path -
            # a raw Windows path is misread by urllib.parse as scheme "c".
            mlflow.pyfunc.log_model(
                name="model",
                python_model=str(_MODEL_FROM_CODE_PATH),
                artifacts={"bundle": bundle_dir.resolve().as_uri()},
                signature=_signature(),
                input_example=_input_example(),
                registered_model_name=REGISTERED_MODEL_NAME,
                pip_requirements=["mlflow", "pandas", "numpy", "pydantic", "pyyaml"],
            )

            mlflow.log_metrics(eval_metrics)

            param_keys = ("version", "data_digest", "config_hash", "git_sha", "git_dirty", "fingerprint")
            mlflow.log_params({k: meta.get(k, "") for k in param_keys})
            mlflow.log_param("cache_version", f"v{version}")

        target_version = max(
            (int(mv.version) for mv in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")),
            default=0,
        )

        # Stamp the audit trail onto the new registry version.
        for key in tag_keys:
            client.set_model_version_tag(REGISTERED_MODEL_NAME, target_version, key, meta.get(key, ""))
        client.set_model_version_tag(REGISTERED_MODEL_NAME, target_version, "cache_version", f"v{version}")

    # Refresh verified evaluation metrics (eval.*), description, and alias on the
    # target version - for both a fresh and a reused version, so a re-run always
    # re-points the alias and updates the displayed metrics.
    for metric_key, metric_val in eval_metrics.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, target_version, metric_key, str(round(metric_val, 4)))

    if description:
        client.update_model_version(REGISTERED_MODEL_NAME, target_version, description=description)
    if alias:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, target_version)

    return target_version
