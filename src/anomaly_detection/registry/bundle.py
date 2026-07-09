from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRACKING_URI = f"sqlite:///{(_REPO_ROOT / 'mlflow.db').as_posix()}"
REGISTERED_MODEL_NAME = "anomaly-detector-current"


class AnomalyDetectorBundle(mlflow.pyfunc.PythonModel):
    """One pyfunc holding all per-scenario fitted models; routes by ``sensor_id``.

    Input (DataFrame): one batch/window of readings with columns
        ``sensor_id, timestamp, uptime, vel_rms_*, accel_rms_*``. Rows for
        several sensors may be mixed; they are grouped by ``sensor_id``.
    Output (DataFrame): one row per sensor with ``anomaly_status``,
        ``alert_score``, ``occupancy_score`` and the resolved ``scenario_group``.

    Routing: ``sensor_id`` -> ``scenario_id`` -> ``get_scenario_group_key`` ->
    that scenario's fitted model (whose params already encode the group's
    hyperparameters). A single served endpoint therefore covers all sensors and
    all four group configs.
    """

    def load_context(self, context):
        import glob
        import os
        import pickle

        self._models = {}
        # MLflow records the artifact sub-path with the OS separator at log time
        # (its os.path.join bug means a model registered on Windows stores
        # ``artifacts\\v1``). On a POSIX server that backslash is a literal
        # filename char, so normalize it before globbing the bundle directory.
        bundle_dir = context.artifacts["bundle"].replace("\\", "/")
        for pkl in sorted(glob.glob(os.path.join(bundle_dir, "*.pkl"))):
            sid = int(Path(pkl).stem)
            with open(pkl, "rb") as f:
                self._models[sid] = pickle.load(f)

    @staticmethod
    def _scenario_id(sensor_id) -> int | None:
        from sample_processing.model.scenario_groups import scenario_id_from_sensor_id
        return scenario_id_from_sensor_id(sensor_id)

    def _to_timeseries(self, frame: pd.DataFrame):
        from datetime import datetime

        from sample_processing.model.current.interface import DataPoint, TimeSeries

        points = []
        for row in frame.itertuples(index=False):
            ts = row.timestamp
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            points.append(
                DataPoint(
                    timestamp=ts,
                    uptime=bool(getattr(row, "uptime", True)),
                    vel_x=float(row.vel_rms_x),
                    vel_y=float(row.vel_rms_y),
                    vel_z=float(row.vel_rms_z),
                    acc_x=float(row.accel_rms_x),
                    acc_y=float(row.accel_rms_y),
                    acc_z=float(row.accel_rms_z),
                )
            )
        return TimeSeries(data=points)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        from sample_processing.model.scenario_groups import get_scenario_group_key

        results = []
        for sensor_id, frame in model_input.groupby("sensor_id"):
            sid = self._scenario_id(sensor_id)
            group = get_scenario_group_key(sid)
            model = self._models.get(sid)
            if model is None:
                results.append(
                    {
                        "sensor_id": sensor_id,
                        "scenario_group": group,
                        "anomaly_status": None,
                        "alert_score": None,
                        "occupancy_score": None,
                    }
                )
                continue
            pred = model.predict(self._to_timeseries(frame))
            results.append(
                {
                    "sensor_id": sensor_id,
                    "scenario_group": group,
                    "anomaly_status": bool(pred.anomaly_status),
                    "alert_score": float(pred.alert_score),
                    "occupancy_score": float(pred.occupancy_score),
                }
            )
        return pd.DataFrame(results)


def resolve_tracking_uri() -> str:
    """Return the MLflow tracking URI to use for serving.

    If the environment variable ``MLFLOW_TRACKING_URI`` is set, that is used.
    Otherwise, a local SQLite file in the repo root is used.
    """
    import os

    return os.environ.get("MLFLOW_TRACKING_URI", _TRACKING_URI)


def load_for_serving(alias: str = "production"):
    """Load the aliased registry version as a pyfunc - what FastAPI would call."""
    mlflow.set_tracking_uri(resolve_tracking_uri())
    return mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{alias}")


def load_for_serving_with_metadata(alias: str = "production") -> tuple[object, dict]:
    """Load the aliased registry pyfunc together with its provenance metadata.

    Returns ``(pyfunc, meta)`` where ``meta`` carries the resolved registry
    version and the provenance tags (``fingerprint``, ``git_sha``, ``eval.f1``)
    recorded at registration time, so the API can report exactly what it is
    serving without re-querying MLflow on every request.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri())
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    tags = mv.tags or {}
    meta = {
        "registry_version": mv.version,
        "model_fingerprint": tags.get("fingerprint"),
        "git_sha": tags.get("git_sha"),
        "eval_f1": float(tags["eval.f1"]) if "eval.f1" in tags else None,
    }
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{alias}")
    return model, meta
