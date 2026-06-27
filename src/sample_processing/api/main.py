"""FastAPI service exposing per-sensor fit and predict endpoints.

Serving sources, in priority order:
  1. ``_models``  - runtime-fit models keyed by the raw ``sensor_id`` string,
     produced by ``/fit`` (or injected by the offline evaluation harness).
  2. ``_bundle``  - the registered pre-fitted bundle keyed by int ``scenario_id``,
     loaded from the MLflow registry ``@production`` alias at startup.

The registry client (``mlflow``) and the offline ``analysis`` package are NOT
runtime dependencies of the lean service/image, so the bundle load is best-effort:
when unavailable (lean Docker image, test container, no ``@production`` alias) the
service degrades silently to runtime-fit serving via ``/fit``.
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sample_processing.model.current.alerting import AlertEngine
from sample_processing.model.current.anomaly_model import AnomalyModel, load_alert_params
from sample_processing.model.current.interface import DataPoint as _DataPoint
from sample_processing.model.current.interface import TimeSeries

_logger = logging.getLogger(__name__)

_ALERT_PARAMS = load_alert_params()

REGISTRY_ALIAS = "production"

# Per-sensor serving state (see module docstring for the resolution order).
_models: dict[str, AnomalyModel] = {}   # runtime-fit, keyed by sensor_id (priority)
_bundle: dict[int, AnomalyModel] = {}   # registry bundle, keyed by scenario_id
_engines: dict[str, AlertEngine] = {}   # stateful alerting, keyed by sensor_id


def _load_registry_bundle() -> None:
    """Best-effort, load-once population of ``_bundle`` from the registry.

    Imports ``analysis``/``mlflow`` lazily so this module stays importable (and
    the lean container starts) even when neither is installed. Any failure is
    logged and swallowed, leaving ``_bundle`` empty (runtime-fit only).
    """
    if _bundle:  # already loaded in this process
        return
    try:
        from analysis.mlflow.mlflow_registry import load_for_serving

        wrapper = load_for_serving(REGISTRY_ALIAS)
        _bundle.update(wrapper.unwrap_python_model()._models)
        _logger.info("Loaded %d models from registry @%s", len(_bundle), REGISTRY_ALIAS)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any failure
        _logger.warning(
            "Registry bundle unavailable (%s); serving runtime-fit models only.", exc
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_registry_bundle()
    yield


app = FastAPI(title="Industrial Sensor Anomaly Detection API", lifespan=lifespan)


# -- Contracts (do not change) -------------------------------------------------


class DataPoint(BaseModel):
    """Single vibration measurement as received over the API."""

    timestamp: str  # ISO 8601
    uptime: bool
    vel_rms_x: float
    vel_rms_y: float
    vel_rms_z: float
    accel_rms_x: float
    accel_rms_y: float
    accel_rms_z: float


class FitRequest(BaseModel):
    """Request body for fitting a sensor-specific model."""

    sensor_id: str
    data: list[DataPoint]


class FitResponse(BaseModel):
    """Response returned after a sensor model is trained."""

    sensor_id: str
    status: str


class PredictRequest(BaseModel):
    """Request body for scoring a sensor batch."""

    sensor_id: str
    data: list[DataPoint]


class PredictResponse(BaseModel):
    """API response with anomaly and alert flags for the submitted batch."""

    sensor_id: str
    anomaly: bool
    alert: bool
    timestamp: str  # ISO 8601 - last timestamp of the submitted batch


# -- Helpers -------------------------------------------------------------------


def _to_timeseries(points: list[DataPoint]) -> TimeSeries:
    data = []
    for p in points:
        ts = datetime.fromisoformat(p.timestamp.replace("Z", "+00:00"))
        data.append(
            _DataPoint(
                timestamp=ts,
                uptime=p.uptime,
                vel_x=p.vel_rms_x,
                vel_y=p.vel_rms_y,
                vel_z=p.vel_rms_z,
                acc_x=p.accel_rms_x,
                acc_y=p.accel_rms_y,
                acc_z=p.accel_rms_z,
            )
        )
    return TimeSeries(data=data)


def _scenario_id_from_sensor_id(sensor_id: str) -> int | None:
    match = re.fullmatch(r"(?:sensor|analysis_sensor)_(\d+)", sensor_id.strip())
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


# -- Endpoints -----------------------------------------------------------------


@app.post("/fit", response_model=FitResponse)
def fit(req: FitRequest):
    """Fit or refit the model for one sensor and reset its alert state."""
    if not req.data:
        raise HTTPException(status_code=422, detail="data must not be empty")
    ts = _to_timeseries(req.data)
    if req.sensor_id not in _models:
        _models[req.sensor_id] = AnomalyModel(scenario_id=_scenario_id_from_sensor_id(req.sensor_id))
        _engines[req.sensor_id] = AlertEngine(_ALERT_PARAMS)
    _models[req.sensor_id].fit(ts)
    _engines[req.sensor_id] = AlertEngine(_ALERT_PARAMS)  # reset alert state on retrain
    return FitResponse(sensor_id=req.sensor_id, status="trained")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Score one sensor batch using its fitted model and alert engine.

    Resolution order: a runtime-fit model for this exact ``sensor_id`` wins;
    otherwise fall back to the registered bundle entry for the sensor's
    ``scenario_id``. 404 if neither exists.
    """
    if not req.data:
        raise HTTPException(status_code=422, detail="data must not be empty")

    model = _models.get(req.sensor_id)
    if model is None:
        sid = _scenario_id_from_sensor_id(req.sensor_id)
        model = _bundle.get(sid) if sid is not None else None
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sensor '{req.sensor_id}' has no fitted model (call /fit or register a bundle)",
        )

    if req.sensor_id not in _engines:
        _engines[req.sensor_id] = AlertEngine(_ALERT_PARAMS)

    ts = _to_timeseries(req.data)
    pred = model.predict(ts)
    decision = _engines[req.sensor_id].predict(pred)
    return PredictResponse(
        sensor_id=req.sensor_id,
        anomaly=pred.anomaly_status,
        alert=decision.alert,
        timestamp=req.data[-1].timestamp,
    )


@app.get("/health")
def health():
    """Return service health for orchestration and contract tests."""
    return {"status": "ok"}
