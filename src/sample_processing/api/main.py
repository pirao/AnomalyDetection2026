"""FastAPI service exposing per-sensor fit and predict endpoints."""

from __future__ import annotations

from datetime import datetime
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sample_processing.model.current.alerting import AlertEngine
from sample_processing.model.current.anomaly_model import AnomalyModel, load_alert_params
from sample_processing.model.current.interface import DataPoint as _DataPoint
from sample_processing.model.current.interface import TimeSeries

_ALERT_PARAMS = load_alert_params()

app = FastAPI(title="Industrial Sensor Anomaly Detection API")

# one model + engine instance per sensor
_models: dict[str, AnomalyModel] = {}
_engines: dict[str, AlertEngine] = {}


# ── Contracts (do not change) ─────────────────────────────────────────────────


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
    timestamp: str  # ISO 8601 — last timestamp of the submitted batch


# ── Helpers ───────────────────────────────────────────────────────────────────


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


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/fit", response_model=FitResponse)
def fit(req: FitRequest):
    """Fit or refit the model for one sensor and reset its alert state."""
    if not req.data:
        raise HTTPException(status_code=422, detail="data must not be empty")
    ts = _to_timeseries(req.data)
    if req.sensor_id not in _models:
        _models[req.sensor_id] = AnomalyModel(
            scenario_id=_scenario_id_from_sensor_id(req.sensor_id),
        )
        _engines[req.sensor_id] = AlertEngine(_ALERT_PARAMS)
    _models[req.sensor_id].fit(ts)
    _engines[req.sensor_id] = AlertEngine(_ALERT_PARAMS)  # reset alert state on retrain
    return FitResponse(sensor_id=req.sensor_id, status="trained")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Score one sensor batch using its fitted model and alert engine."""
    if not req.data:
        raise HTTPException(status_code=422, detail="data must not be empty")
    model = _models.get(req.sensor_id)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sensor '{req.sensor_id}' has not been trained yet",
        )
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
