"""FastAPI service for per-sensor anomaly fitting, scoring, and serving observability.

Endpoints:
  - ``POST /fit`` / ``POST /predict`` - train and score per-sensor models.
  - ``GET /health``   - liveness; the process is up. Frozen contract ``{"status": "ok"}``.
  - ``GET /ready``    - readiness; 200 only when the registry bundle is loaded, else 503.
  - ``GET /metadata`` - which model is serving and its provenance (registry version,
    fingerprint, git sha, eval F1), read from in-memory startup state.
  - ``GET /metrics``  - Prometheus exposition (request count + latency histogram).

Serving sources, in priority order:
  1. ``_models``  - runtime-fit models keyed by the raw ``sensor_id`` string,
     produced by ``/fit`` (or injected by the offline evaluation harness).
  2. ``_bundle``  - the registered pre-fitted bundle keyed by int ``scenario_id``,
     loaded from the MLflow registry ``@production`` alias at startup.

The ``analysis`` package is NOT a runtime dependency of the lean service/image.
``mlflow`` IS a runtime dep (registry client); the bundle load is still best-effort:
when the registry server is unreachable or has no ``@production`` alias the service
degrades to runtime-fit serving via ``/fit`` - but that degradation is now *reported*
(``/ready`` returns 503, ``/metadata`` shows ``bundle_loaded: false``), never silent.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
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
_serving_meta: dict = {}

# -- Prometheus metrics (module scope: registered once, never per request) -----
# The Counter is named ``http_requests`` because prometheus-client appends the
# ``_total`` suffix automatically, yielding the conventional ``http_requests_total``
# series in the exposition.
REQUEST_COUNT = Counter(
    "http_requests",
    "Total HTTP requests processed, labelled by method, endpoint and status.",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds, labelled by endpoint.",
    ["endpoint"],
)


def _load_registry_bundle() -> None:
    """Best-effort, load-once population of ``_bundle`` from the registry.

    Imports ``analysis``/``mlflow`` lazily so this module stays importable (and
    the lean container starts) even when neither is installed. Any failure is
    logged and swallowed, leaving ``_bundle`` empty (runtime-fit only).
    """
    if _bundle:
        return
    try:
        from sample_processing.serving.registry import load_for_serving_with_metadata
        wrapper, meta = load_for_serving_with_metadata(REGISTRY_ALIAS)
        _bundle.update(wrapper.unwrap_python_model()._models)
        _serving_meta.update(meta)
        _logger.info("Loaded %d models from registry @%s", len(_bundle), REGISTRY_ALIAS)
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "Registry bundle unavailable (%s); serving runtime-fit models only.", exc
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_registry_bundle()
    yield


app = FastAPI(title="Industrial Sensor Anomaly Detection API", lifespan=lifespan)


@app.middleware("http")
async def _record_request_metrics(request: Request, call_next):
    """Time and count every request for Prometheus, labelled by route + status."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
    REQUEST_COUNT.labels(
        method=request.method, endpoint=endpoint, status=response.status_code
    ).inc()
    return response


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
    from sample_processing.model.scenario_groups import scenario_id_from_sensor_id
    return scenario_id_from_sensor_id(sensor_id)


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


@app.get("/metadata")
def metadata():
    """Report which model version is currently serving."""
    return {
        "model_source": "registry" if _bundle else "runtime_fit",
        "registry_alias": REGISTRY_ALIAS,
        "bundle_loaded": bool(_bundle),
        "registry_version": _serving_meta.get("registry_version"),
        "n_bundle_models": len(_bundle),
        "n_runtime_models": len(_models),
        "model_fingerprint": _serving_meta.get("model_fingerprint"),
        "git_sha": _serving_meta.get("git_sha"),
        "eval_f1": _serving_meta.get("eval_f1"),
    }


@app.get("/ready")
def ready():
    """Readiness probe: 200 only when the registry bundle is loaded.

    Distinct from ``/health`` (liveness). A process that started but could not
    load the ``@production`` bundle is *alive* yet *not ready* to serve the
    registered model, so an orchestrator should hold it out of traffic rotation
    (503) rather than restart it.
    """
    if _bundle:
        return {"ready": True}
    return JSONResponse(
        status_code=503,
        content={"ready": False, "reason": "registry bundle not loaded"},
    )


@app.get("/metrics")
def metrics():
    """Expose request counters and latency histograms in Prometheus text format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)