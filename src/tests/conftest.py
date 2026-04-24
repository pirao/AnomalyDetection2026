from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from sample_processing.model.anomaly_model import load_pipeline_params

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE = Path(__file__).parent.parent.parent
DATA_DIR = _BASE / "data"
LABELS_PATH = _BASE / "labels" / "incidents.yaml"

# ── Pipeline windowing parameters (from pipeline_hyperparams.yaml) ────────────

_PIPELINE = load_pipeline_params()
WINDOW_HOURS = _PIPELINE.model_window_size_hours  # default 2.0
OVERLAP_HOURS = _PIPELINE.window_overlap_hours  # default 1.0
STRIDE_HOURS = WINDOW_HOURS - OVERLAP_HOURS  # default 1.0

# Kept for performance tests that need a row-count baseline
BATCH_SIZE = 50

# ── Scenario IDs ──────────────────────────────────────────────────────────────

ALL_SCENARIO_IDS = list(range(1, 30))


def _load_incidents_raw() -> dict[int, list[dict]]:
    with open(LABELS_PATH) as f:
        raw = yaml.safe_load(f)
    return {int(k): (v or []) for k, v in raw.items()}


def _valid_windows(inc_list: list[dict]) -> list[dict]:
    return [
        i for i in inc_list if i.get("start") and i.get("end") and i["start"] < i["end"]
    ]


_INCIDENTS_RAW = _load_incidents_raw()

NO_INCIDENT_IDS = [
    sid for sid in ALL_SCENARIO_IDS if not _valid_windows(_INCIDENTS_RAW.get(sid, []))
]
INCIDENT_IDS = [sid for sid in ALL_SCENARIO_IDS if sid not in NO_INCIDENT_IDS]
SINGLE_INCIDENT_IDS = [
    sid for sid in INCIDENT_IDS if len(_valid_windows(_INCIDENTS_RAW[sid])) == 1
]
MULTI_INCIDENT_IDS = [
    sid for sid in INCIDENT_IDS if len(_valid_windows(_INCIDENTS_RAW[sid])) >= 2
]

# ── Reference scenario for contract / performance tests ───────────────────────
#
# Scenario 8 is used as the reference for all tests that need a "fit payload",
# a "normal predict batch", and an "anomalous predict batch".
#
# Why scenario 8?
#   - The baseline model produces a confirmed TP on this scenario (alert at
#     2026-02-27T09:57 falls inside the incident window 2026-02-26T18:00 –
#     2026-02-27T12:00), making it a reliable source of anomalous batches.
#   - The first window of the predict file pre-dates the incident by ~9 days,
#     giving a clean "normal" reference batch.
#
_REF_SCENARIO = 8
# Start of the anomalous window within scenario 8's incident period.
# This timestamp is within the confirmed anomalous region (deviation ratio > 0.20).
_ANOMALOUS_BATCH_START = "2026-02-27T01:47:00+00:00"


def _load_ref_data() -> tuple[list[dict], list[dict], list[dict]]:
    """
    Return (fit_payload, normal_batch, anomalous_batch) for the reference
    scenario, all as lists of DataPoint dicts ready for the API.
    Batches are sized according to pipeline_hyperparams.yaml (WINDOW_HOURS).
    Called after df_to_data_points is defined below.
    """
    fit_df = pd.read_parquet(DATA_DIR / f"sensor_data_fit_{_REF_SCENARIO}.parquet")
    pred_df = pd.read_parquet(DATA_DIR / f"sensor_data_pred_{_REF_SCENARIO}.parquet")

    window = timedelta(hours=WINDOW_HOURS)

    # Normal: first WINDOW_HOURS of the predict series (pre-incident)
    first_ts = pred_df["sampled_at"].iloc[0]
    normal_mask = (pred_df["sampled_at"] >= first_ts) & (
        pred_df["sampled_at"] < first_ts + window
    )
    normal_batch = df_to_data_points(pred_df[normal_mask])

    # Anomalous: WINDOW_HOURS window starting at the confirmed anomalous timestamp
    anom_start = pd.Timestamp(_ANOMALOUS_BATCH_START)
    anom_mask = (pred_df["sampled_at"] >= anom_start) & (
        pred_df["sampled_at"] < anom_start + window
    )
    anomalous_batch = df_to_data_points(pred_df[anom_mask])

    return df_to_data_points(fit_df), normal_batch, anomalous_batch


# ── App state isolation ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_app_state():
    """Clear per-sensor state between every test."""
    from sample_processing.api import main as m

    m._models.clear()
    m._engines.clear()
    yield
    m._models.clear()
    m._engines.clear()


# ── TestClient ────────────────────────────────────────────────────────────────


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    from sample_processing.api.main import app

    with TestClient(app) as c:
        yield c


# ── Ground-truth labels ───────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def incidents() -> dict[int, list[dict]]:
    """Load incidents.yaml once per session."""
    with open(LABELS_PATH) as f:
        raw = yaml.safe_load(f)
    return {int(k): (v or []) for k, v in raw.items()}


# ── Pre-computed scenario alerts (session-scoped) ─────────────────────────────


@pytest.fixture(scope="session")
def scenario_alerts() -> dict[int, list[str]]:
    """
    Run all 29 scenarios once per session and cache their alert timestamps.

    Each scenario uses a unique sensor_id so there is no cross-contamination.
    Tests in test_evaluation.py look up results here instead of re-running the
    pipeline, reducing total scenario executions from O(tests) to O(scenarios).
    """
    from sample_processing.api import main as m
    from sample_processing.api.main import app

    m._models.clear()
    m._engines.clear()
    results: dict[int, list[str]] = {}
    with TestClient(app) as client:
        for sid in ALL_SCENARIO_IDS:
            results[sid] = run_scenario(client, sid)
    m._models.clear()
    m._engines.clear()
    return results


# ── Parquet helpers ───────────────────────────────────────────────────────────


def load_scenario(n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (fit_df, pred_df) for scenario *n*."""
    fit = pd.read_parquet(DATA_DIR / f"sensor_data_fit_{n}.parquet")
    pred = pd.read_parquet(DATA_DIR / f"sensor_data_pred_{n}.parquet")
    return fit, pred


def df_to_data_points(df: pd.DataFrame) -> list[dict]:
    """Convert a parquet DataFrame to a list of DataPoint dicts for the API."""
    return [
        {
            "timestamp": row["sampled_at"].isoformat(),
            "uptime": bool(row["uptime"]),
            "vel_rms_x": float(row["vel_rms_x"]),
            "vel_rms_y": float(row["vel_rms_y"]),
            "vel_rms_z": float(row["vel_rms_z"]),
            "accel_rms_x": float(row["accel_rms_x"]),
            "accel_rms_y": float(row["accel_rms_y"]),
            "accel_rms_z": float(row["accel_rms_z"]),
        }
        for _, row in df.iterrows()
    ]

# Loaded once at import time — session-scoped effectively, no session fixture overhead
_REF_FIT, _REF_NORMAL, _REF_ANOMALOUS = _load_ref_data()


# ── Pipeline runner ───────────────────────────────────────────────────────────


def run_scenario(
    client: TestClient,
    scenario_id: int,
) -> list[str]:
    """
    Fit then time-windowed-predict for *scenario_id*. Returns ISO timestamps
    where `alert=True` was returned.

    Windowing is driven by pipeline_hyperparams.yaml:
      - window size : WINDOW_HOURS  (default 2 h)
      - stride      : STRIDE_HOURS  (default 1 h = window - overlap)
    """
    fit_df, pred_df = load_scenario(scenario_id)
    sensor_id = f"sensor_{scenario_id}"

    r = client.post(
        "/fit", json={"sensor_id": sensor_id, "data": df_to_data_points(fit_df)}
    )
    assert r.status_code == 200, f"Fit failed for scenario {scenario_id}: {r.text}"

    window = timedelta(hours=WINDOW_HOURS)
    stride = timedelta(hours=STRIDE_HOURS)

    current = pred_df["sampled_at"].iloc[0]
    end_ts = pred_df["sampled_at"].iloc[-1]

    alerts: list[str] = []
    while current <= end_ts:
        mask = (pred_df["sampled_at"] >= current) & (
            pred_df["sampled_at"] < current + window
        )
        batch_df = pred_df[mask]
        if not batch_df.empty:
            r = client.post(
                "/predict",
                json={"sensor_id": sensor_id, "data": df_to_data_points(batch_df)},
            )
            assert r.status_code == 200, f"Predict failed at {current}: {r.text}"
            if r.json()["alert"]:
                alerts.append(r.json()["timestamp"])
        current += stride

    return alerts


# ── Incident matching ─────────────────────────────────────────────────────────

_GRACE = timedelta(hours=2)


def alert_hits_window(alert_ts_str: str, start_str: str, end_str: str) -> bool:
    """
    Return True if *alert_ts_str* falls in [start - GRACE, end + GRACE].
    Returns False for malformed windows (start >= end).
    """
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    if start >= end:
        return False
    ts = datetime.fromisoformat(alert_ts_str)
    return (start - _GRACE) <= ts <= (end + _GRACE)


def any_alert_in_incidents(alerts: list[str], inc_list: list[dict]) -> bool:
    """Return True if at least one alert falls within any incident window."""
    return any(
        alert_hits_window(a, inc["start"], inc["end"])
        for a in alerts
        for inc in inc_list
        if inc.get("start") and inc.get("end") and inc["start"] < inc["end"]
    )
