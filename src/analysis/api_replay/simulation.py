"""Single-scenario API replay — the per-batch model + alert-engine loop.

``simulate_api_replay_one_scenario`` is the public entry point. It picks a
batching mode (``time`` or ``row``), drives ``_simulate_replay_batches`` over
that iterator, and returns a debug DataFrame with one row per batch that
captures the full prediction/alert diagnostics used by notebook 02 and the
scoring widgets.

Callers: ``plotting.scoring.api_replay_widget``, ``evaluation``, notebook 02.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sample_processing.model.alert_engine import AlertEngine
from sample_processing.model.anomaly_model import AnomalyModel, load_alert_params, load_pipeline_params
from sample_processing.model.interface import AlertParams, ModelParams, PredictOutput

from .batching import df_to_timeseries, iter_row_batches, iter_time_batches
from .incidents import get_incident_spans

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = _REPO_ROOT / "data"


def _infer_expected_samples_from_pipeline(
    batch_df: pd.DataFrame,
    *,
    time_col: str,
    window_hours: float,
    default_minutes: float = 10.0,
) -> int:
    """Infer the nominal full-window row count from batch cadence.

    Returns the nominal denominator for a full window (e.g. 2h), not the
    observed row count. Falls back to ``default_minutes`` cadence when
    timestamps are too sparse to infer a dominant delta.
    """
    ts = pd.to_datetime(batch_df[time_col], errors="coerce").dropna().sort_values().drop_duplicates()
    if len(ts) < 2:
        cadence_minutes = default_minutes
    else:
        deltas = ts.diff().dt.total_seconds().div(60.0).dropna()
        deltas = deltas[deltas > 0]
        if deltas.empty:
            cadence_minutes = default_minutes
        else:
            mode = deltas.mode()
            cadence_minutes = float(mode.iloc[0]) if not mode.empty else float(deltas.median())
            if cadence_minutes <= 0:
                cadence_minutes = default_minutes

    expected = int(round((float(window_hours) * 60.0) / cadence_minutes))
    return max(expected, 1)


def _simulate_replay_batches(
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    batch_iterator,
    model: AnomalyModel | None = None,
    is_cyclic: bool = False,
    baseline_scaler: str | None = None,
    params_path: Path | None = None,
    model_params_override: ModelParams | None = None,
    alert_params: AlertParams | None = None,
    include_alert_engine: bool = True,
    sensor_id: str | None = None,
    scenario_col: str = "scenario_id",
    time_col: str = "sampled_at",
    label_col: str = "label",
    normal_label: str = "normal",
    batching_mode: str,
    alert_params_source: str,
    configured_batch_size: int | None = None,
) -> pd.DataFrame:
    """Shared replay implementation used by both API and report-style modes."""
    pipeline = load_pipeline_params()
    if model is None:
        model = AnomalyModel(
            params_path=params_path,
            is_cyclic=is_cyclic,
        )
        if model_params_override is not None:
            model.params = model_params_override
            if hasattr(model._backend, "baseline_scaler"):
                model._backend.baseline_scaler = model.params.baseline_scaler
        if baseline_scaler is not None:
            if baseline_scaler != "standard":
                raise ValueError("baseline_scaler must be 'standard'")
            model.params.baseline_scaler = baseline_scaler
            if hasattr(model._backend, "baseline_scaler"):
                model._backend.baseline_scaler = baseline_scaler
        model.fit(df_to_timeseries(fit_df, time_col=time_col))
    else:
        if model_params_override is not None:
            model.params = model_params_override
    engine = AlertEngine(alert_params) if include_alert_engine else None

    scenario_id = (
        pred_df[scenario_col].iloc[0]
        if scenario_col in pred_df.columns and not pred_df.empty
        else None
    )
    sensor_name = sensor_id if sensor_id is not None else (
        f"sensor_{scenario_id}" if scenario_id is not None else None
    )

    rows: list[dict[str, Any]] = []
    for batch_index, window_start, window_end, batch_df in batch_iterator:
        batch_ts = df_to_timeseries(batch_df, time_col=time_col)
        expected_samples = _infer_expected_samples_from_pipeline(
            batch_df,
            time_col=time_col,
            window_hours=pipeline.model_window_size_hours,
        )
        details = model.predict_batch_details(
            batch_ts,
            expected_samples_per_window=expected_samples,
        )

        alert = False
        relalert_reason = ""
        alert_debug: dict[str, Any] = {}
        decision_timestamp = details["timestamp"]
        anchored_timestamp = details["timestamp"]
        owner_level = -1
        owner_kind = ""
        group_family = ""
        if engine is not None:
            all_channel_details = {
                **details["vel_channel_details"],
                **details["accel_channel_details"],
            }
            prediction = PredictOutput(
                anomaly_status=bool(details["anomaly_status"]),
                timestamp=details["timestamp"],
                occupancy_score=float(details["occupancy_score"]),
                alert_score=float(details["alert_score"]),
                mean_d_score=float(details["alert_score"]),
                active_channels=list(details["active_channels"]),
                active_modalities=list(details["active_modalities"]),
                channel_max_residual={
                    col: float(info["max_residual"])
                    for col, info in all_channel_details.items()
                },
            )
            decision = engine.predict(prediction)
            alert = bool(decision.alert)
            relalert_reason = engine.last_realert_reason or decision.message
            alert_debug = dict(engine.last_debug)
            decision_timestamp = getattr(decision, "decision_timestamp", None) or details["timestamp"]
            anchored_timestamp = getattr(decision, "anchored_timestamp", None) or getattr(decision, "timestamp", details["timestamp"])
            owner_level = int(getattr(decision, "owner_level", alert_debug.get("owner_level", -1)))
            owner_kind = str(getattr(decision, "owner_kind", alert_debug.get("owner_kind", "")))
            group_family = str(getattr(decision, "group_family", alert_debug.get("group_family", "")))

        channel_max_residual = {
            **{
                col: float(info["max_residual"])
                for col, info in details["vel_channel_details"].items()
            },
            **{
                col: float(info["max_residual"])
                for col, info in details["accel_channel_details"].items()
            },
        }
        window_mid = window_start + (window_end - window_start) / 2
        plot_time = details["timestamp"] if batching_mode == "row" else window_mid

        rows.append(
            {
                "scenario_id": scenario_id,
                "sensor_id": sensor_name,
                "is_cyclic": bool(is_cyclic),
                "batch_index": int(batch_index),
                "batching_mode": str(batching_mode),
                "alert_params_source": str(alert_params_source),
                "configured_batch_size": (
                    int(configured_batch_size)
                    if configured_batch_size is not None
                    else pd.NA
                ),
                "window_start": window_start,
                "window_end": window_end,
                "window_mid": window_mid,
                "plot_time": plot_time,
                "window_hours": (window_end - window_start).total_seconds() / 3600.0,
                "batch_rows": int(len(batch_df)),
                "expected_samples_per_window": int(details["expected_samples_per_window"]),
                "coverage": (
                    float(len(batch_df) / details["expected_samples_per_window"])
                    if int(details["expected_samples_per_window"]) > 0
                    else 0.0
                ),
                "vel_occupancy": float(details["vel_occupancy"]),
                "accel_occupancy": float(details["accel_occupancy"]),
                "vel_occupancy_fixed": float(details.get("vel_occupancy_fixed", details["vel_occupancy"])),
                "accel_occupancy_fixed": float(details.get("accel_occupancy_fixed", details["accel_occupancy"])),
                "occupancy_score": float(details["occupancy_score"]),
                "occupancy_score_fixed": float(details.get("occupancy_score_fixed", details["occupancy_score"])),
                "vel_occupancy_observed": float(details["vel_occupancy_observed"]),
                "accel_occupancy_observed": float(details["accel_occupancy_observed"]),
                "occupancy_score_observed": float(details["occupancy_score_observed"]),
                "alert_score": float(details["alert_score"]),
                "max_residual_active": float(details["max_residual_active"]),
                "anomaly_status": bool(details["anomaly_status"]),
                "alert": bool(alert),
                "active_modalities": list(details["active_modalities"]),
                "active_channels": list(details["active_channels"]),
                "relalert_reason": relalert_reason,
                "alert_event": str(alert_debug.get("event", "")),
                "triggered_channels": list(alert_debug.get("triggered_channels", [])),
                "opened_channels": list(alert_debug.get("opened_channels", [])),
                "realerted_channels": list(alert_debug.get("realerted_channels", [])),
                "reset_channels": list(alert_debug.get("reset_channels", [])),
                "watched_channels": list(alert_debug.get("watched_channels", [])),
                "group_mode_active": bool(alert_debug.get("group_mode_active", False)),
                "group_mode_type": str(alert_debug.get("group_mode_type", "")),
                "group_mode_kind": str(alert_debug.get("group_mode_kind", "")),
                "group_metric_label": str(alert_debug.get("group_metric_label", "")),
                "group_event": str(alert_debug.get("group_event", "")),
                "group_opened": bool(alert_debug.get("group_opened", False)),
                "group_realerted": bool(alert_debug.get("group_realerted", False)),
                "group_reset": bool(alert_debug.get("group_reset", False)),
                "group_channels": list(alert_debug.get("group_channels", [])),
                "group_active_channels": list(alert_debug.get("group_active_channels", [])),
                "group_reference_severity": float(alert_debug.get("group_reference_severity", 0.0)),
                "group_current_severity": float(alert_debug.get("group_current_severity", 0.0)),
                "group_cooldown_windows": int(alert_debug.get("group_cooldown_windows", 0)),
                "group_reset_streak": int(alert_debug.get("group_reset_streak", 0)),
                "suppressed_channel_alerts": list(alert_debug.get("suppressed_channel_alerts", [])),
                "pending_channels": list(alert_debug.get("pending_channels", [])),
                "pending_lower_priority_events": list(alert_debug.get("pending_lower_priority_events", [])),
                "suppressed_by_priority": list(alert_debug.get("suppressed_by_priority", [])),
                "suppression_target": list(alert_debug.get("suppression_target", [])),
                "promotion_candidate_kind": str(alert_debug.get("promotion_candidate_kind", "")),
                "promotion_holdback_remaining": int(alert_debug.get("promotion_holdback_remaining", 0)),
                "promotion_resolution_state": str(alert_debug.get("promotion_resolution_state", "")),
                "pending_event_release_reason": str(alert_debug.get("pending_event_release_reason", "")),
                "suppression_window_expires_at": int(alert_debug.get("suppression_window_expires_at", -1)),
                "emitted_event_scope": str(alert_debug.get("emitted_event_scope", "")),
                "individual_alert_mode": str(alert_debug.get("individual_alert_mode", "")),
                "decision_timestamp": decision_timestamp,
                "anchored_timestamp": anchored_timestamp,
                "owner_level": owner_level,
                "owner_kind": owner_kind,
                "group_family": group_family or str(alert_debug.get("group_family", "")),
                "triggered_reasons_by_channel": dict(alert_debug.get("triggered_reasons_by_channel", {})),
                "current_channels": list(alert_debug.get("current_channels", [])),
                "reference_max_by_channel": dict(alert_debug.get("reference_max_by_channel", {})),
                "current_channel_max_residual": dict(alert_debug.get("current_channel_max_residual", {})),
                "current_max_by_channel": dict(alert_debug.get("current_max_by_channel", {})),
                "reset_streak_by_channel": dict(alert_debug.get("reset_streak_by_channel", {})),
                "cooldown_by_channel": dict(alert_debug.get("cooldown_by_channel", {})),
                "channel_max_residual": channel_max_residual,
                "batch_rows_vel": int(details["batch_rows_vel"]),
                "batch_rows_accel": int(details["batch_rows_accel"]),
                "timestamp": anchored_timestamp,
                "vel_channel_details": details["vel_channel_details"],
                "accel_channel_details": details["accel_channel_details"],
            }
        )

    replay_df = pd.DataFrame(rows)
    if replay_df.empty:
        return replay_df

    if label_col in pred_df.columns:
        replay_df["incident_spans"] = [
            get_incident_spans(
                pred_df,
                time_col=time_col,
                label_col=label_col,
                normal_label=normal_label,
            )
        ] * len(replay_df)

    return replay_df


def simulate_api_replay_one_scenario(
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    model: AnomalyModel | None = None,
    is_cyclic: bool = False,
    baseline_scaler: str | None = None,
    params_path: Path | None = None,
    model_params_override: ModelParams | None = None,
    alert_params: AlertParams | None = None,
    include_alert_engine: bool = True,
    sensor_id: str | None = None,
    scenario_col: str = "scenario_id",
    time_col: str = "sampled_at",
    label_col: str = "label",
    normal_label: str = "normal",
    batching_mode: str = "time",
    row_batch_size: int = 50,
    use_default_alert_params: bool = False,
) -> pd.DataFrame:
    """Replay one scenario batch by batch and return a compact debug dataframe.

    Parameters
    ----------
    model :
        Optional pre-fitted ``AnomalyModel``. When provided the ``fit`` step is
        skipped entirely — only ``model_params_override`` (alarm-rule params)
        is applied before each batch. Pass ``None`` (default) to keep the
        existing fit-from-data behaviour.
    """
    pipeline = load_pipeline_params()
    effective_alert_params = AlertParams() if use_default_alert_params else (alert_params or load_alert_params())

    if batching_mode == "time":
        batch_iterator = iter_time_batches(
            pred_df,
            window_hours=pipeline.model_window_size_hours,
            overlap_hours=pipeline.window_overlap_hours,
            time_col=time_col,
        )
        configured_batch_size = None
    elif batching_mode == "row":
        batch_iterator = iter_row_batches(
            pred_df,
            batch_size=row_batch_size,
            time_col=time_col,
        )
        configured_batch_size = row_batch_size
    else:
        raise ValueError("batching_mode must be 'time' or 'row'")

    return _simulate_replay_batches(
        fit_df=fit_df,
        pred_df=pred_df,
        batch_iterator=batch_iterator,
        model=model,
        is_cyclic=is_cyclic,
        baseline_scaler=baseline_scaler,
        params_path=params_path,
        model_params_override=model_params_override,
        alert_params=effective_alert_params,
        include_alert_engine=include_alert_engine,
        sensor_id=sensor_id,
        scenario_col=scenario_col,
        time_col=time_col,
        label_col=label_col,
        normal_label=normal_label,
        batching_mode=batching_mode,
        alert_params_source="defaults" if use_default_alert_params else "yaml",
        configured_batch_size=configured_batch_size,
    )
