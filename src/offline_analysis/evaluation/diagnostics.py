"""Per-scenario / per-alert diagnostics: group reassignment and replay explanation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_detection.model.grouped_residual.params import AlertParams, load_alert_params
from anomaly_detection.model.shared.scenario_groups import (
    GROUP_DEFINITIONS,
    get_scenario_group_key,
    get_scenario_group_label,
)

from .incidents import (
    _normalize_incidents,
    load_incidents_by_scenario,
)
from .metrics import (
    _candidate_comparison_outcome,
    _run_api_scenario,
    prepare_scenario_frames,
    summarize_inference_test_metrics,
)
from .simulation import DEFAULT_DATA_DIR


def diagnose_group_reassignment(
    scenario_id: int,
    *,
    full_df: pd.DataFrame | None = None,
    data_dir: Path | str | None = None,
    labels_path: Path | str | None = None,
    alert_params: AlertParams | None = None,
    time_col: str = "sampled_at",
) -> pd.DataFrame:
    """Run scenario_id through every candidate group and show side-by-side metrics.

    Prints a readable table and returns the full comparison DataFrame so you can
    inspect it interactively. Useful for debugging why no reassignment was found,
    or confirming that no group is strictly better or marginal.

    Returns
    -------
    pd.DataFrame
        One row per candidate group with columns:
        candidate_group, candidate_status, candidate_missed, candidate_covered,
        candidate_n_alerts, delta_n_alerts, comparison_outcome.
        The current group is shown as a header line, not a row.
    """
    from fastapi.testclient import TestClient

    from anomaly_detection.api import main as api_main

    resolved_data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    incidents_by_scenario = load_incidents_by_scenario(labels_path)

    effective_alert_params = alert_params or load_alert_params()
    api_main._ALERT_PARAMS = effective_alert_params

    scenario_frames = prepare_scenario_frames(
        full_df=full_df,
        data_dir=resolved_data_dir,
        ordered_ids=[scenario_id],
        scenario_col="scenario_id",
        split_col="split",
        fit_value="fit",
        pred_value="pred",
        time_col=time_col,
    )
    fit_df, pred_df = scenario_frames[scenario_id]

    current_group = get_scenario_group_key(scenario_id)

    with TestClient(api_main.app) as client:
        current_alerts = _run_api_scenario(
            client,
            scenario_id=scenario_id,
            fit_df=fit_df,
            pred_df=pred_df,
            time_col=time_col,
        )
        current_report = summarize_inference_test_metrics(
            {scenario_id: current_alerts},
            incidents_by_scenario,
            scenario_ids=[scenario_id],
        )
        current_row = current_report["scenarios_df"].iloc[0].to_dict()

        rows = []
        for candidate_group in GROUP_DEFINITIONS.keys():
            if candidate_group == current_group:
                continue
            alt_alerts = _run_api_scenario(
                client,
                scenario_id=scenario_id,
                fit_df=fit_df,
                pred_df=pred_df,
                time_col=time_col,
                model_group_key=candidate_group,
            )
            alt_report = summarize_inference_test_metrics(
                {scenario_id: alt_alerts},
                incidents_by_scenario,
                scenario_ids=[scenario_id],
            )
            alt_row = alt_report["scenarios_df"].iloc[0].to_dict()
            candidate_metrics = {
                "missed_incident_count": int(alt_row["missed_incident_count"]),
                "status": str(alt_row["status"]),
                "n_alerts": int(alt_row["n_alerts"]),
            }
            current_metrics = {
                "missed_incident_count": int(current_row["missed_incident_count"]),
                "status": str(current_row["status"]),
                "n_alerts": int(current_row["n_alerts"]),
            }
            rows.append({
                "candidate_group": candidate_group,
                "candidate_group_label": str(GROUP_DEFINITIONS[candidate_group]["label"]),
                "candidate_status": str(alt_row["status"]),
                "candidate_missed": int(alt_row["missed_incident_count"]),
                "candidate_covered": int(alt_row["covered_incident_count"]),
                "candidate_n_alerts": int(alt_row["n_alerts"]),
                "delta_n_alerts": int(alt_row["n_alerts"]) - int(current_row["n_alerts"]),
                "comparison_outcome": _candidate_comparison_outcome(candidate_metrics, current_metrics),
            })

    print(
        f"Scenario {scenario_id} - current group: {current_group} "
        f"({get_scenario_group_label(scenario_id)})"
    )
    print(
        f"  status={current_row['status']}  "
        f"missed={current_row['missed_incident_count']}  "
        f"covered={current_row['covered_incident_count']}  "
        f"n_alerts={current_row['n_alerts']}\n"
    )

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        print(result_df.to_string(index=False))
    return result_df


def _signed_delta_to_expanded_window(
    ts: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tol: pd.Timedelta,
) -> pd.Timedelta:
    """Return signed distance from *ts* to the grace-expanded incident window."""
    left = start - tol
    right = end + tol
    if ts < left:
        return ts - left
    if ts > right:
        return ts - right
    return pd.Timedelta(0)


def diagnose_replay_against_incidents(
    replay_df: pd.DataFrame,
    incidents: Any,
    *,
    alert_col: str = "alert",
    timestamp_col: str = "timestamp",
    tolerance_hours: float = 2.0,
) -> dict[str, Any]:
    """Return detailed per-alert and per-incident evaluation diagnostics."""
    tol = pd.Timedelta(hours=tolerance_hours)
    if replay_df.empty:
        return {
            "summary": {
                "alerts": 0,
                "incidents": 0,
                "covered_incidents": 0,
                "missed_incidents": 0,
                "in_window": 0,
                "early": 0,
                "late": 0,
                "spurious": 0,
            },
            "alerts_df": pd.DataFrame(),
            "incidents_df": pd.DataFrame(),
        }

    alerts = replay_df.loc[replay_df[alert_col].fillna(False), timestamp_col].copy()
    alert_times = pd.to_datetime(alerts, errors="coerce").dropna().sort_values().tolist()
    incident_windows = _normalize_incidents(incidents)

    alert_rows: list[dict[str, Any]] = []
    for alert_idx, alert_ts in enumerate(alert_times):
        if not incident_windows:
            alert_rows.append(
                {
                    "alert_idx": int(alert_idx),
                    "timestamp": alert_ts,
                    "classification": "spurious",
                    "nearest_incident_idx": pd.NA,
                    "nearest_incident_start": pd.NaT,
                    "nearest_incident_end": pd.NaT,
                    "delta_to_grace": pd.NaT,
                    "delta_to_grace_hours": pd.NA,
                }
            )
            continue

        best_idx: int | None = None
        best_delta: pd.Timedelta | None = None
        for inc_idx, inc in enumerate(incident_windows):
            delta = _signed_delta_to_expanded_window(alert_ts, inc["start"], inc["end"], tol)
            if best_delta is None or abs(delta) < abs(best_delta):
                best_idx = inc_idx
                best_delta = delta

        assert best_idx is not None and best_delta is not None
        nearest = incident_windows[best_idx]
        if best_delta == pd.Timedelta(0):
            classification = "in-window"
        elif best_delta < pd.Timedelta(0):
            classification = "early"
        else:
            classification = "late"

        alert_rows.append(
            {
                "alert_idx": int(alert_idx),
                "timestamp": alert_ts,
                "classification": classification,
                "nearest_incident_idx": int(best_idx),
                "nearest_incident_start": nearest["start"],
                "nearest_incident_end": nearest["end"],
                "delta_to_grace": best_delta,
                "delta_to_grace_hours": round(best_delta.total_seconds() / 3600.0, 3),
            }
        )

    incident_rows: list[dict[str, Any]] = []
    for inc_idx, inc in enumerate(incident_windows):
        matching_alerts = [
            alert_ts for alert_ts in alert_times
            if (inc["start"] - tol) <= alert_ts <= (inc["end"] + tol)
        ]

        best_alert_ts = pd.NaT
        best_delta = pd.NaT
        best_hours = pd.NA
        if alert_times:
            winner: tuple[pd.Timestamp, pd.Timedelta] | None = None
            for alert_ts in alert_times:
                delta = _signed_delta_to_expanded_window(alert_ts, inc["start"], inc["end"], tol)
                if winner is None or abs(delta) < abs(winner[1]):
                    winner = (alert_ts, delta)
            if winner is not None:
                best_alert_ts = winner[0]
                best_delta = winner[1]
                best_hours = round(winner[1].total_seconds() / 3600.0, 3)

        incident_rows.append(
            {
                "incident_idx": int(inc_idx),
                "start": inc["start"],
                "end": inc["end"],
                "hit": bool(matching_alerts),
                "matching_alert_count": int(len(matching_alerts)),
                "first_matching_alert": matching_alerts[0] if matching_alerts else pd.NaT,
                "nearest_alert": best_alert_ts,
                "nearest_alert_delta_to_grace": best_delta,
                "nearest_alert_delta_hours": best_hours,
            }
        )

    alerts_df = pd.DataFrame(alert_rows)
    incidents_df = pd.DataFrame(incident_rows)
    summary = {
        "alerts": int(len(alert_rows)),
        "incidents": int(len(incident_rows)),
        "covered_incidents": int(incidents_df["hit"].sum()) if not incidents_df.empty else 0,
        "missed_incidents": int((~incidents_df["hit"]).sum()) if not incidents_df.empty else 0,
        "in_window": int((alerts_df["classification"] == "in-window").sum()) if not alerts_df.empty else 0,
        "early": int((alerts_df["classification"] == "early").sum()) if not alerts_df.empty else 0,
        "late": int((alerts_df["classification"] == "late").sum()) if not alerts_df.empty else 0,
        "spurious": int((alerts_df["classification"] == "spurious").sum()) if not alerts_df.empty else 0,
    }
    return {
        "summary": summary,
        "alerts_df": alerts_df,
        "incidents_df": incidents_df,
    }
