"""Inference-test metric computation and the canonical evaluation entry point.

``run_inference_test_evaluation`` orchestrates the multi-scenario benchmark;
``summarize_inference_test_metrics`` is the shared metric computation reused by
tests, tracking, and the diagnostics module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_detection.model.grouped_residual.alerting import AlertEngine
from anomaly_detection.model.grouped_residual.detector import GroupedResidualDetector
from anomaly_detection.model.grouped_residual.params import AlertParams, load_alert_params
from anomaly_detection.model.shared.config import load_pipeline_params
from anomaly_detection.model.shared.scenario_groups import (
    GROUP_DEFINITIONS,
    get_scenario_group_key,
    get_scenario_group_label,
)

from .batching import df_to_timeseries, iter_time_batches
from .incidents import (
    _alert_hits_incident_window,
    _normalize_incidents,
    _serialize_incident_window,
    load_incidents_by_scenario,
)
from .report_tables import build_inference_test_notebook_summary
from .simulation import DEFAULT_DATA_DIR

_DEFAULT_GRACE_HOURS = 2.0


_STATUS_SEVERITY = {"FN": 4, "PARTIAL": 3, "FP": 2, "TP": 1, "TN": 0}


def summarize_inference_test_metrics(
    alerts_by_scenario: dict[int, list[str]],
    incidents_by_scenario: dict[int, list[dict[str, Any]]],
    *,
    scenario_ids: list[int] | None = None,
    tolerance_hours: float = _DEFAULT_GRACE_HOURS,
) -> dict[str, Any]:
    """Summarize alerts using the exact scenario-level semantics from test_evaluation.py."""
    tol = pd.Timedelta(hours=float(tolerance_hours))
    all_ids = sorted(
        set(int(k) for k in alerts_by_scenario.keys()) | set(int(k) for k in incidents_by_scenario.keys())
    )
    ordered_ids = [int(s) for s in scenario_ids] if scenario_ids is not None else all_ids

    scenario_rows: list[dict[str, Any]] = []
    for sid in ordered_ids:
        incident_windows = _normalize_incidents(incidents_by_scenario.get(int(sid), []))
        alert_times = (
            pd.to_datetime(pd.Series(alerts_by_scenario.get(int(sid), [])), errors="coerce", utc=True)
            .dropna()
            .sort_values()
            .tolist()
        )

        covered_idx = [
            idx
            for idx, inc in enumerate(incident_windows)
            if any(_alert_hits_incident_window(alert_ts, inc, tolerance=tol) for alert_ts in alert_times)
        ]
        missed_idx = [idx for idx in range(len(incident_windows)) if idx not in covered_idx]
        has_incident = bool(incident_windows)
        has_alert_in_window = bool(covered_idx)
        all_incident_windows_hit = (len(missed_idx) == 0) if incident_windows else True

        if not has_incident and not alert_times:
            status = "TN"
        elif not has_incident and alert_times:
            status = "FP"
        elif has_alert_in_window and all_incident_windows_hit:
            status = "TP"
        elif has_alert_in_window:
            status = "PARTIAL"
        else:
            status = "FN"

        _n_alerts = int(len(alert_times))
        _n_covered = int(len(covered_idx))
        _alert_efficiency = (
            round(_n_covered / _n_alerts, 3) if _n_alerts > 0 else (1.0 if not has_incident else 0.0)
        )

        scenario_rows.append(
            {
                "scenario_id": int(sid),
                "scenario_group": get_scenario_group_key(sid),
                "scenario_group_label": get_scenario_group_label(sid),
                "n_incidents": int(len(incident_windows)),
                "n_alerts": _n_alerts,
                "alerts": [ts.isoformat() for ts in alert_times],
                "incident_windows": [_serialize_incident_window(inc) for inc in incident_windows],
                "has_alert_in_window": bool(has_alert_in_window),
                "covered_incident_count": _n_covered,
                "covered_incident_windows": [
                    _serialize_incident_window(incident_windows[idx]) for idx in covered_idx
                ],
                "missed_incident_count": int(len(missed_idx)),
                "missed_incident_windows": [
                    _serialize_incident_window(incident_windows[idx]) for idx in missed_idx
                ],
                "all_incident_windows_hit": bool(all_incident_windows_hit),
                "status": status,
                "alert_efficiency": _alert_efficiency,
            }
        )

    scenarios_df = pd.DataFrame(scenario_rows)
    if not scenarios_df.empty:
        scenarios_df = scenarios_df.sort_values("scenario_id").reset_index(drop=True)

    precision_tp = precision_fp = 0
    recall_tp = recall_fn = 0
    f1_tp = f1_fp = f1_fn = tn = 0

    for row in scenario_rows:
        has_incident = bool(row["n_incidents"])
        has_alerts = bool(row["n_alerts"])
        has_alert_in_window = bool(row["has_alert_in_window"])

        if has_alerts:
            if has_alert_in_window:
                precision_tp += 1
            else:
                precision_fp += 1

        if has_incident:
            if has_alert_in_window:
                recall_tp += 1
            else:
                recall_fn += 1

        if has_alert_in_window:
            f1_tp += 1
        elif has_alerts and not has_incident:
            f1_fp += 1
        elif has_incident and not has_alert_in_window:
            f1_fn += 1
        elif not has_incident and not has_alerts:
            tn += 1

    precision = precision_tp / (precision_tp + precision_fp) if (precision_tp + precision_fp) else 0.0
    recall = recall_tp / (recall_tp + recall_fn) if (recall_tp + recall_fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    _agg_covered = sum(r["covered_incident_count"] for r in scenario_rows)
    _agg_alerts = sum(r["n_alerts"] for r in scenario_rows)
    _agg_efficiency = round(_agg_covered / _agg_alerts, 3) if _agg_alerts > 0 else 1.0

    summary = {
        "tp": int(f1_tp),
        "fp": int(f1_fp),
        "fn": int(f1_fn),
        "tn": int(tn),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
        "total_alerts": int(_agg_alerts),
        "alert_efficiency": _agg_efficiency,
    }
    return {
        "summary": summary,
        "scenarios_df": scenarios_df,
    }


def _df_to_api_payload(
    df: pd.DataFrame,
    *,
    time_col: str = "sampled_at",
) -> list[dict[str, Any]]:
    rows = (
        df.sort_values(time_col)
        .reset_index(drop=True)
    )
    return [
        {
            "timestamp": row[time_col].isoformat(),
            "uptime": bool(row["uptime"]),
            "vel_rms_x": float(row["vel_rms_x"]),
            "vel_rms_y": float(row["vel_rms_y"]),
            "vel_rms_z": float(row["vel_rms_z"]),
            "accel_rms_x": float(row["accel_rms_x"]),
            "accel_rms_y": float(row["accel_rms_y"]),
            "accel_rms_z": float(row["accel_rms_z"]),
        }
        for _, row in rows.iterrows()
    ]


def _load_scenario_frames_from_full_df(
    full_df: pd.DataFrame,
    *,
    scenario_id: int,
    scenario_col: str,
    split_col: str,
    fit_value: str,
    pred_value: str,
    time_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sid = full_df[full_df[scenario_col] == scenario_id].copy()
    fit_df = df_sid[df_sid[split_col] == fit_value].copy()
    pred_df = df_sid[df_sid[split_col] == pred_value].copy()
    fit_df[time_col] = pd.to_datetime(fit_df[time_col], errors="coerce", utc=True)
    pred_df[time_col] = pd.to_datetime(pred_df[time_col], errors="coerce", utc=True)
    fit_df = fit_df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    pred_df = pred_df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return fit_df, pred_df


def _load_scenario_frames_from_data_dir(
    data_dir: Path,
    *,
    scenario_id: int,
    time_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        fit_df = pd.read_parquet(data_dir / f"sensor_data_fit_{scenario_id}.parquet")
        pred_df = pd.read_parquet(data_dir / f"sensor_data_pred_{scenario_id}.parquet")
    except OSError as exc:
        raise RuntimeError(
            f"Failed to read parquet files for scenario {scenario_id} from {data_dir}. "
            "If you already have the dataset loaded in memory, pass `full_df=` to "
            "`run_inference_test_evaluation(...)`."
        ) from exc
    fit_df[time_col] = pd.to_datetime(fit_df[time_col], errors="coerce", utc=True)
    pred_df[time_col] = pd.to_datetime(pred_df[time_col], errors="coerce", utc=True)
    fit_df = fit_df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    pred_df = pred_df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return fit_df, pred_df


def prepare_scenario_frames(
    *,
    full_df: pd.DataFrame | None,
    data_dir: Path,
    ordered_ids: list[int],
    scenario_col: str,
    split_col: str,
    fit_value: str,
    pred_value: str,
    time_col: str,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Load and normalize all requested scenario frames once for reuse."""
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    if full_df is not None:
        df = full_df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.dropna(subset=[time_col]).copy()
        for sid in ordered_ids:
            frames[int(sid)] = _load_scenario_frames_from_full_df(
                df,
                scenario_id=int(sid),
                scenario_col=scenario_col,
                split_col=split_col,
                fit_value=fit_value,
                pred_value=pred_value,
                time_col=time_col,
            )
        return frames

    for sid in ordered_ids:
        frames[int(sid)] = _load_scenario_frames_from_data_dir(
            data_dir,
            scenario_id=int(sid),
            time_col=time_col,
        )
    return frames


def scenario_ids_from_data_dir(data_dir: Path) -> list[int]:
    ids: set[int] = set()
    for path in data_dir.glob("sensor_data_fit_*.parquet"):
        try:
            ids.add(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(ids)


def _run_api_scenario(
    client,
    *,
    scenario_id: int,
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    time_col: str,
    model_group_key: str | None = None,
) -> list[str]:
    from anomaly_detection.api import main as api_main

    pipeline = load_pipeline_params()
    sensor_id = f"analysis_sensor_{scenario_id}"

    if model_group_key is not None:
        model = GroupedResidualDetector(scenario_id=scenario_id, group_key=model_group_key)
        model.fit(df_to_timeseries(fit_df, time_col=time_col))
        api_main._models[sensor_id] = model
        api_main._engines[sensor_id] = AlertEngine(api_main._ALERT_PARAMS)
    else:
        api_main._models.pop(sensor_id, None)
        api_main._engines.pop(sensor_id, None)

    if model_group_key is None:
        fit_resp = client.post(
            "/fit",
            json={"sensor_id": sensor_id, "data": _df_to_api_payload(fit_df, time_col=time_col)},
        )
        if fit_resp.status_code != 200:
            raise RuntimeError(f"Fit failed for scenario {scenario_id}: {fit_resp.text}")

    alerts: list[str] = []
    for _, current, _, batch_df in iter_time_batches(
        pred_df,
        window_hours=pipeline.model_window_size_hours,
        overlap_hours=pipeline.window_overlap_hours,
        time_col=time_col,
    ):
        pred_resp = client.post(
            "/predict",
            json={"sensor_id": sensor_id, "data": _df_to_api_payload(batch_df, time_col=time_col)},
        )
        if pred_resp.status_code != 200:
            raise RuntimeError(
                f"Predict failed for scenario {scenario_id} at {current}: {pred_resp.text}"
            )
        payload = pred_resp.json()
        if bool(payload.get("alert")):
            alerts.append(str(payload["timestamp"]))
    api_main._models.pop(sensor_id, None)
    api_main._engines.pop(sensor_id, None)
    return alerts


def _worst_scenarios_df(
    scenarios_df: pd.DataFrame,
    *,
    worst_n: int,
) -> pd.DataFrame:
    if scenarios_df.empty:
        return pd.DataFrame()
    ranked = scenarios_df.copy()
    ranked["status_severity"] = ranked["status"].map(_STATUS_SEVERITY).fillna(-1).astype(int)
    ranked = ranked.sort_values(
        ["missed_incident_count", "status_severity", "n_alerts", "scenario_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return ranked.head(int(worst_n)).drop(columns=["status_severity"], errors="ignore")


def _candidate_comparison_outcome(candidate: dict[str, Any], current: dict[str, Any]) -> str:
    primary_candidate = (
        int(candidate.get("missed_incident_count", 0)),
        int(_STATUS_SEVERITY.get(str(candidate.get("status", "")), -1)),
    )
    primary_current = (
        int(current.get("missed_incident_count", 0)),
        int(_STATUS_SEVERITY.get(str(current.get("status", "")), -1)),
    )
    if primary_candidate < primary_current:
        return "strictly_better"
    if primary_candidate == primary_current:
        # Primary metrics equal: secondary check is alert count.
        # Fewer or equal alerts = marginal (not worse). More alerts = worse.
        if int(candidate.get("n_alerts", 0)) <= int(current.get("n_alerts", 0)):
            return "marginal"
    return "worse"


def run_inference_test_evaluation(
    *,
    full_df: pd.DataFrame | None = None,
    data_dir: Path | str | None = None,
    labels_path: Path | str | None = None,
    alert_params: AlertParams | None = None,
    alert_params_path: Path | str | None = None,
    include_group_reassignment_analysis: bool = True,
    worst_n: int = 5,
    scenario_ids: list[int] | None = None,
    scenario_col: str = "scenario_id",
    split_col: str = "split",
    fit_value: str = "fit",
    pred_value: str = "pred",
    time_col: str = "sampled_at",
) -> dict[str, Any]:
    """Run the API test replay protocol and print canonical scenario-level metrics.

    Cost note: ``include_group_reassignment_analysis`` (default ``True``) re-fits and
    re-replays the ``worst_n`` scenarios against every alternate scenario group, to
    produce ``best_group_reassignments_df``. That's real work on top of the main
    29-scenario pass - roughly another 50% of total runtime - and it's the only
    difference between this function's cost and ``src/tests/test_evaluation.py``'s
    (which never runs it). Pass ``include_group_reassignment_analysis=False`` if you
    only need the headline precision/recall/F1 and don't need the reassignment diagnostic.
    """
    from fastapi.testclient import TestClient

    from anomaly_detection.api import main as api_main

    resolved_data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    incidents_by_scenario = load_incidents_by_scenario(labels_path)
    if full_df is not None:
        df = full_df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.dropna(subset=[time_col]).copy()
        ordered_ids = (
            [int(s) for s in scenario_ids]
            if scenario_ids is not None
            else sorted(df[scenario_col].dropna().astype(int).unique().tolist())
        )
    else:
        ordered_ids = [int(s) for s in scenario_ids] if scenario_ids is not None else scenario_ids_from_data_dir(resolved_data_dir)
        df = None

    effective_alert_params = (
        alert_params
        if alert_params is not None
        else load_alert_params(Path(alert_params_path)) if alert_params_path is not None
        else load_alert_params()
    )

    api_main._ALERT_PARAMS = effective_alert_params
    api_main._models.clear()
    api_main._engines.clear()

    scenario_frames = prepare_scenario_frames(
        full_df=df,
        data_dir=resolved_data_dir,
        ordered_ids=ordered_ids,
        scenario_col=scenario_col,
        split_col=split_col,
        fit_value=fit_value,
        pred_value=pred_value,
        time_col=time_col,
    )

    alerts_by_scenario: dict[int, list[str]] = {}
    with TestClient(api_main.app) as client:
        for sid in ordered_ids:
            fit_df, pred_df = scenario_frames[int(sid)]
            if fit_df.empty or pred_df.empty:
                alerts_by_scenario[int(sid)] = []
                continue
            alerts_by_scenario[int(sid)] = _run_api_scenario(
                client,
                scenario_id=int(sid),
                fit_df=fit_df,
                pred_df=pred_df,
                time_col=time_col,
            )

    api_main._models.clear()
    api_main._engines.clear()

    report = summarize_inference_test_metrics(
        alerts_by_scenario,
        incidents_by_scenario,
        scenario_ids=ordered_ids,
    )
    pipeline = load_pipeline_params()
    report["protocol"] = {
        "window_hours": float(pipeline.model_window_size_hours),
        "stride_hours": float(pipeline.model_window_size_hours - pipeline.window_overlap_hours),
        "tolerance_hours": float(_DEFAULT_GRACE_HOURS),
    }
    report["alert_params"] = effective_alert_params.model_dump()
    report["alerts_by_scenario"] = alerts_by_scenario
    notebook_summary = build_inference_test_notebook_summary(report)
    report.update(notebook_summary)

    scenarios_df = report.get("scenarios_df", pd.DataFrame())
    worst_df = _worst_scenarios_df(scenarios_df, worst_n=worst_n)
    report["worst_scenarios_df"] = worst_df

    if include_group_reassignment_analysis and not worst_df.empty:
        reassignment_rows: list[dict[str, Any]] = []
        with TestClient(api_main.app) as client:
            for row in worst_df.to_dict(orient="records"):
                sid = int(row["scenario_id"])
                current_group = str(row["scenario_group"])
                fit_df, pred_df = scenario_frames[sid]
                for candidate_group in GROUP_DEFINITIONS.keys():
                    if candidate_group == current_group:
                        continue
                    alt_alerts = _run_api_scenario(
                        client,
                        scenario_id=sid,
                        fit_df=fit_df,
                        pred_df=pred_df,
                        time_col=time_col,
                        model_group_key=candidate_group,
                    )
                    alt_report = summarize_inference_test_metrics(
                        {sid: alt_alerts},
                        incidents_by_scenario,
                        scenario_ids=[sid],
                    )
                    alt_row = alt_report["scenarios_df"].iloc[0].to_dict()
                    candidate_row = {
                        "scenario_id": sid,
                        "current_group": current_group,
                        "current_group_label": get_scenario_group_label(sid),
                        "candidate_group": candidate_group,
                        "candidate_group_label": str(GROUP_DEFINITIONS[candidate_group]["label"]),
                        "current_status": str(row["status"]),
                        "candidate_status": str(alt_row["status"]),
                        "current_n_alerts": int(row["n_alerts"]),
                        "candidate_n_alerts": int(alt_row["n_alerts"]),
                        "current_covered_incident_count": int(row["covered_incident_count"]),
                        "candidate_covered_incident_count": int(alt_row["covered_incident_count"]),
                        "current_missed_incident_count": int(row["missed_incident_count"]),
                        "candidate_missed_incident_count": int(alt_row["missed_incident_count"]),
                        "current_has_alert_in_window": bool(row["has_alert_in_window"]),
                        "candidate_has_alert_in_window": bool(alt_row["has_alert_in_window"]),
                        "current_all_incident_windows_hit": bool(row["all_incident_windows_hit"]),
                        "candidate_all_incident_windows_hit": bool(alt_row["all_incident_windows_hit"]),
                        "delta_n_alerts": int(alt_row["n_alerts"]) - int(row["n_alerts"]),
                    }
                    candidate_metrics = {
                        "missed_incident_count": candidate_row["candidate_missed_incident_count"],
                        "status": candidate_row["candidate_status"],
                        "n_alerts": candidate_row["candidate_n_alerts"],
                    }
                    current_metrics = {
                        "missed_incident_count": candidate_row["current_missed_incident_count"],
                        "status": candidate_row["current_status"],
                        "n_alerts": candidate_row["current_n_alerts"],
                    }
                    candidate_row["comparison_outcome"] = _candidate_comparison_outcome(
                        candidate_metrics,
                        current_metrics,
                    )
                    candidate_row["strictly_better"] = candidate_row["comparison_outcome"] == "strictly_better"
                    candidate_row["not_worse"] = candidate_row["comparison_outcome"] in {
                        "strictly_better",
                        "marginal",
                    }
                    reassignment_rows.append(candidate_row)
        group_reassignment_df = pd.DataFrame(reassignment_rows)
        report["group_reassignment_df"] = group_reassignment_df
        if group_reassignment_df.empty:
            report["best_group_reassignments_df"] = pd.DataFrame()
        else:
            non_worse_df = group_reassignment_df.loc[group_reassignment_df["not_worse"]].copy()
            if not non_worse_df.empty:
                outcome_order = {"strictly_better": 0, "marginal": 1}
                non_worse_df["comparison_rank"] = (
                    non_worse_df["comparison_outcome"].map(outcome_order).fillna(99).astype(int)
                )
                non_worse_df = non_worse_df.sort_values(
                    [
                        "scenario_id",
                        "comparison_rank",
                        "candidate_missed_incident_count",
                        "candidate_covered_incident_count",
                        "candidate_n_alerts",
                        "candidate_group",
                    ],
                    ascending=[True, True, True, False, True, True],
                ).drop(columns=["comparison_rank"])
            report["best_group_reassignments_df"] = non_worse_df
    else:
        report["group_reassignment_df"] = pd.DataFrame()
        report["best_group_reassignments_df"] = pd.DataFrame()

    return report
