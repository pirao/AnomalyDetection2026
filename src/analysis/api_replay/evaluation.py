"""Multi-scenario private-benchmark orchestration, metrics, and diagnostics.

``run_inference_test_evaluation`` is the canonical evaluation entry point —
it mirrors the private evaluation logic in ``src/tests/test_evaluation.py``, prints the
aggregate metrics plus a compact per-scenario coverage table, and optionally
explores group reassignments for the worst performers.

``diagnose_replay_against_incidents`` is a separate per-alert / per-incident
classifier used by the API replay widget to explain early/late/spurious
alerts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from sample_processing.model.alert_engine import AlertEngine
from sample_processing.model.anomaly_model import AnomalyModel, load_alert_params, load_pipeline_params
from sample_processing.model.interface import AlertParams
from sample_processing.model.scenario_groups import (
    GROUP_DEFINITIONS,
    get_scenario_group_key,
    get_scenario_group_label,
)

from .batching import df_to_timeseries
from .incidents import (
    _alert_hits_incident_window,
    _normalize_incidents,
    _serialize_incident_window,
    load_incidents_by_scenario,
)
from .simulation import DEFAULT_DATA_DIR, simulate_api_replay_one_scenario

_DEFAULT_GRACE_HOURS = 2.0
_PRECISION_THRESHOLD = 0.50
_RECALL_THRESHOLD = 0.30
_F1_THRESHOLD = 0.35
_STATUS_SEVERITY = {"FN": 4, "PARTIAL": 3, "FP": 2, "TP": 1, "TN": 0}
_NOTEBOOK_TEST_GROUPS = [
    ("test_no_alert_when_no_incident", lambda df: df["n_incidents"] == 0),
    ("test_alert_fires_in_incident_window_single", lambda df: df["n_incidents"] == 1),
    ("test_at_least_one_alert_in_any_incident_window_multi", lambda df: df["n_incidents"] >= 2),
    ("test_every_incident_window_gets_an_alert", lambda df: df["n_incidents"] >= 2),
]


def summarize_inference_test_metrics(
    alerts_by_scenario: dict[int, list[str]],
    incidents_by_scenario: dict[int, list[dict[str, Any]]],
    *,
    scenario_ids: list[int] | None = None,
    tolerance_hours: float = _DEFAULT_GRACE_HOURS,
    precision_threshold: float = _PRECISION_THRESHOLD,
    recall_threshold: float = _RECALL_THRESHOLD,
    f1_threshold: float = _F1_THRESHOLD,
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

        scenario_rows.append(
            {
                "scenario_id": int(sid),
                "scenario_group": get_scenario_group_key(sid),
                "scenario_group_label": get_scenario_group_label(sid),
                "n_incidents": int(len(incident_windows)),
                "n_alerts": int(len(alert_times)),
                "alerts": [ts.isoformat() for ts in alert_times],
                "incident_windows": [_serialize_incident_window(inc) for inc in incident_windows],
                "has_alert_in_window": bool(has_alert_in_window),
                "covered_incident_count": int(len(covered_idx)),
                "covered_incident_windows": [
                    _serialize_incident_window(incident_windows[idx]) for idx in covered_idx
                ],
                "missed_incident_count": int(len(missed_idx)),
                "missed_incident_windows": [
                    _serialize_incident_window(incident_windows[idx]) for idx in missed_idx
                ],
                "all_incident_windows_hit": bool(all_incident_windows_hit),
                "status": status,
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

    summary = {
        "tp": int(f1_tp),
        "fp": int(f1_fp),
        "fn": int(f1_fn),
        "tn": int(tn),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
    }
    thresholds = {
        "precision_min": float(precision_threshold),
        "recall_min": float(recall_threshold),
        "f1_min": float(f1_threshold),
        "precision_pass": bool(precision >= precision_threshold),
        "recall_pass": bool(recall >= recall_threshold),
        "f1_pass": bool(f1 >= f1_threshold),
        "all_pass": bool(
            precision >= precision_threshold
            and recall >= recall_threshold
            and f1 >= f1_threshold
        ),
        "precision_counts": {"tp": int(precision_tp), "fp": int(precision_fp)},
        "recall_counts": {"tp": int(recall_tp), "fn": int(recall_fn)},
        "f1_counts": {"tp": int(f1_tp), "fp": int(f1_fp), "fn": int(f1_fn), "tn": int(tn)},
    }
    return {
        "summary": summary,
        "thresholds": thresholds,
        "scenarios_df": scenarios_df,
    }


def build_inference_test_metric_cards_df(report: dict[str, Any]) -> pd.DataFrame:
    """Return notebook-friendly aggregate metric cards for test-aligned evaluation."""
    summary = dict(report.get("summary", {}))
    thresholds = dict(report.get("thresholds", {}))
    return pd.DataFrame(
        [
            {
                "metric": "precision",
                "value": float(summary.get("precision", 0.0)),
                "threshold": float(thresholds.get("precision_min", 0.0)),
                "pass": bool(thresholds.get("precision_pass", False)),
                "formula": "TP / (TP + FP)",
            },
            {
                "metric": "recall",
                "value": float(summary.get("recall", 0.0)),
                "threshold": float(thresholds.get("recall_min", 0.0)),
                "pass": bool(thresholds.get("recall_pass", False)),
                "formula": "TP / (TP + FN)",
            },
            {
                "metric": "f1",
                "value": float(summary.get("f1", 0.0)),
                "threshold": float(thresholds.get("f1_min", 0.0)),
                "pass": bool(thresholds.get("f1_pass", False)),
                "formula": "2PR / (P + R)",
            },
        ]
    )


def build_inference_test_confusion_matrix_df(
    report: dict[str, Any],
    *,
    normalize: Literal["count", "row"] = "count",
) -> pd.DataFrame:
    """Return the test-aligned scenario confusion matrix for notebook display."""
    summary = dict(report.get("summary", {}))
    matrix = pd.DataFrame(
        [
            [int(summary.get("tp", 0)), int(summary.get("fn", 0))],
            [int(summary.get("fp", 0)), int(summary.get("tn", 0))],
        ],
        index=["actual positive", "actual negative"],
        columns=["predicted positive", "predicted negative"],
    )
    if normalize == "count":
        return matrix
    if normalize == "row":
        row_sums = matrix.sum(axis=1).replace(0, pd.NA)
        return matrix.div(row_sums, axis=0).fillna(0.0)
    raise ValueError("normalize must be 'count' or 'row'")


def build_inference_test_scenario_coverage_df(report: dict[str, Any]) -> pd.DataFrame:
    """Return the notebook-friendly scenario coverage table."""
    scenarios_df = report.get("scenarios_df", pd.DataFrame())
    if not isinstance(scenarios_df, pd.DataFrame) or scenarios_df.empty:
        return pd.DataFrame()
    cols = [
        "scenario_id",
        "scenario_group",
        "scenario_group_label",
        "status",
        "n_incidents",
        "covered_incident_count",
        "missed_incident_count",
        "n_alerts",
        "has_alert_in_window",
        "all_incident_windows_hit",
    ]
    available = [col for col in cols if col in scenarios_df.columns]
    return scenarios_df.loc[:, available].copy()


def build_inference_test_blocking_scenarios_df(report: dict[str, Any]) -> pd.DataFrame:
    """Return only FN/PARTIAL scenarios that still need coverage work."""
    coverage_df = build_inference_test_scenario_coverage_df(report)
    if coverage_df.empty or "status" not in coverage_df.columns:
        return pd.DataFrame()
    blocking_df = coverage_df.loc[coverage_df["status"].isin(["FN", "PARTIAL"])].copy()
    if blocking_df.empty:
        return blocking_df
    return blocking_df.sort_values(
        ["status", "missed_incident_count", "scenario_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _split_pass_fail_scenarios_for_test(
    coverage_df: pd.DataFrame,
    *,
    test_name: str,
    scenario_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Return scenario ids that pass or fail a specific test_evaluation.py case."""
    passed: list[int] = []
    failed: list[int] = []
    for sid in scenario_ids:
        row = coverage_df.loc[sid]
        has_alert_in_window = bool(row["has_alert_in_window"])
        all_hit = bool(row["all_incident_windows_hit"])
        n_alerts = int(row["n_alerts"])

        if test_name == "test_no_alert_when_no_incident":
            ok = n_alerts == 0
        elif test_name in {
            "test_alert_fires_in_incident_window_single",
            "test_at_least_one_alert_in_any_incident_window_multi",
        }:
            ok = has_alert_in_window
        elif test_name == "test_every_incident_window_gets_an_alert":
            ok = has_alert_in_window and all_hit
        else:
            ok = False

        (passed if ok else failed).append(int(sid))
    return passed, failed


def build_inference_test_per_test_results_df(report: dict[str, Any]) -> pd.DataFrame:
    """Return notebook-friendly per-test results aligned to test_evaluation.py."""
    coverage_df = report.get("scenario_coverage_df", pd.DataFrame())
    if not isinstance(coverage_df, pd.DataFrame) or coverage_df.empty:
        coverage_df = build_inference_test_scenario_coverage_df(report)
    if not isinstance(coverage_df, pd.DataFrame) or coverage_df.empty:
        return pd.DataFrame(
            columns=["test", "parametrization", "passed", "failed", "failing_scenarios"]
        )

    indexed = coverage_df.set_index("scenario_id", drop=False)
    rows: list[dict[str, Any]] = []
    for test_name, selector in _NOTEBOOK_TEST_GROUPS:
        scenario_ids = sorted(indexed.loc[selector(indexed)].index.tolist())
        passed, failed = _split_pass_fail_scenarios_for_test(
            indexed,
            test_name=test_name,
            scenario_ids=scenario_ids,
        )
        rows.append(
            {
                "test": test_name,
                "parametrization": f"{len(scenario_ids)} scenarios",
                "passed": len(passed),
                "failed": len(failed),
                "failing_scenarios": ", ".join(str(sid) for sid in failed) if failed else "-",
            }
        )
    return pd.DataFrame(rows)


def build_incident_window_confusion_matrix_df(
    report: dict[str, Any],
    *,
    normalize: Literal["count", "row"] = "count",
) -> pd.DataFrame:
    """Window-level confusion matrix in the same format as the scenario-level one.

    Each incident window is one instance (actual positive). Each no-incident
    scenario is one instance (actual negative). A PARTIAL scenario with 2
    incident windows contributes 1 TP + 1 FN instead of a single scenario-level TP.

    TP = incident windows covered by at least one alert.
    FN = incident windows with no alert.
    FP = no-incident scenarios that fired at least one alert.
    TN = no-incident scenarios with no alerts.
    """
    scenarios_df = report.get("scenarios_df", pd.DataFrame())
    if not isinstance(scenarios_df, pd.DataFrame) or scenarios_df.empty:
        return pd.DataFrame()

    tp = int(scenarios_df["covered_incident_count"].sum())
    fn = int(scenarios_df["missed_incident_count"].sum())
    no_incident = scenarios_df.loc[scenarios_df["n_incidents"] == 0]
    fp = int((no_incident["n_alerts"] > 0).sum())
    tn = int((no_incident["n_alerts"] == 0).sum())

    matrix = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["actual positive", "actual negative"],
        columns=["predicted positive", "predicted negative"],
    )
    if normalize == "count":
        return matrix
    if normalize == "row":
        row_sums = matrix.sum(axis=1).replace(0, pd.NA)
        return matrix.div(row_sums, axis=0).fillna(0.0)
    raise ValueError("normalize must be 'count' or 'row'")


def build_inference_test_notebook_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Assemble notebook-oriented summary artifacts from a test-aligned report."""
    metric_cards_df = build_inference_test_metric_cards_df(report)
    confusion_matrix_df = build_inference_test_confusion_matrix_df(report, normalize="count")
    confusion_matrix_row_pct_df = build_inference_test_confusion_matrix_df(report, normalize="row")
    scenario_coverage_df = build_inference_test_scenario_coverage_df(report)
    per_test_df = build_inference_test_per_test_results_df(
        {"scenario_coverage_df": scenario_coverage_df}
    )
    blocking_scenarios_df = build_inference_test_blocking_scenarios_df(report)
    window_confusion_matrix_df = build_incident_window_confusion_matrix_df(report, normalize="count")
    window_confusion_matrix_row_pct_df = build_incident_window_confusion_matrix_df(report, normalize="row")
    interpretation_note = (
        "Scenario matrix: each scenario is one instance regardless of how many incident "
        "windows it has — a PARTIAL (1/2 covered) counts the same as a full TP. "
        "Window matrix: each incident window is one instance, so PARTIAL scenarios "
        "contribute both a TP and a FN."
    )
    return {
        "metric_cards_df": metric_cards_df,
        "confusion_matrix_df": confusion_matrix_df,
        "confusion_matrix_row_pct_df": confusion_matrix_row_pct_df,
        "per_test_df": per_test_df,
        "window_confusion_matrix_df": window_confusion_matrix_df,
        "window_confusion_matrix_row_pct_df": window_confusion_matrix_row_pct_df,
        "scenario_coverage_df": scenario_coverage_df,
        "blocking_scenarios_df": blocking_scenarios_df,
        "interpretation_note": interpretation_note,
    }


def _print_inference_test_report(report: dict[str, Any]) -> None:
    """Print the canonical scenario-level replay metrics and status table."""
    summary = report.get("summary", {})
    print(
        "TP={tp}  FP={fp}  FN={fn}  TN={tn}".format(
            tp=int(summary.get("tp", 0)),
            fp=int(summary.get("fp", 0)),
            fn=int(summary.get("fn", 0)),
            tn=int(summary.get("tn", 0)),
        )
    )
    print(
        "Precision={precision:.2%}  Recall={recall:.2%}  F1={f1:.2f}".format(
            precision=float(summary.get("precision", 0.0)),
            recall=float(summary.get("recall", 0.0)),
            f1=float(summary.get("f1", 0.0)),
        )
    )

    scenarios_df = report.get("scenarios_df", pd.DataFrame())
    if not isinstance(scenarios_df, pd.DataFrame) or scenarios_df.empty:
        print("No scenario rows to display.")
        return

    table_cols = [
        "scenario_id",
        "status",
        "n_incidents",
        "n_alerts",
        "has_alert_in_window",
        "all_incident_windows_hit",
    ]
    available_cols = [col for col in table_cols if col in scenarios_df.columns]
    table_df = scenarios_df.loc[:, available_cols].copy()
    print(table_df.to_string(index=False))

    worst_df = report.get("worst_scenarios_df", pd.DataFrame())
    if isinstance(worst_df, pd.DataFrame) and not worst_df.empty:
        print("\nWorst scenarios:")
        cols = [
            "scenario_id",
            "scenario_group",
            "status",
            "missed_incident_count",
            "covered_incident_count",
            "n_alerts",
        ]
        print(worst_df.loc[:, [c for c in cols if c in worst_df.columns]].to_string(index=False))

    best_alt_df = report.get("best_group_reassignments_df", pd.DataFrame())
    if isinstance(best_alt_df, pd.DataFrame) and not best_alt_df.empty:
        print("\nNon-worse alternative group assignments:")
        cols = [
            "scenario_id",
            "current_group",
            "candidate_group",
            "comparison_outcome",
            "current_status",
            "candidate_status",
            "current_missed_incident_count",
            "candidate_missed_incident_count",
            "current_covered_incident_count",
            "candidate_covered_incident_count",
        ]
        print(best_alt_df.loc[:, [c for c in cols if c in best_alt_df.columns]].to_string(index=False))


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


def _prepare_scenario_frames(
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


def _scenario_ids_from_data_dir(data_dir: Path) -> list[int]:
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
    from sample_processing.api import main as api_main

    pipeline = load_pipeline_params()
    window = pd.Timedelta(hours=float(pipeline.model_window_size_hours))
    stride = pd.Timedelta(hours=float(pipeline.model_window_size_hours - pipeline.window_overlap_hours))
    sensor_id = f"analysis_sensor_{scenario_id}"

    if model_group_key is not None:
        model = AnomalyModel(scenario_id=scenario_id, group_key=model_group_key)
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
    current = pred_df[time_col].iloc[0]
    end_ts = pred_df[time_col].iloc[-1]
    while current <= end_ts:
        batch_df = pred_df.loc[
            (pred_df[time_col] >= current) & (pred_df[time_col] < current + window)
        ].copy()
        if not batch_df.empty:
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
        current += stride
    api_main._models.pop(sensor_id, None)
    api_main._engines.pop(sensor_id, None)
    return alerts


def _alert_timestamps_from_replay_df(replay_df: pd.DataFrame) -> list[str]:
    if replay_df.empty or "alert" not in replay_df.columns or "timestamp" not in replay_df.columns:
        return []
    alert_rows = replay_df.loc[replay_df["alert"].fillna(False)].copy()
    if alert_rows.empty:
        return []
    timestamps = pd.to_datetime(alert_rows["timestamp"], errors="coerce", utc=True).dropna().sort_values()
    return [ts.isoformat() for ts in timestamps.tolist()]


def _run_fast_replay_scenario(
    *,
    scenario_id: int,
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    time_col: str,
    alert_params: AlertParams,
    model: AnomalyModel | None = None,
    model_group_key: str | None = None,
) -> list[str]:
    effective_model = model
    if model_group_key is not None:
        effective_model = AnomalyModel(scenario_id=scenario_id, group_key=model_group_key)
        effective_model.fit(df_to_timeseries(fit_df, time_col=time_col))

    replay_df = simulate_api_replay_one_scenario(
        fit_df,
        pred_df,
        model=effective_model,
        alert_params=alert_params,
        sensor_id=f"analysis_sensor_{scenario_id}",
        time_col=time_col,
    )
    return _alert_timestamps_from_replay_df(replay_df)


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


def _candidate_better(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_key = (
        int(candidate.get("missed_incident_count", 0)),
        int(_STATUS_SEVERITY.get(str(candidate.get("status", "")), -1)),
        int(candidate.get("n_alerts", 0)),
    )
    current_key = (
        int(current.get("missed_incident_count", 0)),
        int(_STATUS_SEVERITY.get(str(current.get("status", "")), -1)),
        int(current.get("n_alerts", 0)),
    )
    return (
        candidate_key[0] < current_key[0]
        or (candidate_key[0] == current_key[0] and candidate_key[1] < current_key[1])
        or (
            candidate_key[0] == current_key[0]
            and candidate_key[1] == current_key[1]
            and candidate_key[2] < current_key[2]
        )
    )


def _candidate_comparison_outcome(candidate: dict[str, Any], current: dict[str, Any]) -> str:
    candidate_key = (
        int(candidate.get("missed_incident_count", 0)),
        int(_STATUS_SEVERITY.get(str(candidate.get("status", "")), -1)),
        int(candidate.get("n_alerts", 0)),
    )
    current_key = (
        int(current.get("missed_incident_count", 0)),
        int(_STATUS_SEVERITY.get(str(current.get("status", "")), -1)),
        int(current.get("n_alerts", 0)),
    )
    if candidate_key < current_key:
        return "strictly_better"
    if candidate_key == current_key:
        return "marginal"
    return "worse"


def run_inference_test_evaluation(
    *,
    full_df: pd.DataFrame | None = None,
    data_dir: Path | str | None = None,
    labels_path: Path | str | None = None,
    alert_params: AlertParams | None = None,
    alert_params_path: Path | str | None = None,
    models: dict[int, AnomalyModel] | None = None,
    model_version: int | str | None = None,
    execution_mode: Literal["api_exact", "replay_fast"] = "replay_fast",
    include_group_reassignment_analysis: bool = True,
    worst_n: int = 5,
    scenario_ids: list[int] | None = None,
    scenario_col: str = "scenario_id",
    split_col: str = "split",
    fit_value: str = "fit",
    pred_value: str = "pred",
    time_col: str = "sampled_at",
) -> dict[str, Any]:
    """Run the API/test replay protocol and print canonical scenario-level metrics."""
    from sample_processing.api import main as api_main

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
        ordered_ids = [int(s) for s in scenario_ids] if scenario_ids is not None else _scenario_ids_from_data_dir(resolved_data_dir)
        df = None

    effective_alert_params = (
        alert_params
        if alert_params is not None
        else load_alert_params(Path(alert_params_path)) if alert_params_path is not None
        else load_alert_params()
    )
    cache_meta: dict[str, Any] | None = None
    effective_models = models
    if effective_models is None and model_version is not None:
        from analysis.model_cache import load_fitted_models

        effective_models, cache_meta = load_fitted_models(model_version)

    api_main._ALERT_PARAMS = effective_alert_params
    api_main._models.clear()
    api_main._engines.clear()

    scenario_frames = _prepare_scenario_frames(
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
    if execution_mode == "api_exact":
        from fastapi.testclient import TestClient

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
    else:
        for sid in ordered_ids:
            fit_df, pred_df = scenario_frames[int(sid)]
            if fit_df.empty or pred_df.empty:
                alerts_by_scenario[int(sid)] = []
                continue
            alerts_by_scenario[int(sid)] = _run_fast_replay_scenario(
                scenario_id=int(sid),
                fit_df=fit_df,
                pred_df=pred_df,
                time_col=time_col,
                alert_params=effective_alert_params,
                model=None if effective_models is None else effective_models.get(int(sid)),
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
    report["execution_mode"] = execution_mode
    report["alert_params"] = effective_alert_params.model_dump()
    report["alerts_by_scenario"] = alerts_by_scenario
    if cache_meta is not None:
        report["model_cache_meta"] = cache_meta
    notebook_summary = build_inference_test_notebook_summary(report)
    report.update(notebook_summary)

    scenarios_df = report.get("scenarios_df", pd.DataFrame())
    worst_df = _worst_scenarios_df(scenarios_df, worst_n=worst_n)
    report["worst_scenarios_df"] = worst_df

    if include_group_reassignment_analysis and not worst_df.empty:
        reassignment_rows: list[dict[str, Any]] = []
        if execution_mode == "api_exact":
            from fastapi.testclient import TestClient

            client_ctx = TestClient(api_main.app)
        else:
            client_ctx = None
        try:
            client = client_ctx.__enter__() if client_ctx is not None else None
            for row in worst_df.to_dict(orient="records"):
                sid = int(row["scenario_id"])
                current_group = str(row["scenario_group"])
                fit_df, pred_df = scenario_frames[sid]
                candidate_rows: list[dict[str, Any]] = []
                for candidate_group in GROUP_DEFINITIONS.keys():
                    if candidate_group == current_group:
                        continue
                    if execution_mode == "api_exact":
                        alt_alerts = _run_api_scenario(
                            client,
                            scenario_id=sid,
                            fit_df=fit_df,
                            pred_df=pred_df,
                            time_col=time_col,
                            model_group_key=candidate_group,
                        )
                    else:
                        alt_alerts = _run_fast_replay_scenario(
                            scenario_id=sid,
                            fit_df=fit_df,
                            pred_df=pred_df,
                            time_col=time_col,
                            alert_params=effective_alert_params,
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
        finally:
            if client_ctx is not None:
                client_ctx.__exit__(None, None, None)
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
