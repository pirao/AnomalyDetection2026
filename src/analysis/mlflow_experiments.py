"""MLflow comparison experiment: baseline vs current model.

Run ``compare_baseline_vs_current()`` from a notebook to populate the
"baseline-vs-current" experiment in the SQLite store at ``mlflow.db``.
Then compare the two parent runs with::

    mlflow ui --backend-store-uri sqlite:///mlflow.db

**Baseline files required** — before calling this function you must provide:
    src/sample_processing/model/baseline/anomaly_model.py
    src/sample_processing/model/baseline/alert_engine.py
    src/sample_processing/model/baseline/__init__.py
    src/sample_processing/model/baseline/hyperparameters/model_hyperparams.yaml

The pipeline (windowing) config is shared across models and read from
``src/sample_processing/model/shared/pipeline_hyperparams.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

# MLflow's file store (mlruns/) is deprecated and disabled in current versions,
# so we track to a SQLite DB at the repo root. View with:
#   mlflow ui --backend-store-uri sqlite:///mlflow.db
_TRACKING_URI = f"sqlite:///{(_REPO_ROOT / 'mlflow.db').as_posix()}"


def _evaluate_baseline_scenarios(
    scenario_frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    ordered_ids: list[int],
    pipeline: Any,
    *,
    time_col: str = "sampled_at",
) -> dict[int, list[str]]:
    """Run the baseline model (no API layer) on each scenario.

    Returns a dict mapping scenario_id → list of alert timestamp ISO strings,
    in the same format that ``summarize_inference_test_metrics`` expects.
    """
    from sample_processing.model.baseline.alert_engine import AlertEngine as BaselineAlertEngine
    from sample_processing.model.baseline.anomaly_model import AnomalyModel as BaselineModel

    from analysis.evaluation.batching import df_to_timeseries, iter_time_batches

    alerts_by_scenario: dict[int, list[str]] = {}
    for sid in ordered_ids:
        fit_df, pred_df = scenario_frames[sid]
        if fit_df.empty or pred_df.empty:
            alerts_by_scenario[sid] = []
            continue

        model = BaselineModel()
        model.fit(df_to_timeseries(fit_df, time_col=time_col))
        engine = BaselineAlertEngine()

        alert_timestamps: list[str] = []
        for _idx, _start, _end, batch_df in iter_time_batches(
            pred_df,
            window_hours=pipeline.model_window_size_hours,
            overlap_hours=pipeline.window_overlap_hours,
            time_col=time_col,
        ):
            if batch_df.empty:
                continue
            prediction = model.predict(df_to_timeseries(batch_df, time_col=time_col))
            decision = engine.predict(prediction)
            if decision.alert:
                alert_timestamps.append(str(decision.timestamp))

        alerts_by_scenario[sid] = alert_timestamps

    return alerts_by_scenario


def _build_experiment_report(
    alerts_by_scenario: dict[int, list[str]],
    incidents_by_scenario: dict[int, list[dict[str, Any]]],
    ordered_ids: list[int],
) -> dict[str, Any]:
    """Compute metrics and notebook-summary DataFrames from alert results."""
    from analysis.evaluation.evaluation import (
        build_inference_test_notebook_summary,
        summarize_inference_test_metrics,
    )

    report = summarize_inference_test_metrics(
        alerts_by_scenario,
        incidents_by_scenario,
        scenario_ids=ordered_ids,
    )
    report.update(build_inference_test_notebook_summary(report))
    return report


def _log_model_run(
    report: dict[str, Any],
    *,
    model_tag: str,
    extra_params: dict[str, Any],
    pipeline_params: dict[str, Any],
) -> None:
    """Log one parent run + one nested child per scenario.

    Parent run holds global metrics (precision/recall/F1), per-test-type
    pass/fail counts, and scenario coverage tables as MLflow table artifacts.
    Each nested child holds per-scenario tags and metrics for drill-down
    comparison in the MLflow UI.
    """
    import mlflow

    summary = report["summary"]
    scenario_coverage_df: pd.DataFrame = report.get("scenario_coverage_df", pd.DataFrame())
    blocking_scenarios_df: pd.DataFrame = report.get("blocking_scenarios_df", pd.DataFrame())
    per_test_df: pd.DataFrame = report.get("per_test_df", pd.DataFrame())
    scenarios_df: pd.DataFrame = report.get("scenarios_df", pd.DataFrame())

    all_params = {
        **pipeline_params,
        **{k: v for k, v in extra_params.items() if isinstance(v, (int, float, bool, str))},
    }

    with mlflow.start_run(tags={"model": model_tag}):
        mlflow.log_params(all_params)

        mlflow.log_metrics(
            {
                "precision": float(summary["precision"]),
                "recall": float(summary["recall"]),
                "f1": float(summary["f1"]),
                "tp": float(summary["tp"]),
                "fp": float(summary["fp"]),
                "fn": float(summary["fn"]),
                "tn": float(summary["tn"]),
            }
        )

        if isinstance(per_test_df, pd.DataFrame) and not per_test_df.empty:
            for row in per_test_df.itertuples(index=False):
                key = str(row.test).removeprefix("test_")
                mlflow.log_metrics(
                    {
                        f"test_{key}_passed": int(row.passed),
                        f"test_{key}_failed": int(row.failed),
                    }
                )

        if isinstance(scenario_coverage_df, pd.DataFrame) and not scenario_coverage_df.empty:
            mlflow.log_table(
                data=scenario_coverage_df.to_dict(orient="list"),
                artifact_file="scenario_coverage.json",
            )

        if isinstance(blocking_scenarios_df, pd.DataFrame) and not blocking_scenarios_df.empty:
            mlflow.log_table(
                data=blocking_scenarios_df.to_dict(orient="list"),
                artifact_file="blocking_scenarios.json",
            )

        if isinstance(scenarios_df, pd.DataFrame) and not scenarios_df.empty:
            for row in scenarios_df.itertuples(index=False):
                sid = int(row.scenario_id)
                with mlflow.start_run(nested=True, run_name=f"scenario_{sid}"):
                    mlflow.set_tags(
                        {
                            "scenario_id": str(sid),
                            "scenario_group": str(row.scenario_group),
                            "scenario_group_label": str(row.scenario_group_label),
                            "status": str(row.status),
                        }
                    )
                    mlflow.log_metrics(
                        {
                            "has_alert_in_window": float(bool(row.has_alert_in_window)),
                            "n_alerts": float(int(row.n_alerts)),
                            "n_incidents": float(int(row.n_incidents)),
                            "covered_incident_count": float(int(row.covered_incident_count)),
                            "missed_incident_count": float(int(row.missed_incident_count)),
                        }
                    )


def compare_baseline_vs_current(
    *,
    full_df: pd.DataFrame | None = None,
    data_dir: Path | str | None = None,
    labels_path: Path | str | None = None,
    alert_params_path: Path | str | None = None,
    scenario_ids: list[int] | None = None,
    scenario_col: str = "scenario_id",
    split_col: str = "split",
    fit_value: str = "fit",
    pred_value: str = "pred",
    time_col: str = "sampled_at",
) -> dict[str, dict[str, Any]]:
    """Compare baseline and current model in MLflow experiment "baseline-vs-current".

    Logs two parent runs to the repo-root ``mlruns/`` directory:
    - Run A (model=baseline): velocity-only L2-norm, evaluated directly without API
    - Run B (model=current): 6-channel sigmoid, evaluated through the FastAPI test client

    Each parent run contains:
    - global metrics: precision, recall, f1, tp, fp, fn, tn
    - per-test-type pass/fail counts (mirrors notebook 02 Section 6 ``per_test_df``)
    - ``scenario_coverage.json`` table artifact (all 29 scenarios)
    - ``blocking_scenarios.json`` table artifact (FN/PARTIAL scenarios only)
    - 29 nested child runs, one per scenario, with per-scenario metrics and tags

    **Requires baseline model files** — see module docstring.

    Parameters
    ----------
    full_df :
        Full sensor dataset (all scenarios, both splits). When provided, parquet
        files are not read from disk. Omit to read parquets directly (canonical,
        matches ``make inference-test``).
    data_dir :
        Directory containing ``sensor_data_fit_{id}.parquet`` files. Ignored when
        ``full_df`` is provided. Defaults to ``data/``.
    labels_path :
        Path to ``labels/incidents.yaml``. Defaults to the standard location.
    alert_params_path :
        Path to alert hyperparameters YAML. Defaults to the standard location.
    scenario_ids :
        Subset of scenario IDs to evaluate. Defaults to all available scenarios.

    Returns
    -------
    dict
        ``{"baseline_report": ..., "current_report": ...}`` — the full report
        dicts from each evaluation run, including all notebook summary DataFrames.
    """
    import mlflow

    from analysis.evaluation.evaluation import (
        _prepare_scenario_frames,
        _scenario_ids_from_data_dir,
        run_inference_test_evaluation,
    )
    from analysis.evaluation.incidents import load_incidents_by_scenario
    from analysis.evaluation.simulation import DEFAULT_DATA_DIR
    from sample_processing.model.current.anomaly_model import load_alert_params
    from sample_processing.model.baseline.anomaly_model import (
        AnomalyModel as BaselineModel,
        load_pipeline_params as load_baseline_pipeline_params,
    )

    resolved_data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    incidents_by_scenario = load_incidents_by_scenario(labels_path)

    if full_df is not None:
        _df = full_df.copy()
        _df[time_col] = pd.to_datetime(_df[time_col], errors="coerce", utc=True)
        _df = _df.dropna(subset=[time_col]).copy()
        ordered_ids = (
            [int(s) for s in scenario_ids]
            if scenario_ids is not None
            else sorted(_df[scenario_col].dropna().astype(int).unique().tolist())
        )
    else:
        ordered_ids = (
            [int(s) for s in scenario_ids]
            if scenario_ids is not None
            else _scenario_ids_from_data_dir(resolved_data_dir)
        )
        _df = None

    scenario_frames = _prepare_scenario_frames(
        full_df=_df,
        data_dir=resolved_data_dir,
        ordered_ids=ordered_ids,
        scenario_col=scenario_col,
        split_col=split_col,
        fit_value=fit_value,
        pred_value=pred_value,
        time_col=time_col,
    )

    baseline_pipeline = load_baseline_pipeline_params()
    baseline_model_params = BaselineModel().params

    effective_alert_params = (
        load_alert_params(Path(alert_params_path))
        if alert_params_path is not None
        else load_alert_params()
    )

    pipeline_params: dict[str, Any] = {
        "n_scenarios": len(ordered_ids),
        "window_hours": float(baseline_pipeline.model_window_size_hours),
        "stride_hours": float(
            baseline_pipeline.model_window_size_hours - baseline_pipeline.window_overlap_hours
        ),
    }

    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment("baseline-vs-current")

    # --- Run A: baseline (direct evaluation, no API layer) ---
    print("Evaluating baseline model ...")
    baseline_alerts = _evaluate_baseline_scenarios(
        scenario_frames,
        ordered_ids,
        baseline_pipeline,
        time_col=time_col,
    )
    baseline_report = _build_experiment_report(baseline_alerts, incidents_by_scenario, ordered_ids)
    _log_model_run(
        baseline_report,
        model_tag="baseline",
        extra_params={
            "z_threshold": int(baseline_model_params.z_threshold),
            "window_anomaly_ratio": float(baseline_model_params.window_anomaly_ratio),
        },
        pipeline_params=pipeline_params,
    )

    # --- Run B: current model (via FastAPI test client) ---
    # Re-assert tracking URI / experiment defensively in case the evaluation
    # call below ever touches global MLflow state.
    print("Evaluating current model ...")
    current_report = run_inference_test_evaluation(
        full_df=full_df,
        data_dir=data_dir,
        labels_path=labels_path,
        alert_params=effective_alert_params,
        scenario_ids=ordered_ids,
        scenario_col=scenario_col,
        split_col=split_col,
        fit_value=fit_value,
        pred_value=pred_value,
        time_col=time_col,
        include_group_reassignment_analysis=False,
    )
    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment("baseline-vs-current")

    alert_params_flat = {
        k: v
        for k, v in effective_alert_params.model_dump().items()
        if isinstance(v, (int, float, bool, str))
    }
    _log_model_run(
        current_report,
        model_tag="current",
        extra_params=alert_params_flat,
        pipeline_params=pipeline_params,
    )

    print("Done. Open 'mlflow ui' from the repo root to view results.")
    return {"baseline_report": baseline_report, "current_report": current_report}
