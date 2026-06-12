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

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _data_digest(data_dir: Path) -> str:
    """MD5 fingerprint of all parquet files in data_dir.

    Sorted by filename for determinism across OSes. Includes the filename in
    the hash so that renaming a file (same bytes, different name) also changes
    the digest.
    """
    h = hashlib.md5()
    for f in sorted(data_dir.glob("*.parquet")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]

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
    model_params: dict[str, Any],
    pipeline_params: dict[str, Any],
    alert_params_dict: dict[str, Any] | None = None,
    data_digest: str = "",
    model_cache_version: str = "",
    model_cache_notes: str = "",
) -> None:
    """Log one parent run + one nested child per scenario.

    Parent run params use dot notation:
    - ``pipeline.*``  — windowing/batching config (shared across models)
    - ``model.*``     — model hyperparameters (incl. ``model.groups.group_N.*``)
    - ``alert.*``     — alert engine hyperparameters (current model only)

    Parent metrics: precision/recall/f1 at machine and event level, total_alerts.
    Each nested child holds per-scenario tags and metrics for drill-down.
    """
    import mlflow

    summary = report["summary"]
    scenarios_df: pd.DataFrame = report.get("scenarios_df", pd.DataFrame())
    window_confusion_df: pd.DataFrame = report.get("window_confusion_matrix_df", pd.DataFrame())

    all_params: dict[str, Any] = {f"pipeline.{k}": v for k, v in pipeline_params.items()}
    all_params.update(
        {f"model.{k}": v for k, v in model_params.items() if isinstance(v, (int, float, bool, str))}
    )
    if alert_params_dict:
        all_params.update(
            {f"alert.{k}": v for k, v in alert_params_dict.items() if isinstance(v, (int, float, bool, str))}
        )

    # Delete any existing run for this model_tag so we don't accumulate stale runs.
    client = mlflow.tracking.MlflowClient()
    _exp = client.get_experiment_by_name("baseline-vs-current")
    if _exp is not None:
        for _r in client.search_runs(
            experiment_ids=[_exp.experiment_id],
            filter_string=f"tags.model = '{model_tag}'",
        ):
            client.delete_run(_r.info.run_id)

    with mlflow.start_run(
        run_name=model_tag,
        tags={
            "model": model_tag,
            "data_digest": data_digest,
            "model_cache_version": model_cache_version,
            "model_cache_notes": model_cache_notes,
        },
    ):
        mlflow.log_params(all_params)

        # Per-machine summary (scenario-level; PARTIAL counts as TP)
        mlflow.log_metrics(
            {
                "precision": float(summary["precision"]),
                "recall":    float(summary["recall"]),
                "f1":        float(summary["f1"]),
            }
        )

        # Per-event summary (fault-window-level; PARTIAL → 1 TP + 1 FN)
        if isinstance(window_confusion_df, pd.DataFrame) and not window_confusion_df.empty:
            _e_tp  = int(window_confusion_df.loc["actual positive", "predicted positive"])
            _e_fn  = int(window_confusion_df.loc["actual positive", "predicted negative"])
            _e_fp  = int(window_confusion_df.loc["actual negative", "predicted positive"])
            _e_prec = _e_tp / (_e_tp + _e_fp) if (_e_tp + _e_fp) else 0.0
            _e_rec  = _e_tp / (_e_tp + _e_fn) if (_e_tp + _e_fn) else 0.0
            _e_f1   = 2 * _e_prec * _e_rec / (_e_prec + _e_rec) if (_e_prec + _e_rec) else 0.0
            mlflow.log_metrics(
                {"event_precision": _e_prec, "event_recall": _e_rec, "event_f1": _e_f1}
            )

        if isinstance(scenarios_df, pd.DataFrame) and not scenarios_df.empty:
            _total_covered = float(scenarios_df["covered_incident_count"].sum())
            _total_n_alerts = float(scenarios_df["n_alerts"].sum())
            _alert_eff = _total_covered / _total_n_alerts if _total_n_alerts > 0 else 1.0
            mlflow.log_metrics({
                "total_alerts": _total_n_alerts,
                "alert_efficiency": round(_alert_eff, 3),
            })

            _ds = mlflow.data.from_pandas(
                scenarios_df[["scenario_id", "status", "n_alerts", "n_incidents"]],
                source=str(_REPO_ROOT / "data"),
                name="sensor_scenarios",
                digest=data_digest or None,
            )
            mlflow.log_input(_ds, context="evaluation")

            for row in scenarios_df.itertuples(index=False):
                sid = int(row.scenario_id)
                n_inc = int(row.n_incidents)
                n_al = int(row.n_alerts)
                n_cov = int(row.covered_incident_count)
                ev_recall = n_cov / n_inc if n_inc > 0 else 0.0
                false_alerts = float(n_al) if n_inc == 0 else 0.0
                child_eff = (
                    round(n_cov / n_al, 3) if n_al > 0 else (1.0 if n_inc == 0 else 0.0)
                )
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
                            "n_alerts": float(n_al),
                            "n_incidents": float(n_inc),
                            "covered_incident_count": float(n_cov),
                            "missed_incident_count": float(int(row.missed_incident_count)),
                            "event_recall": ev_recall,
                            "false_alerts": false_alerts,
                            "alert_efficiency": child_eff,
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
    model_version: int | str = "latest",
) -> dict[str, dict[str, Any]]:
    """Compare baseline and current model in MLflow experiment "baseline-vs-current".

    Logs two parent runs to the repo-root ``mlruns/`` directory:
    - Run A (model=baseline): velocity-only L2-norm, evaluated directly without API
    - Run B (model=current): 6-channel sigmoid, evaluated through the FastAPI test client

    Each parent run contains:
    - params: ``pipeline.*``, ``model.*`` (incl. group overrides), ``alert.*``
    - metrics: precision/recall/f1 at machine and event level, total_alerts
    - dataset input: scenario summary digest for reproducibility
    - N nested child runs, one per scenario, with per-scenario tags and metrics

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
    from sample_processing.model.current.anomaly_model import DEFAULT_PARAMS_PATH, load_alert_params, load_model_params as load_current_model_params
    from sample_processing.model.baseline.anomaly_model import (
        AnomalyModel as BaselineModel,
        load_pipeline_params as load_baseline_pipeline_params,
    )

    resolved_data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    incidents_by_scenario = load_incidents_by_scenario(labels_path)

    # Resolve model cache version for traceability tags and param source.
    from analysis.model_cache import DEFAULT_CACHE_ROOT as _CACHE_ROOT
    _yaml_from_cache: dict | None = None
    _model_cache_notes = ""
    try:
        if str(model_version) == "latest":
            _global_meta = json.loads((_CACHE_ROOT / "meta.json").read_text())
            _resolved_model_version = int(_global_meta.get("latest_version", 0))
        else:
            _resolved_model_version = int(model_version)
        _version_meta = json.loads((_CACHE_ROOT / f"v{_resolved_model_version}" / "meta.json").read_text())
        model_cache_version_tag = f"v{_resolved_model_version}"
        _model_cache_notes = str(_version_meta.get("notes", ""))
        _yaml_from_cache = _version_meta.get("yaml_snapshot")
    except (FileNotFoundError, KeyError, ValueError, OSError):
        model_cache_version_tag = "unknown"

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

    current_model_defaults = load_current_model_params()
    _pipeline_skip = {"model_window_size_hours", "window_overlap_hours"}
    current_model_params_flat = {
        k: v
        for k, v in current_model_defaults.model_dump().items()
        if isinstance(v, (int, float, bool, str)) and k not in _pipeline_skip
    }

    if _yaml_from_cache is not None:
        _raw_yaml = _yaml_from_cache
    else:
        import yaml
        _raw_yaml = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text())
    if isinstance(_raw_yaml, dict) and "groups" in _raw_yaml:
        for _gname, _gvals in _raw_yaml["groups"].items():
            for _gparam, _gval in _gvals.items():
                if isinstance(_gval, (int, float, bool, str)):
                    current_model_params_flat[f"groups.{_gname}.{_gparam}"] = _gval

    pipeline_params: dict[str, Any] = {
        "n_scenarios": len(ordered_ids),
        "window_hours": float(baseline_pipeline.model_window_size_hours),
        "stride_hours": float(
            baseline_pipeline.model_window_size_hours - baseline_pipeline.window_overlap_hours
        ),
    }

    data_version = _data_digest(resolved_data_dir)

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
        model_params={
            "z_threshold": int(baseline_model_params.z_threshold),
            "window_anomaly_ratio": float(baseline_model_params.window_anomaly_ratio),
        },
        pipeline_params=pipeline_params,
        alert_params_dict=None,
        data_digest=data_version,
        model_cache_version="n/a",
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
        model_params=current_model_params_flat,
        pipeline_params=pipeline_params,
        alert_params_dict=alert_params_flat,
        data_digest=data_version,
        model_cache_version=model_cache_version_tag,
        model_cache_notes=_model_cache_notes,
    )

    print("Done. Open 'mlflow ui' from the repo root to view results.")
    return {"baseline_report": baseline_report, "current_report": current_report}
