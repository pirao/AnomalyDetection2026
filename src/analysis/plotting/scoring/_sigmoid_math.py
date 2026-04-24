"""Sigmoid math, cache keys, and the two-level scoring-payload builder.

The payload builder is the engine behind ``create_sigmoid_scoring_widget`` —
it pre-scores each batch once and reuses that work whenever sigmoid sliders
(alpha, beta, threshold, top_k, fusion) move. Two module-level caches keep
the widget responsive:

- ``_fit_cache`` — keyed on (scenario_id, vel_col, accel_col); stores fitted
  ``SensorModel`` instances so that swapping only sigmoid params does not
  trigger a refit.
- ``_score_base_cache`` — keyed on ``_ScoreBaseKey``; stores pre-scored
  batches. Alpha/beta/threshold are NOT part of this key, so slider moves
  reuse this cache and skip the expensive ``_score_df()`` call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from analysis.api_replay import df_to_timeseries, get_incident_spans, iter_time_batches
from sample_processing.model.anomaly_model import load_pipeline_params
from sample_processing.model.baselines import apply_norm_scores, fit_norm_baselines
from sample_processing.model.scenario_groups import (
    get_scenario_group_key,
    get_scenario_group_label,
)
from sample_processing.model.sensor_model import SensorModel

from ._helpers import (
    _ACCEL_COLS,
    _VEL_COLS,
    _safe_ts,
    _select_by_mask_mode,
    _series_dict_from_scored,
    _split_index_and_time,
)


# Fit-layer cache: keyed on (scenario_id, vel_col, accel_col) — the only
# inputs that require a model re-fit. All post-fit params (alpha, beta,
# threshold, top_k, fusion) run off the cached SensorModel.
_fit_cache: dict[tuple, SensorModel] = {}

# Score-base cache: keyed on (_ScoreBaseKey) — stores scored display data
# and pre-scored batch tuples. Sigmoid params are NOT part of this key, so
# alpha/beta/threshold slider moves reuse this cache and skip _score_df().
_score_base_cache: dict[Any, dict[str, Any]] = {}


def _sigmoid(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = np.clip(alpha * (x - beta), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _solve_alpha_beta_from_two_anchors(
    *,
    residual_1: float,
    score_1: float,
    residual_2: float,
    score_2: float,
    fallback_alpha: float,
    fallback_beta: float,
) -> tuple[float, float]:
    r1 = float(residual_1)
    r2 = float(residual_2)
    p1 = float(np.clip(score_1, 1e-4, 1 - 1e-4))
    p2 = float(np.clip(score_2, 1e-4, 1 - 1e-4))

    if abs(r2 - r1) < 1e-9:
        return float(fallback_alpha), float(fallback_beta)
    logit1 = math.log(p1 / (1.0 - p1))
    logit2 = math.log(p2 / (1.0 - p2))
    if abs(logit2 - logit1) < 1e-9:
        return float(fallback_alpha), float(fallback_beta)

    alpha = (logit2 - logit1) / (r2 - r1)
    if abs(alpha) < 1e-9:
        return float(fallback_alpha), float(fallback_beta)
    beta = r1 - (logit1 / alpha)
    return float(alpha), float(beta)


def _recompute_dnorm_frame(
    df: pd.DataFrame,
    *,
    scenario_col: str,
    split_col: str,
    uptime_col: str,
    vel_cols: list[str],
    accel_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()
    out["global_mask_vel"] = ~out[uptime_col].fillna(False).astype(bool)
    out["global_mask_accel"] = ~out[uptime_col].fillna(False).astype(bool)

    fit_df = out[out[split_col] == "fit"].copy() if split_col in out.columns else out.copy()
    baselines = fit_norm_baselines(
        df=fit_df,
        scenario_col=scenario_col,
        vel_cols=vel_cols,
        accel_cols=accel_cols,
        vel_mask_col="global_mask_vel",
        accel_mask_col="global_mask_accel",
        scaler="standard",
    )
    return apply_norm_scores(
        df=out,
        baselines=baselines,
        scenario_col=scenario_col,
        vel_cols=vel_cols,
        accel_cols=accel_cols,
        vel_mask_col="global_mask_vel",
        accel_mask_col="global_mask_accel",
    )


@dataclass(frozen=True)
class _FastPayloadKey:
    scenario_id: Any
    show: str
    uptime_only: bool
    cyclic_only: bool
    on_mask_mode: str
    is_cyclic: bool
    vel_col: str
    accel_col: str
    alpha: float
    beta: float
    threshold_vel: float
    threshold_accel: float
    window_top_k: int
    fusion_thr: float


@dataclass(frozen=True)
class _ScoreBaseKey:
    """Cache key for pre-scored data — no sigmoid params."""
    scenario_id: Any
    vel_col: str
    accel_col: str
    show: str
    uptime_only: bool
    cyclic_only: bool
    on_mask_mode: str
    is_cyclic: bool


def _build_score_base(
    *,
    full_df: pd.DataFrame,
    scenario_id: Any,
    vel_col: str,
    accel_col: str,
    show: str,
    uptime_only: bool,
    cyclic_only: bool,
    on_mask_mode: str,
    scenario_col: str,
    split_col: str,
    fit_value: str,
    pred_value: str,
    time_col: str,
    label_col: str,
    normal_label: str,
    uptime_col: str,
    cyclic_col: str | None,
) -> dict[str, Any]:
    """Expensive base: filter data, fit/score model, pre-score batches.

    Results are cached by _ScoreBaseKey — sigmoid params (alpha, beta,
    threshold, top_k, fusion) are not part of this computation.
    """
    is_cyclic = False

    df_sid = full_df[full_df[scenario_col] == scenario_id].copy()
    df_sid[time_col] = pd.to_datetime(df_sid[time_col], errors="coerce", utc=True)
    df_sid = df_sid.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if df_sid.empty:
        raise ValueError(f"Scenario {scenario_id} is not present in the dataframe.")

    fit_df_orig = df_sid[df_sid[split_col] == fit_value].copy()
    if fit_df_orig.empty:
        raise ValueError(f"Scenario {scenario_id} is missing the fit split.")

    df_display = df_sid.copy()
    if uptime_only and uptime_col in df_display.columns:
        df_display = df_display[df_display[uptime_col].fillna(False)]
    if cyclic_only and is_cyclic and cyclic_col and cyclic_col in df_display.columns:
        df_display = df_display[df_display[cyclic_col].fillna(False)]
    fit_df_display = df_sid[df_sid[split_col] == fit_value].copy()
    pred_df_display = df_sid[df_sid[split_col] == pred_value].copy()
    if uptime_only and uptime_col in fit_df_display.columns:
        fit_df_display = fit_df_display[fit_df_display[uptime_col].fillna(False)]
        pred_df_display = pred_df_display[pred_df_display[uptime_col].fillna(False)]
    if cyclic_only and is_cyclic and cyclic_col and cyclic_col in fit_df_display.columns:
        fit_df_display = fit_df_display[fit_df_display[cyclic_col].fillna(False)]
        pred_df_display = pred_df_display[pred_df_display[cyclic_col].fillna(False)]

    _fit_key = (scenario_id, vel_col, accel_col)
    sensor = _fit_cache.get(_fit_key)
    if sensor is None:
        sensor = SensorModel(is_cyclic=is_cyclic, baseline_scaler="standard")
        sensor.fit(df_to_timeseries(fit_df_orig, time_col=time_col))
        _fit_cache[_fit_key] = sensor

    scored_display = sensor._score_df(df_to_timeseries(df_display, time_col=time_col)).copy()
    scored_display[time_col] = pd.to_datetime(scored_display[time_col], errors="coerce", utc=True)
    scored_display = scored_display.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    raw_meta_cols = [c for c in [time_col, split_col, label_col, uptime_col, cyclic_col] if c and c in df_display.columns]
    merge_meta = df_display[raw_meta_cols].copy().sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
    scored_display = scored_display.merge(merge_meta, on=time_col, how="left", suffixes=("", "_meta"))

    if show == "fit":
        scored_display = scored_display[scored_display[split_col] == fit_value].copy()
        df_display = df_display[df_display[split_col] == fit_value].copy()
    elif show == "pred":
        scored_display = scored_display[scored_display[split_col] == pred_value].copy()
        df_display = df_display[df_display[split_col] == pred_value].copy()

    if df_display.empty or scored_display.empty:
        raise ValueError(f"Scenario {scenario_id} has no rows for show={show!r} after filtering.")

    d_vel_col = f"d_{vel_col}"
    d_accel_col = f"d_{accel_col}"

    vel_d_by_col = _series_dict_from_scored(scored_display, _VEL_COLS)
    accel_d_by_col = _series_dict_from_scored(scored_display, _ACCEL_COLS)
    vel_d = pd.to_numeric(scored_display.get(d_vel_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy()
    accel_d = pd.to_numeric(scored_display.get(d_accel_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy()

    vel_display_mask = _select_by_mask_mode(scored_display, mask_col="global_mask_vel", mode=on_mask_mode)
    accel_display_mask = _select_by_mask_mode(scored_display, mask_col="global_mask_accel", mode=on_mask_mode)

    pipeline_params = load_pipeline_params()

    pre_scored_batches: list[dict[str, Any]] = []

    def _prescore_split(split_name: str, split_df_local: pd.DataFrame) -> None:
        if split_df_local.empty:
            return
        for _, window_start, window_end, batch_df in iter_time_batches(
            split_df_local,
            window_hours=pipeline_params.model_window_size_hours,
            overlap_hours=pipeline_params.window_overlap_hours,
            time_col=time_col,
        ):
            batch_ts = df_to_timeseries(batch_df, time_col=time_col)
            _, up_vel, up_accel, expected_samples = sensor._score_batch(
                batch_ts,
                model_window_size_hours=pipeline_params.model_window_size_hours,
                expected_samples_per_window=None,
            )
            pre_scored_batches.append({
                "split": split_name,
                "window_start": window_start,
                "window_end": window_end,
                "window_mid": window_start + (window_end - window_start) / 2,
                "up_vel": up_vel,
                "up_accel": up_accel,
                "expected_samples": expected_samples,
                "timestamp": batch_ts.data[-1].timestamp,
            })

    if show == "fit":
        _prescore_split("fit", fit_df_display)
    elif show == "pred":
        _prescore_split("pred", pred_df_display)
    else:
        _prescore_split("fit", fit_df_display)
        _prescore_split("pred", pred_df_display)

    incidents = (
        get_incident_spans(df_display, label_col=label_col, time_col=time_col, normal_label=normal_label)
        if label_col in df_display.columns
        else []
    )
    df_display_reset = df_display.reset_index(drop=True)
    split_idx, split_time = _split_index_and_time(df_display_reset, split_col, fit_value)

    return {
        "scenario_id": scenario_id,
        "is_cyclic": is_cyclic,
        "show": show,
        "sensor": sensor,
        "df_sid": df_sid,
        "df_display": df_display_reset,
        "scored_display": scored_display.reset_index(drop=True),
        "fit_df_display": fit_df_display.reset_index(drop=True),
        "pred_df_display": pred_df_display.reset_index(drop=True),
        "incidents": incidents,
        "split_idx": split_idx,
        "split_time": split_time,
        "time_values": _safe_ts(df_display[time_col]),
        "vel_col": vel_col,
        "accel_col": accel_col,
        "d_vel_col": d_vel_col,
        "d_accel_col": d_accel_col,
        "vel_d": vel_d,
        "accel_d": accel_d,
        "vel_d_by_col": vel_d_by_col,
        "accel_d_by_col": accel_d_by_col,
        "vel_display_mask": vel_display_mask.fillna(False).astype(bool).tolist(),
        "accel_display_mask": accel_display_mask.fillna(False).astype(bool).tolist(),
        "raw_vel_values": pd.to_numeric(df_display.get(vel_col, pd.Series(dtype=float)), errors="coerce").tolist(),
        "raw_accel_values": pd.to_numeric(df_display.get(accel_col, pd.Series(dtype=float)), errors="coerce").tolist(),
        "proc_vel_values": pd.to_numeric(scored_display.get(vel_col, pd.Series(dtype=float)), errors="coerce").tolist(),
        "proc_accel_values": pd.to_numeric(scored_display.get(accel_col, pd.Series(dtype=float)), errors="coerce").tolist(),
        "pre_scored_batches": pre_scored_batches,
        "pipeline_params": pipeline_params,
        "split_col": split_col,
        "fit_value": fit_value,
    }


def _build_payload_from_base(
    base: dict[str, Any],
    *,
    alpha_vel: float,
    beta_vel: float,
    threshold_vel: float,
    alpha_accel: float,
    beta_accel: float,
    threshold_accel: float,
    window_top_k: int,
    fusion_thr: float,
) -> dict[str, Any]:
    """Fast path: apply sigmoid params to pre-scored base data.

    Only runs sigmoid math and _build_batch_details_from_scored() per batch —
    skips the expensive _score_df() / _score_batch() calls entirely.
    """
    sensor = base["sensor"]
    vel_d = base["vel_d"]
    accel_d = base["accel_d"]
    vel_d_by_col = base["vel_d_by_col"]
    accel_d_by_col = base["accel_d_by_col"]
    split_col = base["split_col"]
    fit_value = base["fit_value"]

    vel_residual = vel_d - float(threshold_vel)
    accel_residual = accel_d - float(threshold_accel)
    vel_score = _sigmoid(vel_residual, alpha=alpha_vel, beta=beta_vel)
    accel_score = _sigmoid(accel_residual, alpha=alpha_accel, beta=beta_accel)
    vel_residual_by_col = {
        col: (np.asarray(vals, dtype=float) - float(threshold_vel)).tolist()
        for col, vals in vel_d_by_col.items()
    }
    accel_residual_by_col = {
        col: (np.asarray(vals, dtype=float) - float(threshold_accel)).tolist()
        for col, vals in accel_d_by_col.items()
    }
    vel_score_by_col = {
        col: _sigmoid(np.asarray(vals, dtype=float), alpha=alpha_vel, beta=beta_vel).tolist()
        for col, vals in vel_residual_by_col.items()
    }
    accel_score_by_col = {
        col: _sigmoid(np.asarray(vals, dtype=float), alpha=alpha_accel, beta=beta_accel).tolist()
        for col, vals in accel_residual_by_col.items()
    }

    batch_rows: list[dict[str, Any]] = []
    for b in base["pre_scored_batches"]:
        details = sensor._build_batch_details_from_scored(
            scored=pd.DataFrame(),
            up_vel=b["up_vel"],
            up_accel=b["up_accel"],
            timestamp=b["timestamp"],
            alpha_vel=alpha_vel,
            alpha_accel=alpha_accel,
            beta_vel=beta_vel,
            beta_accel=beta_accel,
            threshold_vel=threshold_vel,
            threshold_accel=threshold_accel,
            window_top_k=window_top_k,
            fusion_threshold=fusion_thr,
            expected_samples_per_window=b["expected_samples"],
        )
        batch_rows.append({
            "split": b["split"],
            "window_start": b["window_start"],
            "window_end": b["window_end"],
            "window_mid": b["window_mid"],
            **details,
        })

    batch_df_plot = (
        pd.DataFrame(batch_rows).sort_values("window_mid").reset_index(drop=True)
        if batch_rows else pd.DataFrame()
    )
    if not batch_df_plot.empty:
        batch_df_plot["batch_index"] = np.arange(len(batch_df_plot), dtype=int)

    def _batch_metric_frame(detail_col: str, metric_key: str, cols: list[str]) -> dict[str, list[float]]:
        metrics: dict[str, list[float]] = {col: [] for col in cols}
        if batch_df_plot.empty or detail_col not in batch_df_plot.columns:
            return metrics
        for detail_map in batch_df_plot[detail_col]:
            detail_map = detail_map if isinstance(detail_map, dict) else {}
            for col in cols:
                info = detail_map.get(col, {}) if isinstance(detail_map, dict) else {}
                metrics[col].append(float(info.get(metric_key, 0.0)))
        return metrics

    vel_raw_window_score_by_col = _batch_metric_frame("vel_channel_details", "occupancy_raw", _VEL_COLS)
    accel_raw_window_score_by_col = _batch_metric_frame("accel_channel_details", "occupancy_raw", _ACCEL_COLS)
    vel_occupancy_by_col = _batch_metric_frame("vel_channel_details", "occupancy_observed", _VEL_COLS)
    accel_occupancy_by_col = _batch_metric_frame("accel_channel_details", "occupancy_observed", _ACCEL_COLS)

    batch_split_idx = None
    batch_split_time = None
    if not batch_df_plot.empty and split_col in batch_df_plot.columns:
        fit_batch_mask = batch_df_plot[split_col].eq(fit_value).to_numpy()
        fit_batch_count = int(fit_batch_mask.sum())
        if 0 < fit_batch_count < len(batch_df_plot):
            batch_split_idx = fit_batch_count - 0.5
            batch_split_time = pd.to_datetime(batch_df_plot.iloc[fit_batch_count - 1]["window_mid"], utc=True)

    scenario_id = base["scenario_id"]
    return {
        "scenario_id": scenario_id,
        "scenario_group": get_scenario_group_key(scenario_id),
        "scenario_group_label": get_scenario_group_label(scenario_id),
        "is_cyclic": base["is_cyclic"],
        "show": base["show"],
        "sensor": sensor,
        "df_sid": base["df_sid"],
        "df_display": base["df_display"],
        "scored_display": base["scored_display"],
        "fit_df_display": base["fit_df_display"],
        "pred_df_display": base["pred_df_display"],
        "batch_df_plot": batch_df_plot,
        "incidents": base["incidents"],
        "split_idx": base["split_idx"],
        "split_time": base["split_time"],
        "time_values": base["time_values"],
        "batch_index_x": batch_df_plot["batch_index"].tolist() if not batch_df_plot.empty else [],
        "batch_split_idx": batch_split_idx,
        "batch_split_time": batch_split_time,
        "raw_vel_col": base["vel_col"],
        "raw_accel_col": base["accel_col"],
        "raw_vel_values": base["raw_vel_values"],
        "raw_accel_values": base["raw_accel_values"],
        "proc_vel_values": base["proc_vel_values"],
        "proc_accel_values": base["proc_accel_values"],
        "vel_d": vel_d.tolist(),
        "accel_d": accel_d.tolist(),
        "vel_d_by_col": vel_d_by_col,
        "accel_d_by_col": accel_d_by_col,
        "vel_residual": vel_residual.tolist(),
        "accel_residual": accel_residual.tolist(),
        "vel_residual_by_col": vel_residual_by_col,
        "accel_residual_by_col": accel_residual_by_col,
        "vel_score": vel_score.tolist(),
        "accel_score": accel_score.tolist(),
        "vel_score_by_col": vel_score_by_col,
        "accel_score_by_col": accel_score_by_col,
        "vel_raw_window_score_by_col": vel_raw_window_score_by_col,
        "accel_raw_window_score_by_col": accel_raw_window_score_by_col,
        "vel_occupancy_by_col": vel_occupancy_by_col,
        "accel_occupancy_by_col": accel_occupancy_by_col,
        "alpha_vel": alpha_vel,
        "beta_vel": beta_vel,
        "threshold_vel": threshold_vel,
        "alpha_accel": alpha_accel,
        "beta_accel": beta_accel,
        "threshold_accel": threshold_accel,
        "window_top_k": int(window_top_k),
        "fusion_thr": fusion_thr,
        "vel_col": base["vel_col"],
        "accel_col": base["accel_col"],
        "vel_display_mask": base["vel_display_mask"],
        "accel_display_mask": base["accel_display_mask"],
    }


def build_sigmoid_scoring_payload_fast(
    *,
    full_df: pd.DataFrame,
    scenario_id: Any,
    vel_col: str,
    accel_col: str,
    show: str,
    uptime_only: bool,
    cyclic_only: bool,
    on_mask_mode: str,
    alpha_vel: float,
    beta_vel: float,
    threshold_vel: float,
    alpha_accel: float,
    beta_accel: float,
    threshold_accel: float,
    window_top_k: int,
    fusion_thr: float,
    scenario_col: str = "scenario_id",
    split_col: str = "split",
    fit_value: str = "fit",
    pred_value: str = "pred",
    time_col: str = "sampled_at",
    label_col: str = "label",
    normal_label: str = "normal",
    uptime_col: str = "uptime",
    cyclic_col: str | None = None,
) -> dict[str, Any]:
    """Prepare the full plotting payload for the scoring widget.

    Uses a two-level cache:
    - Level 1 (_score_base_cache): scored_display + pre-scored batch tuples,
      keyed by scenario/column/filter params (no sigmoid params).
    - Level 2 (payload_cache in widget): full payload keyed by all params.

    Alpha/beta/threshold slider moves hit only level 2 miss → level 1 hit,
    skipping _score_df() and all per-batch _score_batch() calls entirely.
    """
    is_cyclic = False
    base_key = _ScoreBaseKey(
        scenario_id=scenario_id,
        vel_col=vel_col,
        accel_col=accel_col,
        show=show,
        uptime_only=uptime_only,
        cyclic_only=cyclic_only,
        on_mask_mode=on_mask_mode,
        is_cyclic=is_cyclic,
    )
    base = _score_base_cache.get(base_key)
    if base is None:
        base = _build_score_base(
            full_df=full_df,
            scenario_id=scenario_id,
            vel_col=vel_col,
            accel_col=accel_col,
            show=show,
            uptime_only=uptime_only,
            cyclic_only=cyclic_only,
            on_mask_mode=on_mask_mode,
            scenario_col=scenario_col,
            split_col=split_col,
            fit_value=fit_value,
            pred_value=pred_value,
            time_col=time_col,
            label_col=label_col,
            normal_label=normal_label,
            uptime_col=uptime_col,
            cyclic_col=cyclic_col,
        )
        _score_base_cache[base_key] = base
    return _build_payload_from_base(
        base,
        alpha_vel=alpha_vel,
        beta_vel=beta_vel,
        threshold_vel=threshold_vel,
        alpha_accel=alpha_accel,
        beta_accel=beta_accel,
        threshold_accel=threshold_accel,
        window_top_k=window_top_k,
        fusion_thr=fusion_thr,
    )


def clear_sigmoid_scoring_caches() -> None:
    """Clear the scoring widget's module-level fit and payload caches."""
    _score_base_cache.clear()
    _fit_cache.clear()
