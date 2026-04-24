"""Feature-column sets, axis/date formatters, bounds, masking, and span helpers."""

from __future__ import annotations

from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_VEL_COLS = ["vel_rms_x", "vel_rms_y", "vel_rms_z"]
_ACCEL_COLS = ["accel_rms_x", "accel_rms_y", "accel_rms_z"]


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _build_mask(
    df: pd.DataFrame,
    *,
    uptime_col: str,
    uptime_only: bool,
    global_mask_col: str | None,
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if uptime_only and uptime_col in df.columns:
        mask &= df[uptime_col].fillna(False).astype(bool)
    if global_mask_col and global_mask_col in df.columns:
        mask &= ~df[global_mask_col].fillna(True).astype(bool)
    return mask


def _date_fmt(ax: plt.Axes, use_index: bool) -> None:
    if use_index:
        return
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _range_with_pad(
    values: list[float] | np.ndarray,
    *,
    floor_pad: float,
    min_span: float,
) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return -floor_pad, floor_pad
    vmin = float(arr.min())
    vmax = float(arr.max())
    span = vmax - vmin
    if span < min_span:
        center = 0.5 * (vmin + vmax)
        half = max(min_span / 2.0, floor_pad)
        return center - half, center + half
    pad = max(0.08 * span, floor_pad)
    return vmin - pad, vmax + pad


def _residual_bounds(values: list[float] | np.ndarray) -> tuple[float, float]:
    return _range_with_pad(values, floor_pad=0.2, min_span=1.0)


def _safe_ts(series: pd.Series) -> list[pd.Timestamp]:
    return pd.to_datetime(series, errors="coerce", utc=True).dropna().tolist()


def _split_index_and_time(df_display: pd.DataFrame, split_col: str, fit_value: str) -> tuple[int | None, pd.Timestamp | None]:
    fit_mask = df_display[split_col].eq(fit_value).to_numpy()
    fit_count = int(fit_mask.sum())
    if fit_count <= 0 or fit_count >= len(df_display):
        return None, None
    split_idx = fit_count - 0.5
    split_time = pd.to_datetime(df_display.iloc[fit_count - 1]["sampled_at"], utc=True)
    return split_idx, split_time


def _incident_spans_for_axis(
    *,
    x_values: list,
    timestamps: list[pd.Timestamp],
    incidents: list[dict[str, Any]],
    use_index: bool,
) -> list[tuple[Any, Any]]:
    if not x_values or not timestamps or not incidents:
        return []
    ts = pd.to_datetime(pd.Series(timestamps), errors="coerce", utc=True)
    spans: list[tuple[Any, Any]] = []
    for inc in incidents:
        start = pd.Timestamp(inc["start"])
        end = pd.Timestamp(inc["end"])
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        mask = (ts >= start) & (ts <= end)
        idxs = np.flatnonzero(mask.to_numpy())
        if idxs.size == 0:
            continue
        lo_i = int(idxs.min())
        hi_i = int(idxs.max())
        if use_index:
            spans.append((x_values[lo_i], x_values[hi_i]))
        else:
            spans.append((timestamps[lo_i], timestamps[hi_i]))
    return spans


def _incident_spans_for_batches(
    *,
    x_values: list,
    window_starts: list[pd.Timestamp],
    window_ends: list[pd.Timestamp],
    incidents: list[dict[str, Any]],
    use_index: bool,
) -> list[tuple[Any, Any]]:
    if not x_values or not window_starts or not window_ends or not incidents:
        return []
    starts = pd.to_datetime(pd.Series(window_starts), errors="coerce", utc=True)
    ends = pd.to_datetime(pd.Series(window_ends), errors="coerce", utc=True)
    spans: list[tuple[Any, Any]] = []
    for inc in incidents:
        start = pd.Timestamp(inc["start"])
        end = pd.Timestamp(inc["end"])
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        overlap = (starts <= end) & (ends >= start)
        idxs = np.flatnonzero(overlap.to_numpy())
        if idxs.size == 0:
            continue
        lo_i = int(idxs.min())
        hi_i = int(idxs.max())
        if use_index:
            spans.append((x_values[lo_i], x_values[hi_i]))
        else:
            spans.append((window_starts[lo_i], window_ends[hi_i]))
    return spans


def _select_by_mask_mode(
    scored_display: pd.DataFrame,
    *,
    mask_col: str,
    mode: str,
) -> pd.Series:
    if mode == "Both" or mask_col not in scored_display.columns:
        return pd.Series(True, index=scored_display.index)
    usable = ~scored_display[mask_col].fillna(True).astype(bool)
    return usable if mode == "ON only" else ~usable


def _series_dict_from_scored(
    scored_display: pd.DataFrame,
    cols: list[str],
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for col in cols:
        d_col = f"d_{col}"
        out[col] = pd.to_numeric(scored_display.get(d_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0).tolist()
    return out


def _scenario_slug(scenario_id: Any) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(scenario_id))
    return slug.strip("_") or "scenario"
