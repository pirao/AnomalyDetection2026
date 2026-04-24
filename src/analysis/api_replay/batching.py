"""Batch iteration primitives for offline API replay.

These helpers convert a scenario DataFrame into the exact payload shape the
production API consumes and iterate over it in the two batching modes the
service supports:

- **time batches**  — fixed-duration half-open windows anchored at the first
  prediction timestamp with ``stride = window - overlap`` (matches the live
  ``/predict`` rule and ``src/tests/test_evaluation.py``).
- **row batches**   — fixed-row slices used by ``generate_report.py`` for
  quick local replays over a compact number of predictions.
"""

from __future__ import annotations

import pandas as pd

from sample_processing.model.interface import DataPoint, TimeSeries


def df_to_timeseries(
    df: pd.DataFrame,
    *,
    time_col: str = "sampled_at",
) -> TimeSeries:
    """Convert a DataFrame batch into the API ``TimeSeries`` payload shape."""
    rows = (
        df.sort_values(time_col)
        .reset_index(drop=True)[
            [
                time_col,
                "uptime",
                "vel_rms_x",
                "vel_rms_y",
                "vel_rms_z",
                "accel_rms_x",
                "accel_rms_y",
                "accel_rms_z",
            ]
        ]
    )
    return TimeSeries(
        data=[
            DataPoint(
                timestamp=getattr(row, time_col),
                uptime=bool(row.uptime),
                vel_x=float(getattr(row, "vel_rms_x")),
                vel_y=float(getattr(row, "vel_rms_y")),
                vel_z=float(getattr(row, "vel_rms_z")),
                acc_x=float(getattr(row, "accel_rms_x")),
                acc_y=float(getattr(row, "accel_rms_y")),
                acc_z=float(getattr(row, "accel_rms_z")),
            )
            for row in rows.itertuples(index=False)
        ]
    )


def iter_time_batches(
    pred_df: pd.DataFrame,
    *,
    window_hours: float,
    overlap_hours: float,
    time_col: str = "sampled_at",
):
    """Yield fixed-time batches using the evaluation/API half-open rule.

    Windows are anchored at the first valid timestamp and use
    ``[current, current + window)`` with stride ``window - overlap``.
    """
    if pred_df.empty:
        return

    pred = pred_df.copy()
    pred[time_col] = pd.to_datetime(pred[time_col], errors="coerce")
    pred = pred.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if pred.empty:
        return

    window = pd.Timedelta(hours=float(window_hours))
    stride = pd.Timedelta(hours=float(window_hours - overlap_hours))
    if stride <= pd.Timedelta(0):
        raise ValueError("window_hours - overlap_hours must be positive")

    current = pred[time_col].iloc[0]
    end_ts = pred[time_col].iloc[-1]
    batch_index = 0

    while current <= end_ts:
        window_end = current + window
        mask = (pred[time_col] >= current) & (pred[time_col] < window_end)
        batch_df = pred.loc[mask].copy()
        if not batch_df.empty:
            yield batch_index, current, window_end, batch_df
        current += stride
        batch_index += 1


def _infer_cadence_timedelta(
    timestamps: pd.Series,
    *,
    default_minutes: float = 10.0,
) -> pd.Timedelta:
    """Infer the dominant cadence from timestamps for display spans."""
    ts = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(ts) < 2:
        return pd.Timedelta(minutes=default_minutes)

    deltas = ts.diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return pd.Timedelta(minutes=default_minutes)

    mode = deltas.mode()
    cadence = mode.iloc[0] if not mode.empty else deltas.median()
    if cadence <= pd.Timedelta(0):
        cadence = pd.Timedelta(minutes=default_minutes)
    return cadence


def iter_row_batches(
    pred_df: pd.DataFrame,
    *,
    batch_size: int = 50,
    time_col: str = "sampled_at",
):
    """Yield fixed-row batches like ``generate_report.py``.

    Each yielded batch corresponds to a single submitted predict payload over a
    contiguous row slice. The plotting span uses the first timestamp as the
    start and the last timestamp plus one cadence step as the end.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if pred_df.empty:
        return

    pred = pred_df.copy()
    pred[time_col] = pd.to_datetime(pred[time_col], errors="coerce")
    pred = pred.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if pred.empty:
        return

    for batch_index, start_idx in enumerate(range(0, len(pred), batch_size)):
        batch_df = pred.iloc[start_idx : start_idx + batch_size].copy()
        if batch_df.empty:
            continue
        cadence = _infer_cadence_timedelta(batch_df[time_col])
        window_start = batch_df[time_col].iloc[0]
        window_end = batch_df[time_col].iloc[-1] + cadence
        yield batch_index, window_start, window_end, batch_df
