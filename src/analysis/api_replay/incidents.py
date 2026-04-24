"""Incident-label loading, span extraction, and alert-window matching.

Shared by ``simulation`` (to emit ``incident_spans`` alongside replay rows),
by ``evaluation`` (to score alerts against labeled windows), and by the
scoring widgets (to shade incident regions on plots).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS_PATH = _REPO_ROOT / "labels" / "incidents.yaml"


def assign_incident_label(
    ts: pd.Timestamp,
    windows: Iterable[tuple[pd.Timestamp, pd.Timestamp, int]],
    *,
    normal_label: str = "normal",
    incident_prefix: str = "incident_",
) -> str:
    """Return the incident label for ``ts`` or ``normal_label`` if no window matches."""
    for start, end, idx in windows:
        if start <= ts <= end:
            return f"{incident_prefix}{idx}"
    return str(normal_label)


def get_incident_spans(
    df: pd.DataFrame,
    *,
    time_col: str = "sampled_at",
    label_col: str = "label",
    normal_label: str = "normal",
) -> list[dict[str, Any]]:
    """Return contiguous non-normal labeled spans for plotting.

    The output format is a list of dicts:
    ``[{"label": ..., "start": ..., "end": ...}, ...]``.
    """
    if df.empty or label_col not in df.columns or time_col not in df.columns:
        return []

    sub = df[[time_col, label_col]].copy()
    sub[time_col] = pd.to_datetime(sub[time_col], errors="coerce")
    sub = sub.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if sub.empty:
        return []

    is_incident = sub[label_col].astype(str) != str(normal_label)
    if not bool(is_incident.any()):
        return []

    group_id = (is_incident.ne(is_incident.shift(fill_value=False)) | sub[label_col].ne(sub[label_col].shift())).cumsum()

    spans: list[dict[str, Any]] = []
    for _, grp in sub.groupby(group_id, sort=False):
        label = grp[label_col].iloc[0]
        if str(label) == str(normal_label):
            continue
        spans.append(
            {
                "label": label,
                "start": grp[time_col].iloc[0],
                "end": grp[time_col].iloc[-1],
            }
        )
    return spans


def _normalize_incidents(incidents: Any) -> list[dict[str, pd.Timestamp]]:
    """Normalize incidents into ``[{start, end}, ...]`` with valid timestamps."""
    out: list[dict[str, pd.Timestamp]] = []
    if incidents is None:
        return out

    if isinstance(incidents, pd.DataFrame):
        candidates = incidents.to_dict("records")
    else:
        candidates = list(incidents)

    for item in candidates:
        if not isinstance(item, dict):
            continue
        start = pd.to_datetime(item.get("start"), errors="coerce")
        end = pd.to_datetime(item.get("end"), errors="coerce")
        if pd.isna(start) or pd.isna(end) or start >= end:
            continue
        out.append({"start": start, "end": end})
    return out


def load_incidents_by_scenario(
    labels_path: Path | str | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Load incident labels into ``{scenario_id: [incident, ...]}``."""
    import yaml

    path = Path(labels_path) if labels_path is not None else DEFAULT_LABELS_PATH
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {int(k): (v or []) for k, v in raw.items()}


def _serialize_incident_window(inc: dict[str, pd.Timestamp]) -> dict[str, str]:
    return {
        "start": pd.Timestamp(inc["start"]).isoformat(),
        "end": pd.Timestamp(inc["end"]).isoformat(),
    }


def _alert_hits_incident_window(
    alert_ts: pd.Timestamp,
    incident: dict[str, pd.Timestamp],
    *,
    tolerance: pd.Timedelta,
) -> bool:
    return (incident["start"] - tolerance) <= alert_ts <= (incident["end"] + tolerance)
