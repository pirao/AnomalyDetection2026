"""Shared scoring helpers re-exported from scoring internals.

This module keeps common scoring constants, layout helpers, and the default
widget export path in one place so notebook-facing scoring modules can share
the same imports without duplicating plumbing.
"""

from pathlib import Path

from ._colors import (
    _ACCEL_AXIS_COLORS,
    _BOTH_COLOR,
    _CURR_FILL_ACCEL,
    _CURR_FILL_VEL,
    _D_ACCEL_COLOR,
    _D_VEL_COLOR,
    _INCIDENT_ALPHA,
    _INCIDENT_COLOR,
    _REF_EDGE,
    _REF_FILL,
    _SIGMOID_COLOR,
    _SPLIT_COLOR,
    _THRESH_COLOR,
    _VEL_AXIS_COLORS,
)
from ._layout import (
    _ACCEL_COLS,
    _VEL_COLS,
    _build_mask,
    _date_fmt,
    _incident_spans_for_axis,
    _incident_spans_for_batches,
    _range_with_pad,
    _residual_bounds,
    _safe_series,
    _safe_ts,
    _scenario_slug,
    _select_by_mask_mode,
    _series_dict_from_scored,
    _split_index_and_time,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WIDGET_EXPORT_DIR = _REPO_ROOT / "notebooks" / "_generated" / "widget_exports"

__all__ = [
    "DEFAULT_WIDGET_EXPORT_DIR",
    "_ACCEL_AXIS_COLORS",
    "_ACCEL_COLS",
    "_BOTH_COLOR",
    "_CURR_FILL_ACCEL",
    "_CURR_FILL_VEL",
    "_D_ACCEL_COLOR",
    "_D_VEL_COLOR",
    "_INCIDENT_ALPHA",
    "_INCIDENT_COLOR",
    "_REF_EDGE",
    "_REF_FILL",
    "_SIGMOID_COLOR",
    "_SPLIT_COLOR",
    "_THRESH_COLOR",
    "_VEL_AXIS_COLORS",
    "_VEL_COLS",
    "_build_mask",
    "_date_fmt",
    "_incident_spans_for_axis",
    "_incident_spans_for_batches",
    "_range_with_pad",
    "_residual_bounds",
    "_safe_series",
    "_safe_ts",
    "_scenario_slug",
    "_select_by_mask_mode",
    "_series_dict_from_scored",
    "_split_index_and_time",
]
