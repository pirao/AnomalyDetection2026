"""Raw-RMS preprocessing: spike clipping + per-channel standard baselines.

Two stages of the pre-scoring data preparation live here:

1. ``clip_rms_spikes`` replaces gross RMS spikes with neighbour means (preprocessing).
2. ``fit_norm_baselines`` / ``apply_norm_scores`` fit and apply per-channel standard
   (z-score) baselines, producing the ``d_{col}`` deviation columns the scorer reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_VEL_FEATURES = ["vel_rms_x", "vel_rms_y", "vel_rms_z"]
_ACCEL_FEATURES = ["accel_rms_x", "accel_rms_y", "accel_rms_z"]


def clip_rms_spikes(
    df: pd.DataFrame,
    vel_threshold: float = 100.0,
    accel_threshold: float = 10.0,
    vel_features: list[str] = _VEL_FEATURES,
    accel_features: list[str] = _ACCEL_FEATURES,
    scenario_col: str = "scenario_id",
    time_col: str = "sampled_at",
) -> pd.DataFrame:
    """Replace gross RMS spikes with the mean of the nearest valid neighbours."""
    all_features = vel_features + accel_features
    n_vel = len(vel_features)

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([scenario_col, time_col]).reset_index(drop=True)

    for _, grp_idx in df.groupby(scenario_col, sort=False).groups.items():
        vals = df.loc[grp_idx, all_features].to_numpy(dtype=float)

        vel_part = vals[:, :n_vel]
        accel_part = vals[:, n_vel:]
        flagged = (vel_part > vel_threshold).any(axis=1) | (
            accel_part > accel_threshold
        ).any(axis=1)

        if not flagged.any():
            continue

        valid_pos = np.where(~flagged)[0]
        result = vals.copy()

        for pos in np.where(flagged)[0]:
            before = valid_pos[valid_pos < pos]
            after = valid_pos[valid_pos > pos]

            if len(before) > 0 and len(after) > 0:
                result[pos] = (vals[before[-1]] + vals[after[0]]) / 2
            elif len(before) > 0:
                result[pos] = vals[before[-1]]
            elif len(after) > 0:
                result[pos] = vals[after[0]]

        df.loc[grp_idx, all_features] = result

    return df


def _fit_1d_standard(values: np.ndarray) -> tuple[float, float] | None:
    """Fit standard mean/std baseline on a 1-D float array."""
    v = values[np.isfinite(values)]
    if len(v) < 2:
        return None

    mean = float(np.mean(v))
    scale = float(np.std(v, ddof=1))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return mean, scale


def fit_norm_baselines(
    df: pd.DataFrame,
    scenario_col: str = "scenario_id",
    vel_cols: list[str] | None = None,
    accel_cols: list[str] | None = None,
    vel_mask_col: str = "global_mask_vel",
    accel_mask_col: str = "global_mask_accel",
    scaler: str = "standard",
) -> dict:
    """Fit per-scenario 1-D baselines on individual RMS channels."""
    if scaler != "standard":
        raise ValueError(f"scaler must be 'standard', got {scaler!r}")

    if vel_cols is None:
        vel_cols = []
    if accel_cols is None:
        accel_cols = []

    models: dict = {}
    feature_groups = [
        *((col, vel_mask_col) for col in vel_cols),
        *((col, accel_mask_col) for col in accel_cols),
    ]

    for scenario_id in sorted(df[scenario_col].dropna().unique()):
        df_sc = df.loc[df[scenario_col] == scenario_id]
        models[scenario_id] = {}

        for col, mask_col in feature_groups:
            valid = ~df_sc[mask_col].astype(bool)
            sub_vals = df_sc.loc[valid, col].values.astype(float)
            result = _fit_1d_standard(sub_vals)
            if result is None:
                models[scenario_id][col] = {}
                continue

            mean, scale = result
            finite = sub_vals[np.isfinite(sub_vals)]
            d_std = (
                float(np.std((finite - mean) / scale, ddof=1))
                if len(finite) > 1
                else 1.0
            )
            models[scenario_id][col] = {"ALL": (mean, scale, max(d_std, 0.01))}

    return models


def apply_norm_scores(
    df: pd.DataFrame,
    baselines: dict,
    scenario_col: str = "scenario_id",
    vel_cols: list[str] | None = None,
    accel_cols: list[str] | None = None,
    vel_mask_col: str = "global_mask_vel",
    accel_mask_col: str = "global_mask_accel",
) -> pd.DataFrame:
    """Add per-channel standard z-score columns ``d_{col}``."""
    if vel_cols is None:
        vel_cols = []
    if accel_cols is None:
        accel_cols = []

    out = df.copy()
    all_cols = vel_cols + accel_cols
    mask_cols = [vel_mask_col] * len(vel_cols) + [accel_mask_col] * len(accel_cols)

    for col in all_cols:
        out[f"d_{col}"] = np.nan

    for scenario_id, scenario_baselines in baselines.items():
        sc_mask = out[scenario_col] == scenario_id

        for col, mask_col in zip(all_cols, mask_cols):
            d_col = f"d_{col}"
            col_baselines = scenario_baselines.get(col, {})
            if "ALL" not in col_baselines:
                continue

            valid = sc_mask & ~out[mask_col].astype(bool)
            values = out.loc[valid, col].values.astype(float)
            if len(values) == 0:
                continue

            mean, scale = col_baselines["ALL"][:2]
            out.loc[valid, d_col] = (values - mean) / scale

    return out
