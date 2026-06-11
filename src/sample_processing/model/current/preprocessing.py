"""Runtime-safe raw RMS preprocessing helpers."""

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
