"""Group-candidate construction: velocity triplet, accel triplet, mixed, group-6."""

from __future__ import annotations

from ._utils import l2_norm


VEL_GROUP = frozenset({"vel_rms_x", "vel_rms_y", "vel_rms_z"})
ACCEL_GROUP = frozenset({"accel_rms_x", "accel_rms_y", "accel_rms_z"})


def build_group_candidate(
    active_channels: list[str],
    current_max_by_channel: dict[str, float],
    *,
    enable_group6: bool,
) -> dict[str, object] | None:
    active_set = set(active_channels)
    vel_candidate: dict[str, object] | None = None
    accel_candidate: dict[str, object] | None = None

    vel_vals: list[float] = []
    accel_vals: list[float] = []

    if VEL_GROUP.issubset(active_set):
        vel_vals = [float(current_max_by_channel.get(ch, 0.0)) for ch in sorted(VEL_GROUP)]
        vel_candidate = {
            "priority": 1,
            "mode_type": "velocity_cluster_degradation",
            "mode_kind": "group-3",
            "group_family": "canonical-velocity",
            "metric_label": "residual norm",
            "group_channels": sorted(VEL_GROUP),
            "active_channels": sorted(VEL_GROUP),
            "severity": l2_norm(vel_vals),
        }
    if ACCEL_GROUP.issubset(active_set):
        accel_vals = [float(current_max_by_channel.get(ch, 0.0)) for ch in sorted(ACCEL_GROUP)]
        accel_candidate = {
            "priority": 1,
            "mode_type": "acceleration_cluster_degradation",
            "mode_kind": "group-3",
            "group_family": "canonical-acceleration",
            "metric_label": "residual norm",
            "group_channels": sorted(ACCEL_GROUP),
            "active_channels": sorted(ACCEL_GROUP),
            "severity": l2_norm(accel_vals),
        }

    if enable_group6 and vel_candidate is not None and accel_candidate is not None:
        return {
            "priority": 2,
            "mode_type": "full_degradation",
            "mode_kind": "group-6",
            "group_family": "group-6",
            "metric_label": "l2 norm",
            "group_channels": sorted(VEL_GROUP | ACCEL_GROUP),
            "active_channels": list(active_channels),
            "severity": l2_norm(vel_vals + accel_vals),
        }

    if vel_candidate is not None and accel_candidate is not None:
        return (
            vel_candidate
            if float(vel_candidate["severity"]) >= float(accel_candidate["severity"])
            else accel_candidate
        )
    if vel_candidate is not None or accel_candidate is not None:
        return vel_candidate or accel_candidate

    if 3 <= len(active_set) <= 5:
        mixed_channels = sorted(active_set)
        mixed_vals = [float(current_max_by_channel.get(ch, 0.0)) for ch in mixed_channels]
        return {
            "priority": 1,
            "mode_type": "mixed_cluster_degradation",
            "mode_kind": "group-3",
            "group_family": "mixed",
            "metric_label": "residual norm",
            "group_channels": mixed_channels,
            "active_channels": mixed_channels,
            "severity": l2_norm(mixed_vals),
        }
    return None
