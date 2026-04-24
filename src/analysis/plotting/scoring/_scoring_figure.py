"""Figure renderer for ``create_sigmoid_scoring_widget``.

Builds the 9-row × 2-column diagnostics figure that shows every stage of
the scoring pipeline — raw series, processed series, normalized distances,
residuals, per-channel sigmoid scores, occupancy scores, modality fusion,
and active-modality bands. Consumed exclusively by ``scoring_widget`` and
by its export path (export_all_defaults).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from sample_processing.model.anomaly_model import load_alert_params

from ._helpers import (
    _ACCEL_AXIS_COLORS,
    _ACCEL_COLS,
    _BOTH_COLOR,
    _D_ACCEL_COLOR,
    _D_VEL_COLOR,
    _INCIDENT_ALPHA,
    _INCIDENT_COLOR,
    _SPLIT_COLOR,
    _THRESH_COLOR,
    _VEL_AXIS_COLORS,
    _VEL_COLS,
    _date_fmt,
    _incident_spans_for_axis,
    _incident_spans_for_batches,
    _residual_bounds,
)


def _fast_confirmation_anchor_flags(anom_flags: list[bool], *, count: int, window: int) -> list[bool]:
    anchors = [False] * len(anom_flags)
    for i in range(len(anom_flags)):
        win_start = max(0, i - window + 1)
        trimmed = anom_flags[win_start : i + 1]
        if len(trimmed) < window:
            continue
        if sum(trimmed) < count:
            continue
        anchor_idx = win_start + max((len(trimmed) - 1) // 2, 0)
        anchors[anchor_idx] = True
    return anchors


def _build_sigmoid_scoring_figure(
    payload: dict[str, Any],
    *,
    use_index: bool,
    residual_ymin: float,
) -> plt.Figure:
    scored_display = payload["scored_display"]
    batch_df_plot = payload["batch_df_plot"]
    incidents = payload["incidents"]
    raw_time_values = payload["time_values"]
    raw_x = list(range(len(raw_time_values))) if use_index else raw_time_values
    batch_times = pd.to_datetime(batch_df_plot["window_mid"], utc=True).tolist() if not batch_df_plot.empty else []
    batch_x = payload["batch_index_x"] if use_index else batch_times

    fig = plt.figure(figsize=(20, 28))
    gs = fig.add_gridspec(
        9,
        2,
        height_ratios=[1.15, 1.15, 1.0, 1.2, 0.95, 0.85, 0.72, 1.55, 1.05],
    )
    ax_ts_vel = fig.add_subplot(gs[0, 0])
    sample_share = ax_ts_vel
    ax_ts_acc = fig.add_subplot(gs[0, 1], sharex=sample_share)
    ax_proc_vel = fig.add_subplot(gs[1, 0], sharex=sample_share)
    ax_proc_acc = fig.add_subplot(gs[1, 1], sharex=ax_ts_acc)
    ax_dnorm_vel = fig.add_subplot(gs[2, 0], sharex=sample_share)
    ax_dnorm_acc = fig.add_subplot(gs[2, 1], sharex=ax_ts_acc)
    ax_d_vel = fig.add_subplot(gs[3, 0], sharex=sample_share)
    ax_d_acc = fig.add_subplot(gs[3, 1], sharex=ax_ts_acc)
    ax_score_vel = fig.add_subplot(gs[4, 0])
    batch_share = ax_score_vel
    ax_score_acc = fig.add_subplot(gs[4, 1], sharex=batch_share)
    ax_thr_score_vel = fig.add_subplot(gs[5, 0], sharex=batch_share)
    ax_thr_score_acc = fig.add_subplot(gs[5, 1], sharex=batch_share)
    ax_mod_vel = fig.add_subplot(gs[6, 0], sharex=batch_share)
    ax_mod_acc = fig.add_subplot(gs[6, 1], sharex=batch_share)
    ax_fus = fig.add_subplot(gs[7, :], sharex=batch_share)
    ax_cause = fig.add_subplot(gs[8, :], sharex=batch_share)

    raw_display = payload["df_display"]
    split_col = "split"
    fit_mask = raw_display.get(split_col, pd.Series(index=raw_display.index, dtype=object)).eq("fit").to_numpy()
    pred_mask = raw_display.get(split_col, pd.Series(index=raw_display.index, dtype=object)).eq("pred").to_numpy()
    vel_display_mask = np.asarray(payload["vel_display_mask"], dtype=bool)
    accel_display_mask = np.asarray(payload["accel_display_mask"], dtype=bool)
    split_idx = payload["split_idx"]
    split_time = payload["split_time"]
    show = payload["show"]

    if show == "both":
        fit_y = [y if (keep and show_keep) else np.nan for y, keep, show_keep in zip(payload["raw_vel_values"], fit_mask, vel_display_mask)]
        pred_y = [y if (keep and show_keep) else np.nan for y, keep, show_keep in zip(payload["raw_vel_values"], pred_mask, vel_display_mask)]
        ax_ts_vel.plot(raw_x, fit_y, color=_D_VEL_COLOR, lw=1.0, alpha=0.60, label="fit")
        ax_ts_vel.plot(raw_x, pred_y, color=_D_VEL_COLOR, lw=1.2, alpha=0.95, label="pred")
        fit_y_acc = [y if (keep and show_keep) else np.nan for y, keep, show_keep in zip(payload["raw_accel_values"], fit_mask, accel_display_mask)]
        pred_y_acc = [y if (keep and show_keep) else np.nan for y, keep, show_keep in zip(payload["raw_accel_values"], pred_mask, accel_display_mask)]
        ax_ts_acc.plot(raw_x, fit_y_acc, color=_D_ACCEL_COLOR, lw=1.0, alpha=0.60, label="fit")
        ax_ts_acc.plot(raw_x, pred_y_acc, color=_D_ACCEL_COLOR, lw=1.2, alpha=0.95, label="pred")
    else:
        vel_y = [y if show_keep else np.nan for y, show_keep in zip(payload["raw_vel_values"], vel_display_mask)]
        accel_y = [y if show_keep else np.nan for y, show_keep in zip(payload["raw_accel_values"], accel_display_mask)]
        ax_ts_vel.plot(raw_x, vel_y, color=_D_VEL_COLOR, lw=1.1, alpha=0.85, label=show)
        ax_ts_acc.plot(raw_x, accel_y, color=_D_ACCEL_COLOR, lw=1.1, alpha=0.85, label=show)

    ax_ts_vel.set_title(f"{payload['raw_vel_col']} (pre-scoring raw series)")
    ax_ts_vel.set_ylabel(payload["raw_vel_col"])
    ax_ts_acc.set_title(f"{payload['raw_accel_col']} (pre-scoring raw series)")
    ax_ts_acc.set_ylabel(payload["raw_accel_col"])
    incident_patch = Patch(facecolor=_INCIDENT_COLOR, alpha=_INCIDENT_ALPHA, edgecolor="none", label="incident label")
    for ax in (ax_ts_vel, ax_ts_acc):
        handles, labels = ax.get_legend_handles_labels()
        handles = list(handles)
        labels = list(labels)
        if "incident label" not in labels:
            handles.append(incident_patch)
            labels.append("incident label")
        if handles:
            ax.legend(handles, labels, loc="upper left")
        ax.grid(True, alpha=0.25)
        _date_fmt(ax, use_index)

    if show == "both":
        fit_y_proc = [y if (keep and show_keep) else np.nan for y, keep, show_keep in zip(payload["proc_vel_values"], fit_mask, vel_display_mask)]
        pred_y_proc = [y if (keep and show_keep) else np.nan for y, keep, show_keep in zip(payload["proc_vel_values"], pred_mask, vel_display_mask)]
        ax_proc_vel.plot(raw_x, fit_y_proc, color=_D_VEL_COLOR, lw=1.0, alpha=0.60, label="fit")
        ax_proc_vel.plot(raw_x, pred_y_proc, color=_D_VEL_COLOR, lw=1.2, alpha=0.95, label="pred")
        fit_y_proc_acc = [y if (keep and show_keep) else np.nan for y, keep, show_keep in zip(payload["proc_accel_values"], fit_mask, accel_display_mask)]
        pred_y_proc_acc = [y if (keep and show_keep) else np.nan for y, keep, show_keep in zip(payload["proc_accel_values"], pred_mask, accel_display_mask)]
        ax_proc_acc.plot(raw_x, fit_y_proc_acc, color=_D_ACCEL_COLOR, lw=1.0, alpha=0.60, label="fit")
        ax_proc_acc.plot(raw_x, pred_y_proc_acc, color=_D_ACCEL_COLOR, lw=1.2, alpha=0.95, label="pred")
    else:
        vel_proc_y = [y if show_keep else np.nan for y, show_keep in zip(payload["proc_vel_values"], vel_display_mask)]
        accel_proc_y = [y if show_keep else np.nan for y, show_keep in zip(payload["proc_accel_values"], accel_display_mask)]
        ax_proc_vel.plot(raw_x, vel_proc_y, color=_D_VEL_COLOR, lw=1.1, alpha=0.85, label=show)
        ax_proc_acc.plot(raw_x, accel_proc_y, color=_D_ACCEL_COLOR, lw=1.1, alpha=0.85, label=show)

    ax_proc_vel.set_title(f"{payload['vel_col']} (processed series before residual scoring)")
    ax_proc_vel.set_ylabel(payload["vel_col"])
    ax_proc_acc.set_title(f"{payload['accel_col']} (processed series before residual scoring)")
    ax_proc_acc.set_ylabel(payload["accel_col"])
    for ax in (ax_proc_vel, ax_proc_acc):
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left")
        ax.grid(True, alpha=0.25)
        _date_fmt(ax, use_index)

    vel_dnorm_plot = [y if show_keep else np.nan for y, show_keep in zip(payload["vel_d"], vel_display_mask)]
    accel_dnorm_plot = [y if show_keep else np.nan for y, show_keep in zip(payload["accel_d"], accel_display_mask)]
    vel_d_plot = [y if show_keep else np.nan for y, show_keep in zip(payload["vel_residual"], vel_display_mask)]
    accel_d_plot = [y if show_keep else np.nan for y, show_keep in zip(payload["accel_residual"], accel_display_mask)]

    ax_dnorm_vel.plot(raw_x, vel_dnorm_plot, color=_D_VEL_COLOR, lw=1.0)
    ax_dnorm_vel.axhline(payload["threshold_vel"], color=_THRESH_COLOR, lw=1.0, ls="--", alpha=0.7, label=f"thr={payload['threshold_vel']:.2f}")
    ax_dnorm_vel.set_title("velocity normalized baseline distance (standard scaling)")
    ax_dnorm_vel.set_ylabel("norm distance")
    ax_dnorm_vel.legend(loc="upper left")
    ax_dnorm_vel.grid(True, alpha=0.25)
    _date_fmt(ax_dnorm_vel, use_index)

    ax_dnorm_acc.plot(raw_x, accel_dnorm_plot, color=_D_ACCEL_COLOR, lw=1.0)
    ax_dnorm_acc.axhline(payload["threshold_accel"], color=_THRESH_COLOR, lw=1.0, ls="--", alpha=0.7, label=f"thr={payload['threshold_accel']:.2f}")
    ax_dnorm_acc.set_title("acceleration normalized baseline distance (standard scaling)")
    ax_dnorm_acc.set_ylabel("norm distance")
    ax_dnorm_acc.legend(loc="upper left")
    ax_dnorm_acc.grid(True, alpha=0.25)
    _date_fmt(ax_dnorm_acc, use_index)

    ax_d_vel.plot(raw_x, vel_d_plot, color=_D_VEL_COLOR, lw=1.0)
    ax_d_vel.axhline(payload["beta_vel"], color=_D_VEL_COLOR, lw=1.0, ls="--", alpha=0.5, label=f"beta={payload['beta_vel']:.2f}")
    ax_d_vel.set_title("velocity residual (d_norm - threshold_vel)")
    ax_d_vel.set_ylabel("residual")
    ax_d_vel.legend(loc="upper left")
    ax_d_vel.grid(True, alpha=0.25)
    _date_fmt(ax_d_vel, use_index)

    ax_d_acc.plot(raw_x, accel_d_plot, color=_D_ACCEL_COLOR, lw=1.0)
    ax_d_acc.axhline(payload["beta_accel"], color=_D_ACCEL_COLOR, lw=1.0, ls="--", alpha=0.5, label=f"beta={payload['beta_accel']:.2f}")
    ax_d_acc.set_title("acceleration residual (d_norm - threshold_accel)")
    ax_d_acc.set_ylabel("residual")
    ax_d_acc.legend(loc="upper left")
    ax_d_acc.grid(True, alpha=0.25)
    _date_fmt(ax_d_acc, use_index)

    for col in _VEL_COLS:
        label = col.replace("vel_rms_", "")
        ax_score_vel.plot(batch_x, payload["vel_raw_window_score_by_col"][col], color=_VEL_AXIS_COLORS[col], lw=1.0, alpha=0.85, label=label)
    ax_score_vel.set_ylim(-0.05, 1.15)
    ax_score_vel.set_title(f"vel raw ch wnd score (top-{payload['window_top_k']} mean)")
    ax_score_vel.set_ylabel("channel score")
    ax_score_vel.legend(loc="upper left", ncol=3)
    ax_score_vel.grid(True, alpha=0.25)
    _date_fmt(ax_score_vel, use_index)

    for col in _ACCEL_COLS:
        label = col.replace("accel_rms_", "a_")
        ax_score_acc.plot(batch_x, payload["accel_raw_window_score_by_col"][col], color=_ACCEL_AXIS_COLORS[col], lw=1.0, alpha=0.85, label=label)
    ax_score_acc.set_ylim(-0.05, 1.15)
    ax_score_acc.set_title(f"acc raw ch wnd score (top-{payload['window_top_k']} mean)")
    ax_score_acc.set_ylabel("channel score")
    ax_score_acc.legend(loc="upper left", ncol=3)
    ax_score_acc.grid(True, alpha=0.25)
    _date_fmt(ax_score_acc, use_index)

    for col in _VEL_COLS:
        label = col.replace("vel_rms_", "")
        ax_thr_score_vel.plot(batch_x, payload["vel_occupancy_by_col"][col], color=_VEL_AXIS_COLORS[col], lw=1.0, alpha=0.85, label=label)
    ax_thr_score_vel.set_ylim(-0.05, 1.15)
    ax_thr_score_vel.set_title("vel channel occupancy used by anomaly gate")
    ax_thr_score_vel.set_ylabel("channel score")
    ax_thr_score_vel.legend(loc="upper left", ncol=3)
    ax_thr_score_vel.grid(True, alpha=0.25)
    _date_fmt(ax_thr_score_vel, use_index)

    for col in _ACCEL_COLS:
        label = col.replace("accel_rms_", "a_")
        ax_thr_score_acc.plot(batch_x, payload["accel_occupancy_by_col"][col], color=_ACCEL_AXIS_COLORS[col], lw=1.0, alpha=0.85, label=label)
    ax_thr_score_acc.set_ylim(-0.05, 1.15)
    ax_thr_score_acc.set_title("acc channel occupancy used by anomaly gate")
    ax_thr_score_acc.set_ylabel("channel score")
    ax_thr_score_acc.legend(loc="upper left", ncol=3)
    ax_thr_score_acc.grid(True, alpha=0.25)
    _date_fmt(ax_thr_score_acc, use_index)

    ax_mod_vel.plot(batch_x, batch_df_plot.get("vel_occupancy", pd.Series(dtype=float)).tolist(), color=_D_VEL_COLOR, lw=1.4, label="vel max")
    ax_mod_acc.plot(batch_x, batch_df_plot.get("accel_occupancy", pd.Series(dtype=float)).tolist(), color=_D_ACCEL_COLOR, lw=1.4, label="accel max")
    for ax, title in ((ax_mod_vel, "vel modality score"), (ax_mod_acc, "acc modality score")):
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.25)
        _date_fmt(ax, use_index)
    ax_mod_vel.set_ylabel("modality score")

    occ_scores = batch_df_plot.get("occupancy_score", pd.Series(dtype=float)).tolist()
    ax_fus.plot(batch_x, occ_scores, color="#111111", lw=2.0, label="fusion max")
    ax_fus.axhline(payload["fusion_thr"], color=_THRESH_COLOR, lw=1.0, ls="--", label=f"fusion_thr={payload['fusion_thr']:.2f}")
    ap = load_alert_params()
    anom_flags = [s >= payload["fusion_thr"] for s in occ_scores]
    confirmed_anchor_flags = _fast_confirmation_anchor_flags(
        anom_flags,
        count=ap.confirmation_count,
        window=ap.confirmation_window,
    )
    for i in range(len(batch_x) - 1):
        if anom_flags[i]:
            ax_fus.axvspan(batch_x[i], batch_x[i + 1], color="#ffcccc", alpha=0.12, zorder=0)
        if confirmed_anchor_flags[i]:
            ax_fus.axvspan(batch_x[i], batch_x[i + 1], color="#2ca02c", alpha=0.18, zorder=0)
    if batch_x:
        if anom_flags and anom_flags[-1]:
            ax_fus.axvspan(batch_x[-1], batch_x[-1], color="#ffcccc", alpha=0.12, zorder=0)
        if confirmed_anchor_flags and confirmed_anchor_flags[-1]:
            ax_fus.axvspan(batch_x[-1], batch_x[-1], color="#2ca02c", alpha=0.18, zorder=0)
    ax_fus.set_ylim(-0.02, 1.05)
    ax_fus.set_title(f"fusion score — confirmed anomaly ({ap.confirmation_count}/{ap.confirmation_window} gate, centered anchor)")
    ax_fus.set_ylabel("fusion score")
    ax_fus.legend(loc="upper left")
    ax_fus.grid(True, alpha=0.25)
    _date_fmt(ax_fus, use_index)

    mods = batch_df_plot.get("active_modalities", pd.Series(dtype=object))
    vel_on = mods.apply(lambda m: "vel" in m if isinstance(m, list) else False)
    acc_on = mods.apply(lambda m: "accel" in m if isinstance(m, list) else False)
    both = vel_on & acc_on
    vel_only = vel_on & ~acc_on
    acc_only = acc_on & ~vel_on
    ax_cause.vlines([xv for xv, k in zip(batch_x, acc_only) if k], 0, 1, color=_D_ACCEL_COLOR, lw=2.0, label="accel")
    ax_cause.vlines([xv for xv, k in zip(batch_x, both) if k], 1, 2, color=_BOTH_COLOR, lw=2.0, label="both")
    ax_cause.vlines([xv for xv, k in zip(batch_x, vel_only) if k], 2, 3, color=_D_VEL_COLOR, lw=2.0, label="vel")
    ax_cause.set_yticks([0.5, 1.5, 2.5])
    ax_cause.set_yticklabels(["accel", "both", "vel"])
    ax_cause.set_ylim(0, 3)
    ax_cause.set_ylabel("cause")
    ax_cause.set_title("active modality per anomaly window")
    ax_cause.legend(loc="upper right", ncol=3)
    ax_cause.grid(True, alpha=0.2, axis="x")
    _date_fmt(ax_cause, use_index)

    sample_split_x = split_idx if use_index else split_time
    if sample_split_x is not None:
        for ax in (ax_ts_vel, ax_ts_acc, ax_proc_vel, ax_proc_acc, ax_dnorm_vel, ax_dnorm_acc, ax_d_vel, ax_d_acc):
            ax.axvline(sample_split_x, color=_SPLIT_COLOR, ls="--", lw=1.0)

    batch_split_x = payload["batch_split_idx"] if use_index else payload["batch_split_time"]
    if batch_split_x is not None:
        for ax in (ax_score_vel, ax_score_acc, ax_thr_score_vel, ax_thr_score_acc, ax_mod_vel, ax_mod_acc, ax_fus, ax_cause):
            ax.axvline(batch_split_x, color=_SPLIT_COLOR, ls="--", lw=1.0)

    if use_index:
        ax_score_vel.set_xlabel("batch index")
        ax_score_acc.set_xlabel("batch index")
        ax_cause.set_xlabel("batch index")

    raw_spans = _incident_spans_for_axis(x_values=raw_x, timestamps=raw_time_values, incidents=incidents, use_index=use_index)
    batch_spans = _incident_spans_for_batches(
        x_values=batch_x,
        window_starts=pd.to_datetime(batch_df_plot["window_start"], utc=True).tolist() if not batch_df_plot.empty else [],
        window_ends=pd.to_datetime(batch_df_plot["window_end"], utc=True).tolist() if not batch_df_plot.empty else [],
        incidents=incidents,
        use_index=use_index,
    )
    for lo, hi in raw_spans:
        for ax in (ax_ts_vel, ax_ts_acc, ax_proc_vel, ax_proc_acc, ax_dnorm_vel, ax_dnorm_acc, ax_d_vel, ax_d_acc):
            ax.axvspan(lo, hi, color=_INCIDENT_COLOR, alpha=_INCIDENT_ALPHA, zorder=0)
    for lo, hi in batch_spans:
        for ax in (ax_score_vel, ax_score_acc, ax_thr_score_vel, ax_thr_score_acc, ax_mod_vel, ax_mod_acc, ax_fus, ax_cause):
            ax.axvspan(lo, hi, color=_INCIDENT_COLOR, alpha=_INCIDENT_ALPHA, zorder=0)

    res_vel_lo, res_vel_hi = _residual_bounds([y for y, keep in zip(payload["vel_residual"], vel_display_mask) if keep])
    res_acc_lo, res_acc_hi = _residual_bounds([y for y, keep in zip(payload["accel_residual"], accel_display_mask) if keep])
    ax_d_vel.set_ylim(min(float(residual_ymin), res_vel_lo), res_vel_hi)
    ax_d_acc.set_ylim(min(float(residual_ymin), res_acc_lo), res_acc_hi)

    fig.suptitle(
        f"Scenario {payload['scenario_id']} | {payload['scenario_group_label']} | view={payload['show']} "
        f"alpha={payload['alpha_vel']:.2f} beta={payload['beta_vel']:.2f} "
        f"vel_thr={payload['threshold_vel']:.2f} acc_thr={payload['threshold_accel']:.2f} "
        f"top_k={payload['window_top_k']} fus_thr={payload['fusion_thr']:.2f}",
        y=0.982,
    )
    fig.subplots_adjust(left=0.055, right=0.992, top=0.93, bottom=0.04, wspace=0.14, hspace=0.36)
    return fig
