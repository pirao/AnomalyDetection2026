"""Figure rendering helpers for the API replay widget.

Extracts the two compute-heavy functions from ``api_replay_widget`` so that
the widget module stays focused on state management and UI wiring:

- ``_compute_replay_plot_state`` — translates a replay DataFrame into
  pre-classified x-coordinate lists (anomaly, pending, open, realert,
  group events, resets) for both the raw-candidate and emitted-alert views.
- ``_plot_replay_column`` — draws the six-panel diagnostic figure column
  (scores, raw engine candidates, emitted API alerts, cause breakdown,
  active channels, per-channel residuals + group severity).

These functions are called exclusively from ``create_api_replay_widget_ui``.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def _with_split_view(split_view_value: str, pred_replay_df: pd.DataFrame, fit_replay_df: pd.DataFrame) -> pd.DataFrame:
    pred_local = pred_replay_df.copy()
    pred_local["split"] = "pred"
    if split_view_value == "pred":
        return pred_local

    fit_local = fit_replay_df.copy()
    fit_local["split"] = "fit"
    if split_view_value == "fit":
        return fit_local

    sort_col = "plot_time" if "plot_time" in pred_local.columns and "plot_time" in fit_local.columns else "window_mid"
    return (
        pd.concat([fit_local, pred_local], ignore_index=True)
        .sort_values(sort_col)
        .reset_index(drop=True)
    )

# ── Color palette ───────────────────────────────────────────────────────────────

_D_VEL_COLOR = "#1f77b4"
_D_ACCEL_COLOR = "#ff7f0e"
_THRESH_COLOR = "#d62728"
_BOTH_COLOR = "#9467bd"
_INCIDENT_COLOR = "#E07B00"
_SPLIT_COLOR = "black"
_GROUP3_MIXED_COLOR = "#6b5b95"
_GROUP_ACTIVE_COLOR = "#9fc7d8"
_GROUP_ACTIVE_ALPHA = 0.14
_INCIDENT_ALPHA = 0.10
_RAW_PENDING_COLOR = "#ff9800"
_RAW_OPEN_COLOR = "#2ca02c"
_RAW_REALERT_COLOR = _THRESH_COLOR
_RAW_GROUP3_FORMING_COLOR = "#bc8f5a"
_RAW_GROUP6_FORMING_COLOR = "#f28e8b"
_RESET_COLOR = "#7f7f7f"


# ── Low-level utilities ─────────────────────────────────────────────────────────

def _fmt_time_axis(ax):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _list_contains(series, token):
    return series.apply(lambda xs: token in xs if isinstance(xs, (list, tuple, set)) else False)


def _truthy_mask(series: pd.Series) -> pd.Series:
    return series.apply(bool)


def _as_list(value) -> list[object]:
    return list(value) if isinstance(value, (list, tuple, set)) else []


def _marker_handle(
    *,
    marker: str,
    label: str,
    color: str,
    markeredgecolor: str = "black",
    facecolor: str | None = None,
):
    return plt.Line2D(
        [0],
        [0],
        marker=marker,
        color="white",
        markerfacecolor=color if facecolor is None else facecolor,
        markeredgecolor=markeredgecolor,
        markersize=7,
        lw=0,
        label=label,
    )


# ── Plot-state computation ──────────────────────────────────────────────────────

def _compute_replay_plot_state(plot_df: pd.DataFrame, *, use_index: bool, alert_params) -> dict[str, object]:
    time_series = pd.to_datetime(
        plot_df["plot_time"] if "plot_time" in plot_df.columns else plot_df["window_mid"],
        utc=True,
        errors="coerce",
    )
    x = plot_df.index.to_list() if use_index else time_series
    x_list = list(x)

    group_open_mask = plot_df["group_opened"].fillna(False)
    group_realert_mask = plot_df["group_realerted"].fillna(False)
    group_reset_mask = plot_df["group_reset"].fillna(False)
    group_open_kind = plot_df.loc[group_open_mask, "group_mode_kind"].fillna("").tolist()
    group_realert_kind = plot_df.loc[group_realert_mask, "group_mode_kind"].fillna("").tolist()
    group_open_xs = [xx for xx, keep in zip(x_list, group_open_mask) if keep]
    group_realert_xs = [xx for xx, keep in zip(x_list, group_realert_mask) if keep]
    group_reset_xs = [xx for xx, keep in zip(x_list, group_reset_mask) if keep]
    group3_open_xs = [xx for xx, kind in zip(group_open_xs, group_open_kind) if kind == "group-3"]
    group6_open_xs = [xx for xx, kind in zip(group_open_xs, group_open_kind) if kind == "group-6"]
    group3_realert_xs = [xx for xx, kind in zip(group_realert_xs, group_realert_kind) if kind == "group-3"]
    group6_realert_xs = [xx for xx, kind in zip(group_realert_xs, group_realert_kind) if kind == "group-6"]
    show_group6 = bool(
        alert_params.enable_group6_alerts
        or group6_open_xs
        or group6_realert_xs
        or plot_df["group_mode_kind"].fillna("").eq("group-6").any()
        or plot_df["promotion_candidate_kind"].fillna("").eq("group-6").any()
    )

    pending_events = (
        plot_df["pending_lower_priority_events"].apply(_as_list)
        if "pending_lower_priority_events" in plot_df.columns
        else pd.Series([[] for _ in range(len(plot_df))], index=plot_df.index)
    )
    pending_channels_mask = (
        _truthy_mask(plot_df["pending_channels"])
        if "pending_channels" in plot_df.columns
        else pd.Series(False, index=plot_df.index)
    )
    opened_mask = _truthy_mask(plot_df["opened_channels"])
    realerted_mask = _truthy_mask(plot_df["realerted_channels"])
    reset_mask = _truthy_mask(plot_df["reset_channels"])

    raw_open_candidate_mask = opened_mask.copy()
    raw_realert_candidate_mask = realerted_mask.copy()
    for idx, events in enumerate(pending_events.tolist()):
        event_types = {str(event.get("event_type", "")) for event in events}
        if "channel_open" in event_types:
            raw_open_candidate_mask.iloc[idx] = True
        if "channel_realert" in event_types:
            raw_realert_candidate_mask.iloc[idx] = True

    raw_group3_forming_mask = (
        plot_df["promotion_candidate_kind"].fillna("").eq("group-3")
        & ~group_open_mask
        & ~group_realert_mask
    )
    raw_group6_forming_mask = (
        plot_df["promotion_candidate_kind"].fillna("").eq("group-6")
        & ~group_open_mask
        & ~group_realert_mask
    )
    raw_state = {
        "anomaly_xs": [xx for xx, keep in zip(x_list, plot_df["anomaly_status"].fillna(False)) if keep],
        "pending_xs": [xx for xx, keep in zip(x_list, pending_channels_mask) if keep],
        "open_candidate_xs": [xx for xx, keep in zip(x_list, raw_open_candidate_mask) if keep],
        "realert_candidate_xs": [xx for xx, keep in zip(x_list, raw_realert_candidate_mask) if keep],
        "group3_forming_xs": [xx for xx, keep in zip(x_list, raw_group3_forming_mask) if keep],
        "group6_forming_xs": [xx for xx, keep in zip(x_list, raw_group6_forming_mask) if keep],
        "group3_emitted_xs": sorted(set(group3_open_xs + group3_realert_xs)),
        "group6_emitted_xs": sorted(set(group6_open_xs + group6_realert_xs)),
        "reset_xs": [xx for xx, keep in zip(x_list, reset_mask | group_reset_mask) if keep],
    }

    superseded_channels: list[set[str]] = [set() for _ in range(len(plot_df))]
    for idx in np.flatnonzero(group_open_mask.to_numpy()):
        group_channels = set(_as_list(plot_df.iloc[idx].get("group_channels", [])))
        if not group_channels:
            continue
        released = set()
        for back in range(idx - 1, -1, -1):
            if bool(plot_df.iloc[back].get("group_reset", False)):
                break
            released |= set(_as_list(plot_df.iloc[back].get("reset_channels", []))) & group_channels
            if released == group_channels:
                break
            active_episode = (
                bool(plot_df.iloc[back].get("anomaly_status", False))
                or bool(_as_list(plot_df.iloc[back].get("watched_channels", [])))
                or bool(_as_list(plot_df.iloc[back].get("pending_channels", [])))
            )
            if not active_episode:
                break
            for key in ("opened_channels", "realerted_channels"):
                channels = set(_as_list(plot_df.iloc[back].get(key, [])))
                superseded_channels[back] |= channels & (group_channels - released)

    effective_open_mask = pd.Series(False, index=plot_df.index)
    effective_realert_mask = pd.Series(False, index=plot_df.index)
    for idx, channels in enumerate(plot_df["opened_channels"].apply(_as_list)):
        if str(plot_df.iloc[idx].get("owner_kind", "")) == "channel" and any(ch not in superseded_channels[idx] for ch in channels):
            effective_open_mask.iloc[idx] = True
    for idx, channels in enumerate(plot_df["realerted_channels"].apply(_as_list)):
        if str(plot_df.iloc[idx].get("owner_kind", "")) == "channel" and any(ch not in superseded_channels[idx] for ch in channels):
            effective_realert_mask.iloc[idx] = True

    effective_state = {
        "anomaly_xs": raw_state["anomaly_xs"],
        "pending_xs": [xx for xx, keep in zip(x_list, pending_channels_mask & ~plot_df["alert"].fillna(False)) if keep],
        "open_xs": [xx for xx, keep in zip(x_list, effective_open_mask) if keep],
        "realert_xs": [xx for xx, keep in zip(x_list, effective_realert_mask) if keep],
        "group3_xs": sorted(set(group3_open_xs + group3_realert_xs)),
        "group6_xs": sorted(set(group6_open_xs + group6_realert_xs)),
        "reset_xs": [xx for xx, keep in zip(x_list, reset_mask | group_reset_mask) if keep],
    }
    emitted_open_mask = plot_df["alert"].fillna(False) & plot_df["owner_kind"].fillna("").eq("channel") & _truthy_mask(plot_df["opened_channels"])
    emitted_realert_mask = plot_df["alert"].fillna(False) & plot_df["owner_kind"].fillna("").eq("channel") & _truthy_mask(plot_df["realerted_channels"])
    emitted_group3_mask = plot_df["alert"].fillna(False) & plot_df["group_mode_kind"].fillna("").eq("group-3")
    emitted_group6_mask = plot_df["alert"].fillna(False) & plot_df["group_mode_kind"].fillna("").eq("group-6")
    emitted_reset_mask = plot_df["alert"].fillna(False) & (reset_mask | group_reset_mask)
    emitted_state = {
        "anomaly_xs": raw_state["anomaly_xs"],
        "pending_xs": [xx for xx, keep in zip(x_list, pending_channels_mask & ~plot_df["alert"].fillna(False)) if keep],
        "open_xs": [xx for xx, keep in zip(x_list, emitted_open_mask) if keep],
        "realert_xs": [xx for xx, keep in zip(x_list, emitted_realert_mask) if keep],
        "group3_xs": [xx for xx, keep in zip(x_list, emitted_group3_mask) if keep],
        "group6_xs": [xx for xx, keep in zip(x_list, emitted_group6_mask) if keep],
        "reset_xs": [xx for xx, keep in zip(x_list, emitted_reset_mask) if keep],
        "all_alert_xs": [xx for xx, keep in zip(x_list, plot_df["alert"].fillna(False)) if keep],
    }

    return {
        "time_series": time_series,
        "x": x,
        "x_list": x_list,
        "show_group6": show_group6,
        "group_open_xs": group_open_xs,
        "group_realert_xs": group_realert_xs,
        "group_reset_xs": group_reset_xs,
        "group_open_kind": group_open_kind,
        "group_realert_kind": group_realert_kind,
        "group3_open_xs": group3_open_xs,
        "group6_open_xs": group6_open_xs,
        "raw": raw_state,
        "emitted": emitted_state,
        "effective": effective_state,
        "superseded_channels": superseded_channels,
    }


# ── Figure column renderer ──────────────────────────────────────────────────────

def _plot_replay_column(
    axes_col,
    plot_df: pd.DataFrame,
    *,
    incidents,
    use_index: bool,
    split_view_value: str,
    model_params,
    alert_params,
    title: str,
    anomaly_fill_color: str,
    time_col: str,
):
    axes_arr = np.asarray(axes_col).reshape(-1)
    if plot_df.empty:
        for ax in axes_arr:
            ax.text(0.5, 0.5, "No replay windows", transform=ax.transAxes, ha="center", va="center")
            ax.grid(True, alpha=0.2)
        axes_arr[0].set_title(title)
        return

    state = _compute_replay_plot_state(plot_df, use_index=use_index, alert_params=alert_params)
    time_series = state["time_series"]
    x = state["x"]
    x_list = state["x_list"]
    grouped_active = plot_df["group_mode_active"].fillna(False).tolist()
    split_boundaries: list[object] = []
    if split_view_value == "both" and "split" in plot_df.columns:
        split_values = plot_df["split"].astype(str).tolist()
        for i in range(1, len(split_values)):
            if split_values[i] != split_values[i - 1]:
                split_boundaries.append(i - 0.5 if use_index else x_list[i])

    def _span_true_regions(ax, flags, color, alpha):
        for i in range(len(x_list) - 1):
            if flags[i]:
                ax.axvspan(x_list[i], x_list[i + 1], color=color, alpha=alpha, zorder=0)
        if flags and flags[-1]:
            ax.axvspan(x_list[-1], x_list[-1], color=color, alpha=alpha, zorder=0)

    for ax in axes_arr:
        _span_true_regions(ax, grouped_active, _GROUP_ACTIVE_COLOR, _GROUP_ACTIVE_ALPHA)

    axes_arr[0].plot(x, plot_df["vel_occupancy"], lw=1.5, color=_D_VEL_COLOR, label="vel score")
    axes_arr[0].plot(x, plot_df["accel_occupancy"], lw=1.5, color=_D_ACCEL_COLOR, label="accel score")
    axes_arr[0].plot(x, plot_df["occupancy_score"], lw=2.4, color="#111111", label="final score")
    axes_arr[0].axhline(model_params.fusion_threshold, color=_THRESH_COLOR, ls="--", lw=1.0, label=f"fusion_thr={model_params.fusion_threshold:.2f}")
    axes_arr[0].set_ylabel("score")
    axes_arr[0].set_ylim(-0.02, 1.05)
    top_handles, top_labels = axes_arr[0].get_legend_handles_labels()
    top_handles.extend(
        [
            Patch(facecolor=_INCIDENT_COLOR, alpha=_INCIDENT_ALPHA, edgecolor="none", label="incident label"),
            Patch(facecolor=_GROUP_ACTIVE_COLOR, alpha=_GROUP_ACTIVE_ALPHA, edgecolor="none", label="group mode active"),
        ]
    )
    top_labels.extend(["incident label", "group mode active"])
    axes_arr[0].legend(top_handles, top_labels, loc="upper left")
    axes_arr[0].grid(True, alpha=0.25)

    anom_list = plot_df["anomaly_status"].tolist()
    for i in range(len(x_list) - 1):
        if anom_list[i]:
            axes_arr[1].axvspan(x_list[i], x_list[i + 1], color=anomaly_fill_color, alpha=0.22, zorder=0)
            axes_arr[2].axvspan(x_list[i], x_list[i + 1], color=anomaly_fill_color, alpha=0.16, zorder=0)
    if anom_list and anom_list[-1]:
        axes_arr[1].axvspan(x_list[-1], x_list[-1], color=anomaly_fill_color, alpha=0.22, zorder=0)
        axes_arr[2].axvspan(x_list[-1], x_list[-1], color=anomaly_fill_color, alpha=0.16, zorder=0)
    raw = state["raw"]
    if raw["anomaly_xs"]:
        axes_arr[1].scatter(raw["anomaly_xs"], [0.0] * len(raw["anomaly_xs"]), marker="o", s=28, color="#1f77b4", alpha=0.85, label="anomaly", zorder=4)
    if raw["pending_xs"]:
        axes_arr[1].scatter(raw["pending_xs"], [0.35] * len(raw["pending_xs"]), marker="^", s=60, facecolors="none", edgecolors=_RAW_PENDING_COLOR, linewidths=1.5, label="ind pending", zorder=5)
    if raw["open_candidate_xs"]:
        axes_arr[1].scatter(raw["open_candidate_xs"], [0.70] * len(raw["open_candidate_xs"]), marker="^", s=85, color=_RAW_OPEN_COLOR, label="channel open cand", zorder=6)
    if raw["realert_candidate_xs"]:
        axes_arr[1].scatter(raw["realert_candidate_xs"], [0.88] * len(raw["realert_candidate_xs"]), marker="D", s=70, color=_RAW_REALERT_COLOR, label="channel re-alert cand", zorder=6)
    if raw["group3_forming_xs"]:
        axes_arr[1].scatter(raw["group3_forming_xs"], [1.15] * len(raw["group3_forming_xs"]), marker="s", s=88, facecolors="none", edgecolors=_RAW_GROUP3_FORMING_COLOR, linewidths=1.5, label="group-3 forming", zorder=6)
    if state["show_group6"] and raw["group6_forming_xs"]:
        axes_arr[1].scatter(raw["group6_forming_xs"], [1.15] * len(raw["group6_forming_xs"]), marker="H", s=95, facecolors="none", edgecolors=_RAW_GROUP6_FORMING_COLOR, linewidths=1.5, label="group-6 forming", zorder=6)
    if raw["group3_emitted_xs"]:
        axes_arr[1].scatter(raw["group3_emitted_xs"], [1.45] * len(raw["group3_emitted_xs"]), marker="s", s=95, color="#8c564b", label="group-3 engine emit", zorder=7)
    if state["show_group6"] and raw["group6_emitted_xs"]:
        axes_arr[1].scatter(raw["group6_emitted_xs"], [1.45] * len(raw["group6_emitted_xs"]), marker="H", s=105, color="#d62728", label="group-6 engine emit", zorder=7)
    if raw["reset_xs"]:
        axes_arr[1].scatter(raw["reset_xs"], [1.70] * len(raw["reset_xs"]), marker="X", s=95, color=_RESET_COLOR, label="reset", zorder=7)
    axes_arr[1].set_yticks([0.0, 0.35, 0.80, 1.15, 1.45, 1.70])
    axes_arr[1].set_yticklabels(["anomaly", "pending", "ind cand", "group forming", "group emit", "reset"])
    axes_arr[1].set_ylim(-0.2, 1.95)
    axes_arr[1].set_ylabel("engine diagnostics")
    axes_arr[1].legend(loc="upper left", ncol=2)
    axes_arr[1].grid(True, alpha=0.2)

    emitted = state["emitted"]
    if emitted["anomaly_xs"]:
        axes_arr[2].scatter(emitted["anomaly_xs"], [0.0] * len(emitted["anomaly_xs"]), marker="o", s=28, color="#1f77b4", alpha=0.85, label="anomaly", zorder=4)
    if emitted["pending_xs"]:
        axes_arr[2].scatter(emitted["pending_xs"], [0.5] * len(emitted["pending_xs"]), marker="^", s=60, facecolors="none", edgecolors=_RAW_PENDING_COLOR, linewidths=1.5, label="pending", zorder=5)
    if emitted["open_xs"]:
        axes_arr[2].scatter(emitted["open_xs"], [1.0] * len(emitted["open_xs"]), marker="^", s=90, color=_RAW_OPEN_COLOR, label="API alert: channel open", zorder=6)
    if emitted["realert_xs"]:
        axes_arr[2].scatter(emitted["realert_xs"], [1.0] * len(emitted["realert_xs"]), marker="D", s=75, color=_THRESH_COLOR, label="API alert: channel re-alert", zorder=6)
    if emitted["group3_xs"]:
        axes_arr[2].scatter(emitted["group3_xs"], [1.0] * len(emitted["group3_xs"]), marker="s", s=95, color="#8c564b", label="API alert: group-3", zorder=7)
    if state["show_group6"] and emitted["group6_xs"]:
        axes_arr[2].scatter(emitted["group6_xs"], [1.0] * len(emitted["group6_xs"]), marker="H", s=105, color="#d62728", label="API alert: group-6", zorder=7)
    if emitted["reset_xs"]:
        axes_arr[2].scatter(emitted["reset_xs"], [1.0] * len(emitted["reset_xs"]), marker="X", s=100, color=_RESET_COLOR, label="reset", zorder=7)
    axes_arr[2].set_yticks([0, 0.5, 1.0])
    axes_arr[2].set_yticklabels(["anomaly", "pending", "API alert"])
    axes_arr[2].set_ylim(-0.5, 1.5)
    axes_arr[2].set_ylabel("API emitted alerts")
    axes_arr[2].legend(loc="upper left", ncol=2)
    axes_arr[2].grid(True, alpha=0.2)

    vel_sel = _list_contains(plot_df["active_modalities"], "vel")
    acc_sel = _list_contains(plot_df["active_modalities"], "accel")
    both_sel = vel_sel & acc_sel
    vel_only = vel_sel & ~acc_sel
    acc_only = acc_sel & ~vel_sel
    group3_event_x = emitted["group3_xs"]
    group6_event_x = emitted["group6_xs"]
    _clw = 2.0
    axes_arr[3].vlines([xx for xx, k in zip(x_list, acc_only) if k], 0, 1, color=_D_ACCEL_COLOR, lw=_clw, label="accel")
    axes_arr[3].vlines([xx for xx, k in zip(x_list, both_sel) if k], 1, 2, color=_BOTH_COLOR, lw=_clw, label="both")
    axes_arr[3].vlines([xx for xx, k in zip(x_list, vel_only) if k], 2, 3, color=_D_VEL_COLOR, lw=_clw, label="vel")
    axes_arr[3].vlines(sorted(set(group3_event_x)), 3, 4, color="#8c564b", lw=_clw + 0.5, label="group-3")
    if state["show_group6"]:
        axes_arr[3].vlines(sorted(set(group6_event_x)), 4, 5, color="#d62728", lw=_clw + 0.5, label="group-6")
        axes_arr[3].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
        axes_arr[3].set_yticklabels(["accel", "both", "vel", "group-3", "group-6"])
        axes_arr[3].set_ylim(0, 5)
    else:
        axes_arr[3].set_yticks([0.5, 1.5, 2.5, 3.5])
        axes_arr[3].set_yticklabels(["accel", "both", "vel", "group-3"])
        axes_arr[3].set_ylim(0, 4)
    axes_arr[3].set_ylabel("cause")
    axes_arr[3].legend(loc="upper left", ncol=4)
    axes_arr[3].grid(True, alpha=0.2, axis="x")

    channel_order = ["vel_rms_x", "vel_rms_y", "vel_rms_z", "accel_rms_x", "accel_rms_y", "accel_rms_z"]
    ch_colors = ["#1f77b4", "#17becf", "#0d3b66", "#ff7f0e", "#d62728", "#9467bd"]
    ch_labels = ["v_x", "v_y", "v_z", "a_x", "a_y", "a_z"]
    for j, (ch, color) in enumerate(zip(channel_order, ch_colors), start=1):
        ch_sel = _list_contains(plot_df["active_channels"], ch)
        axes_arr[4].scatter([xx for xx, keep in zip(x_list, ch_sel) if keep], [j] * int(ch_sel.sum()), s=30, color=color)
    axes_arr[4].set_yticks(range(1, len(channel_order) + 1))
    axes_arr[4].set_yticklabels(ch_labels)
    axes_arr[4].set_ylabel("channels")
    axes_arr[4].grid(True, alpha=0.2)

    channels_to_plot: list[str] = []
    for ch in channel_order:
        keep = plot_df["watched_channels"].apply(
            lambda xs: ch in xs if isinstance(xs, (list, tuple, set)) else False
        ).any()
        keep = keep or plot_df["triggered_channels"].apply(
            lambda xs: ch in xs if isinstance(xs, (list, tuple, set)) else False
        ).any()
        keep = keep or plot_df["reset_channels"].apply(
            lambda xs: ch in xs if isinstance(xs, (list, tuple, set)) else False
        ).any()
        if keep:
            channels_to_plot.append(ch)
    if not channels_to_plot:
        channels_to_plot = channel_order

    for j, (ch, color) in enumerate(zip(channel_order, ch_colors), start=1):
        if ch not in channels_to_plot:
            continue
        ch_series = plot_df["channel_max_residual"].apply(
            lambda d: float(d.get(ch, 0.0)) if isinstance(d, dict) else 0.0
        )
        axes_arr[5].plot(x, ch_series, lw=1.4, alpha=0.95, color=color, label=ch_labels[j - 1])

        open_mask = plot_df["opened_channels"].apply(lambda xs: ch in xs if isinstance(xs, (list, tuple, set)) else False)
        realert_mask = plot_df["realerted_channels"].apply(lambda xs: ch in xs if isinstance(xs, (list, tuple, set)) else False)
        reset_mask = plot_df["reset_channels"].apply(
            lambda xs: ch in xs if isinstance(xs, (list, tuple, set)) else False
        )
        effective_channel_open_mask = pd.Series(
            [bool(open_mask.iloc[i]) and ch not in state["superseded_channels"][i] and str(plot_df.iloc[i].get("owner_kind", "")) == "channel" for i in range(len(plot_df))],
            index=plot_df.index,
        )
        effective_channel_realert_mask = pd.Series(
            [bool(realert_mask.iloc[i]) and ch not in state["superseded_channels"][i] and str(plot_df.iloc[i].get("owner_kind", "")) == "channel" for i in range(len(plot_df))],
            index=plot_df.index,
        )
        open_x = [xx for xx, keep in zip(x_list, effective_channel_open_mask) if keep]
        open_y = [yy for yy, keep in zip(ch_series.tolist(), effective_channel_open_mask) if keep]
        realert_x = [xx for xx, keep in zip(x_list, effective_channel_realert_mask) if keep]
        realert_y = [yy for yy, keep in zip(ch_series.tolist(), effective_channel_realert_mask) if keep]
        reset_x = [xx for xx, keep in zip(x_list, reset_mask) if keep]
        reset_y = [yy for yy, keep in zip(ch_series.tolist(), reset_mask) if keep]
        if open_x:
            axes_arr[5].scatter(open_x, open_y, marker="^", s=55, color=color, edgecolors="black", linewidths=0.4, zorder=6)
        if realert_x:
            axes_arr[5].scatter(realert_x, realert_y, marker="D", s=48, color=color, edgecolors="black", linewidths=0.4, zorder=6)
        if reset_x:
            axes_arr[5].scatter(reset_x, reset_y, marker="X", s=60, color=color, edgecolors="black", linewidths=0.4, zorder=7)
            for xx in reset_x:
                axes_arr[5].axvline(xx, color=color, lw=2.0, alpha=0.12, zorder=0)

    group_kind_series = plot_df["group_mode_kind"].fillna("")
    group_type_series = plot_df["group_mode_type"].fillna("")
    vel_cluster_series = plot_df["group_current_severity"].where(
        group_kind_series.eq("group-3") & group_type_series.eq("velocity_cluster_degradation")
    )
    accel_cluster_series = plot_df["group_current_severity"].where(
        group_kind_series.eq("group-3") & group_type_series.eq("acceleration_cluster_degradation")
    )
    mixed_cluster_series = plot_df["group_current_severity"].where(
        group_kind_series.eq("group-3") & group_type_series.eq("mixed_cluster_degradation")
    )
    full_series = plot_df["group_current_severity"].where(group_kind_series.eq("group-6"))
    axes_arr[5].plot(x, vel_cluster_series, lw=1.4, color="#4d4d4d", alpha=0.95, label="group-3 sev vel (resid norm)")
    axes_arr[5].plot(x, accel_cluster_series, lw=1.4, color="#8c564b", alpha=0.95, label="group-3 sev accel (resid norm)")
    axes_arr[5].plot(x, mixed_cluster_series, lw=1.4, color=_GROUP3_MIXED_COLOR, alpha=0.95, label="group-3 sev mixed (resid norm)")
    if state["show_group6"]:
        axes_arr[5].plot(x, full_series, lw=1.2, color="#e377c2", alpha=0.95, label="group-6 sev (l2 norm)")
    if state["group_open_xs"]:
        open_df = plot_df.loc[plot_df["group_opened"].fillna(False)]
        open_y = open_df["group_current_severity"].tolist()
        open_types = open_df["group_mode_type"].fillna("").tolist()
        open_kinds_local = open_df["group_mode_kind"].fillna("").tolist()
        open_colors = [
            "#e377c2" if k == "group-6" else _GROUP3_MIXED_COLOR if t == "mixed_cluster_degradation" else "#8c564b" if t == "acceleration_cluster_degradation" else "#4d4d4d"
            for k, t in zip(open_kinds_local, open_types)
        ]
        open_markers = ["H" if k == "group-6" else "s" for k in open_kinds_local]
        for xx, yy, color, marker in zip(state["group_open_xs"], open_y, open_colors, open_markers):
            axes_arr[5].scatter([xx], [yy], marker=marker, s=74 if marker == "H" else 70, color=color, edgecolors="black", linewidths=0.4, zorder=8)
    if state["group_realert_xs"]:
        rel_y = plot_df.loc[plot_df["group_realerted"].fillna(False), "group_current_severity"].tolist()
        rel_types = plot_df.loc[plot_df["group_realerted"].fillna(False), "group_mode_type"].fillna("").tolist()
        rel_colors = ["#e377c2" if kind == "group-6" else _GROUP3_MIXED_COLOR if rel_types[i] == "mixed_cluster_degradation" else "#8c564b" if rel_types[i] == "acceleration_cluster_degradation" else "#4d4d4d" for i, kind in enumerate(state["group_realert_kind"])]
        axes_arr[5].scatter(state["group_realert_xs"], rel_y, marker="P", s=80, color=rel_colors, edgecolors="white", linewidths=0.4, zorder=8)
    if state["group_reset_xs"]:
        reset_y = plot_df.loc[plot_df["group_reset"].fillna(False), "group_current_severity"].tolist()
        axes_arr[5].scatter(state["group_reset_xs"], reset_y, marker="X", s=86, color="#7f7f7f", edgecolors="black", linewidths=0.4, zorder=8)
    axes_arr[5].set_ylabel("residual")
    axes_arr[5].set_xlabel("batch index" if use_index else time_col)
    upper_vals = [float(pd.to_numeric(plot_df["group_current_severity"], errors="coerce").max())]
    for ch in channels_to_plot:
        upper_vals.append(
            float(
                plot_df["channel_max_residual"].apply(
                    lambda d: float(d.get(ch, np.nan)) if isinstance(d, dict) else np.nan
                ).max()
            )
        )
    upper_y = max([v for v in upper_vals if pd.notna(v)], default=5.0)
    axes_arr[5].set_ylim(-5.0, max(upper_y * 1.08, 5.0))
    axes_arr[5].grid(True, alpha=0.2)
    line_handles, line_labels = axes_arr[5].get_legend_handles_labels()
    marker_handles = [
        _marker_handle(marker="^", label="marker: individual open", color="#2ca02c"),
        _marker_handle(marker="D", label="marker: individual re-alert", color=_THRESH_COLOR),
        _marker_handle(marker="s", label="marker: group-3 open", color="#8c564b"),
        _marker_handle(marker="P", label="marker: group re-alert", color="#111111", markeredgecolor="white"),
        _marker_handle(marker="X", label="marker: reset", color="#7f7f7f"),
    ]
    if state["show_group6"]:
        marker_handles.insert(
            3,
            _marker_handle(marker="H", label="marker: group-6 open", color="#d62728"),
        )
    axes_arr[5].legend(line_handles + marker_handles, line_labels + [h.get_label() for h in marker_handles], loc="upper left", ncol=3)

    for ax in axes_arr:
        for inc in incidents:
            start_ts = pd.Timestamp(inc["start"])
            end_ts = pd.Timestamp(inc["end"])
            start_ts = start_ts.tz_convert("UTC") if start_ts.tzinfo is not None else start_ts.tz_localize("UTC")
            end_ts = end_ts.tz_convert("UTC") if end_ts.tzinfo is not None else end_ts.tz_localize("UTC")

            if use_index:
                mask = (time_series >= start_ts) & (time_series <= end_ts)
                idxs = plot_df.index[mask].tolist()
                if idxs:
                    ax.axvspan(min(idxs), max(idxs), color=_INCIDENT_COLOR, alpha=_INCIDENT_ALPHA)
            else:
                ax.axvspan(start_ts, end_ts, color=_INCIDENT_COLOR, alpha=_INCIDENT_ALPHA)
        for split_boundary in split_boundaries:
            ax.axvline(split_boundary, color=_SPLIT_COLOR, ls="--", lw=1.0, alpha=0.9)
        if not use_index:
            _fmt_time_axis(ax)

    axes_arr[0].set_title(title)
