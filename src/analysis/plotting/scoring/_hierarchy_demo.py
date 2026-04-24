"""Visual explainers for the alert engine priority hierarchy.

Two static figures used from ``notebooks/02_model_debugging.ipynb`` section 4:

- ``plot_alert_hierarchy_diagram`` renders a conceptual state-machine diagram
  covering the three priority tiers (individual, group-3, group-6), the
  holdback / suppression relationships between them, and the per-tier
  open / re-alert / reset gates. Reads values live from
  ``alert_hyperparams.yaml`` so the diagram stays in sync with the engine.

- ``plot_alert_hierarchy_trace_scenario3`` renders an actual scenario-3
  replay timeline showing how per-channel and grouped state evolves
  batch-by-batch, anchored to the real replay DataFrame produced by
  ``simulate_api_replay_one_scenario``.
"""

from __future__ import annotations

from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.api_replay import simulate_api_replay_one_scenario
from sample_processing.model.anomaly_model import (
    load_alert_params,
    load_model_params,
)

from ._colors import _ACCEL_AXIS_COLORS, _INCIDENT_ALPHA, _INCIDENT_COLOR, _VEL_AXIS_COLORS


_PENDING_COLOR = "#ff9800"
_PENDING_EMPTY_COLOR = "#ffe5c2"
_OPEN_COLOR = "#2ca02c"
_REALERT_COLOR = "#d62728"
_RESET_COLOR = "#7f7f7f"
_GROUP3_FORMING_COLOR = "#bc8f5a"
_GROUP3_FORMING_EMPTY_COLOR = "#ffe7d0"
_GROUP3_ACTIVE_COLOR = "#9fc7d8"
_GROUP6_DISABLED_COLOR = "#f28e8b"
_GROUP3_BAND_COLOR = "#6b5b95"
_INDIVIDUAL_BAND_COLOR = "#2ca02c"
_TRACK_BG = "#f5f5f5"
_BELOW_ZERO_COLOR = "#7f7f7f"

_CHANNELS = ["vel_rms_x", "vel_rms_y", "vel_rms_z", "accel_rms_x", "accel_rms_y", "accel_rms_z"]
_CHANNEL_DISPLAY = {
    "vel_rms_x": "vel_x",
    "vel_rms_y": "vel_y",
    "vel_rms_z": "vel_z",
    "accel_rms_x": "accel_x",
    "accel_rms_y": "accel_y",
    "accel_rms_z": "accel_z",
}
_CHANNEL_COLORS = {**_VEL_AXIS_COLORS, **_ACCEL_AXIS_COLORS}


# ── Figure 1: conceptual hierarchy diagram ──────────────────────────────────


def plot_alert_hierarchy_diagram() -> plt.Figure:
    """Static conceptual diagram of the alert priority hierarchy."""
    ap = load_alert_params()

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 85)
    ax.set_axis_off()

    fig.suptitle(
        "Alert engine priority hierarchy — individual → group-3 → group-6",
        fontsize=15, fontweight="bold", y=0.97,
    )

    col_x = {"open": 10, "realert": 44, "reset": 72}

    def _band(y, height, *, fill, edge, dashed, tier_label, subtitle, open_lines, realert_lines, reset_lines, priority):
        linestyle = (0, (8, 4)) if dashed else "solid"
        rect = mpatches.FancyBboxPatch(
            (4, y), 88, height,
            boxstyle="round,pad=0.5,rounding_size=1.5",
            linewidth=2, edgecolor=edge, facecolor=fill, linestyle=linestyle,
        )
        ax.add_patch(rect)

        ax.text(6, y + height - 3, tier_label, fontsize=12, fontweight="bold",
                va="top", color="#222")
        if subtitle:
            ax.text(6, y + height - 7, subtitle, fontsize=9, style="italic",
                    color="#a33", va="top")

        base_y = y + height - 11 if subtitle else y + height - 7
        for x_col, header, lines in [
            (col_x["open"], "Open", open_lines),
            (col_x["realert"], "Re-alert", realert_lines),
            (col_x["reset"], "Reset", reset_lines),
        ]:
            ax.text(x_col, base_y, header, fontsize=11, fontweight="bold", va="top")
            for i, line in enumerate(lines):
                ax.text(x_col, base_y - 4 - 3.2 * i, line, fontsize=9, color="#333", va="top")

        ax.text(96, y + height / 2, str(priority), fontsize=14, fontweight="bold",
                ha="center", va="center")

    _band(
        y=60, height=22,
        fill=(1, 0.55, 0.55, 0.15), edge=_GROUP6_DISABLED_COLOR, dashed=True,
        tier_label="TIER 2 — Group-6 (vel triplet + accel triplet)",
        subtitle=f"enable_group6_alerts: {ap.enable_group6_alerts} — implemented, disabled in prod",
        open_lines=[f"{ap.group_confirmation_count}-of-{ap.group_confirmation_window} group confirmations"],
        realert_lines=[
            f"severity ≥ ref × {1.0 + ap.group_relative_threshold:g}",
            f"cooldown ≥ {ap.group_min_cooldown_windows} windows",
        ],
        reset_lines=[
            f"strict majority < 0 for {ap.group_reset_batches_below_zero} batches",
        ],
        priority=2,
    )

    _band(
        y=33, height=22,
        fill=(0.42, 0.36, 0.58, 0.22), edge=_GROUP3_BAND_COLOR, dashed=False,
        tier_label="TIER 1 — Group-3 (vel triplet / accel triplet / mixed 3–5 channels)",
        subtitle=None,
        open_lines=[
            f"{ap.group_confirmation_count}-of-{ap.group_confirmation_window} group confirmations",
            f"(group_confirmation_count = {ap.group_confirmation_count})",
        ],
        realert_lines=[
            f"severity ≥ ref × {1.0 + ap.group_relative_threshold:g}",
            f"cooldown ≥ {ap.group_min_cooldown_windows} windows",
        ],
        reset_lines=[
            f"strict majority < 0 for {ap.group_reset_batches_below_zero} batches",
            "(group_reset_batches_below_zero)",
        ],
        priority=1,
    )

    _band(
        y=6, height=22,
        fill=(0.17, 0.63, 0.17, 0.22), edge=_INDIVIDUAL_BAND_COLOR, dashed=False,
        tier_label=f"TIER 0 — Individual (single channel ownership, {ap.individual_alert_mode} mode)",
        subtitle=None,
        open_lines=[
            f"{ap.confirmation_count}-of-{ap.confirmation_window} confirmations",
            f"(count = {ap.confirmation_count}, window = {ap.confirmation_window})",
        ],
        realert_lines=[
            f"severity ≥ ref × {1.0 + ap.relative_threshold:g}",
            f"cooldown ≥ {ap.min_cooldown_windows} win., global {ap.inter_alert_cooldown_windows}-win. mute",
        ],
        reset_lines=[
            f"{ap.reset_batches_below_zero} consecutive batches < 0",
            "(reset_batches_below_zero)",
        ],
        priority=0,
    )

    _holdback_arrow(
        ax, y_from=56, y_to=60,
        label_lines=[
            f"group3_to_group6_holdback_windows = {ap.group3_to_group6_holdback_windows}",
            f"group-3 events held up to {ap.group3_to_group6_holdback_windows} windows while group-6 forming",
        ],
    )
    _holdback_arrow(
        ax, y_from=28, y_to=33,
        label_lines=[
            f"channel_to_group_holdback_windows = {ap.channel_to_group_holdback_windows}",
            f"individual events held up to {ap.channel_to_group_holdback_windows} windows while group-3 forming",
        ],
    )

    ax.text(96, 80, "Priority", fontsize=10, color="#555", ha="center", va="center")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _holdback_arrow(ax, *, y_from, y_to, label_lines):
    ax.annotate(
        "",
        xy=(50, y_to), xytext=(50, y_from),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.5),
    )
    mid = (y_from + y_to) / 2
    ax.text(52, mid + 0.8, label_lines[0], fontsize=9, color="#333", fontweight="bold", va="center")
    ax.text(52, mid - 1.4, label_lines[1], fontsize=9, color="#333", va="center")


# ── Figure 2: scenario-3 trace ──────────────────────────────────────────────


def plot_alert_hierarchy_trace_scenario3(
    *,
    full_df: pd.DataFrame,
    fitted_models: dict[int, Any],
    scenario_id: int = 3,
    time_col: str = "sampled_at",
    split_col: str = "split",
    fit_value: str = "fit",
    pred_value: str = "pred",
) -> plt.Figure:
    """Render an actual scenario-3 replay trace illustrating the hierarchy."""
    df_sid = full_df[full_df["scenario_id"] == scenario_id].sort_values(time_col).copy()
    fit_df = df_sid[df_sid[split_col] == fit_value].copy()
    pred_df = df_sid[df_sid[split_col] == pred_value].copy()
    if fit_df.empty or pred_df.empty:
        raise ValueError(f"Scenario {scenario_id}: missing fit or pred split.")

    model = fitted_models.get(scenario_id)
    if model is None:
        raise ValueError(f"No fitted model found for scenario {scenario_id} in fitted_models.")

    alert_params = load_alert_params()
    model_params = load_model_params(scenario_id=scenario_id)

    replay_df = simulate_api_replay_one_scenario(
        fit_df=fit_df,
        pred_df=pred_df,
        model=model,
        sensor_id=f"sensor_{scenario_id}",
        baseline_scaler=None,
        model_params_override=model_params,
        alert_params=alert_params,
        time_col=time_col,
    ).reset_index(drop=True)

    if replay_df.empty:
        raise ValueError(f"Replay produced no batches for scenario {scenario_id}.")

    n = len(replay_df)
    batch_x = np.arange(n)

    incident_spans = _incident_batch_spans(df_sid, replay_df, time_col=time_col)
    confirmation_region = _confirmation_region(replay_df)
    open_event = _first_individual_open_event(replay_df)
    group3_open_batch = _first_group3_open(replay_df)
    group3_reset_batch = _first_group3_reset(replay_df)
    individual_reset_batch = _first_individual_reset(replay_df)
    below_zero_start = _first_all_below_zero(replay_df)

    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[2.8, 1.5, 0.7, 0.5],
        hspace=0.15,
        left=0.11, right=0.97, top=0.94, bottom=0.28,
    )
    ax_res = fig.add_subplot(gs[0, 0])
    ax_ind = fig.add_subplot(gs[1, 0], sharex=ax_res)
    ax_g3 = fig.add_subplot(gs[2, 0], sharex=ax_res)
    ax_g6 = fig.add_subplot(gs[3, 0], sharex=ax_res)

    fig.suptitle(
        f"Scenario {scenario_id} — open → group-3 takeover → group-3 reset → individual reset",
        fontsize=14, fontweight="bold", y=0.985,
    )

    for ax in (ax_res, ax_ind, ax_g3, ax_g6):
        for lo, hi in incident_spans:
            ax.axvspan(lo, hi + 1, color=_INCIDENT_COLOR, alpha=_INCIDENT_ALPHA, zorder=0)

    _draw_residuals(ax_res, replay_df, batch_x, below_zero_start=below_zero_start)
    _draw_individual_ribbon(ax_ind, replay_df, batch_x,
                            confirmation_region=confirmation_region,
                            open_event=open_event)
    _draw_group3_ribbon(ax_g3, replay_df, batch_x,
                        confirmation_region=confirmation_region)
    _draw_group6_ribbon(ax_g6, batch_x, alert_params=alert_params)

    for ax in (ax_res, ax_ind, ax_g3):
        plt.setp(ax.get_xticklabels(), visible=False)

    ax_g6.set_xlabel("batch index (1 batch = 1 hour stride)", fontsize=11, fontweight="bold")
    ax_g6.set_xlim(-0.5, n - 0.5)

    _add_numbered_annotations(
        fig,
        replay_df=replay_df,
        confirmation_region=confirmation_region,
        open_event=open_event,
        group3_open_batch=group3_open_batch,
        group3_reset_batch=group3_reset_batch,
        individual_reset_batch=individual_reset_batch,
        ax_top=ax_res,
    )
    _add_legend(fig)

    return fig


# ── trace helpers ───────────────────────────────────────────────────────────


def _incident_batch_spans(
    df_sid: pd.DataFrame,
    replay_df: pd.DataFrame,
    *,
    time_col: str,
) -> list[tuple[int, int]]:
    if "label" not in df_sid.columns:
        return []
    df_sid = df_sid.copy()
    df_sid[time_col] = pd.to_datetime(df_sid[time_col], utc=True, errors="coerce")
    incident_mask = df_sid["label"].astype(str).str.startswith("incident")
    if not incident_mask.any():
        return []
    incident_rows = df_sid.loc[incident_mask]
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_label = None
    current_start = None
    current_end = None
    for ts, lbl in zip(incident_rows[time_col], incident_rows["label"].astype(str)):
        if lbl != current_label:
            if current_label is not None:
                segments.append((current_start, current_end))
            current_label = lbl
            current_start = ts
        current_end = ts
    if current_label is not None:
        segments.append((current_start, current_end))

    starts = pd.to_datetime(replay_df["window_start"], utc=True, errors="coerce")
    ends = pd.to_datetime(replay_df["window_end"], utc=True, errors="coerce")

    spans: list[tuple[int, int]] = []
    for seg_start, seg_end in segments:
        overlap = (starts <= seg_end) & (ends >= seg_start)
        idxs = np.flatnonzero(overlap.to_numpy())
        if idxs.size:
            spans.append((int(idxs.min()), int(idxs.max())))
    return spans


def _as_list(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple, set)) else []


def _first_individual_open_event(replay_df: pd.DataFrame) -> dict | None:
    for idx, row in replay_df.iterrows():
        opened = _as_list(row.get("opened_channels"))
        if opened:
            channel = opened[0]
            pending_start = idx
            for j in range(idx - 1, -1, -1):
                if channel in _as_list(replay_df.iloc[j].get("pending_channels")):
                    pending_start = j
                else:
                    break
            return {"batch": int(idx), "channel": channel, "pending_start": int(pending_start)}
    return None


def _first_group3_open(replay_df: pd.DataFrame) -> int | None:
    mask = replay_df["group_opened"].fillna(False) & replay_df["group_mode_kind"].fillna("").eq("group-3")
    idxs = np.flatnonzero(mask.to_numpy())
    return int(idxs[0]) if idxs.size else None


def _first_group3_reset(replay_df: pd.DataFrame) -> int | None:
    mask = replay_df["group_reset"].fillna(False)
    idxs = np.flatnonzero(mask.to_numpy())
    return int(idxs[0]) if idxs.size else None


def _first_individual_reset(replay_df: pd.DataFrame) -> int | None:
    for idx, row in replay_df.iterrows():
        if _as_list(row.get("reset_channels")):
            return int(idx)
    return None


def _first_all_below_zero(replay_df: pd.DataFrame) -> int | None:
    for idx, row in replay_df.iterrows():
        res = row.get("channel_max_residual")
        if not isinstance(res, dict):
            continue
        vals = [v for v in res.values() if v is not None and not pd.isna(v)]
        if vals and max(vals) < 0:
            return int(idx)
    return None


def _confirmation_region(replay_df: pd.DataFrame) -> tuple[int, int] | None:
    pending_any_mask = replay_df["pending_channels"].apply(lambda v: bool(_as_list(v)))
    pending_idxs = np.flatnonzero(pending_any_mask.to_numpy())
    if pending_idxs.size == 0:
        return None
    start = int(pending_idxs.min())

    group3_open = _first_group3_open(replay_df)
    first_open = _first_individual_open_event(replay_df)
    anchor_candidates = [i for i in (group3_open, first_open["batch"] if first_open else None) if i is not None]
    if anchor_candidates:
        end = max(anchor_candidates) + 2
    else:
        end = int(pending_idxs.max()) + 1
    end = min(end, len(replay_df) - 1)
    return (start, end)


def _draw_residuals(ax, replay_df, batch_x, *, below_zero_start):
    n = len(batch_x)
    residual_matrix = np.full((len(_CHANNELS), n), np.nan, dtype=float)
    for b, row in enumerate(replay_df.itertuples(index=False)):
        res = getattr(row, "channel_max_residual", None)
        if not isinstance(res, dict):
            continue
        for i, ch in enumerate(_CHANNELS):
            val = res.get(ch)
            if val is None:
                continue
            try:
                residual_matrix[i, b] = float(val)
            except (TypeError, ValueError):
                pass

    for i, ch in enumerate(_CHANNELS):
        color = _CHANNEL_COLORS.get(ch, "#555")
        ax.plot(batch_x, residual_matrix[i], lw=1.3, color=color, label=_CHANNEL_DISPLAY[ch])

    ax.axhline(0, color="#333", linestyle="--", linewidth=0.8, zorder=1)
    if below_zero_start is not None:
        ymin, ymax = ax.get_ylim() if ax.has_data() else (-1, 1)
        ax.axvspan(below_zero_start - 0.5, n - 0.5,
                   ymin=0, ymax=0.5, color=_BELOW_ZERO_COLOR, alpha=0.05, zorder=0)

    ax.set_ylabel("residuals\n6 channels", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", ncol=6, framealpha=0.95)
    ax.grid(True, alpha=0.2)


def _draw_individual_ribbon(ax, replay_df, batch_x, *, confirmation_region, open_event):
    n = len(batch_x)
    n_ch = len(_CHANNELS)

    for i in range(n_ch):
        ax.axhspan(n_ch - 1 - i - 0.45, n_ch - 1 - i + 0.45, color=_TRACK_BG, zorder=0)

    channel_open: dict[str, bool] = {ch: False for ch in _CHANNELS}
    for b in range(n):
        row = replay_df.iloc[b]
        opened  = set(_as_list(row.get("opened_channels")))
        reset   = set(_as_list(row.get("reset_channels")))
        realert = set(_as_list(row.get("realerted_channels")))
        pending = set(_as_list(row.get("pending_channels")))

        channel_open.update({ch: True for ch in opened})
        channel_open.update({ch: False for ch in reset})

        for i, ch in enumerate(_CHANNELS):
            y = n_ch - 1 - i
            if ch in reset:
                _cell(ax, b, y, _RESET_COLOR)
            elif ch in realert:
                _cell(ax, b, y, _REALERT_COLOR)
            elif ch in opened:
                _cell(ax, b, y, _OPEN_COLOR)
            elif channel_open[ch]:
                _cell(ax, b, y, _OPEN_COLOR, alpha=0.30)
            elif ch in pending:
                _cell(ax, b, y, _PENDING_COLOR)

    if confirmation_region is not None:
        _draw_batch_guides(ax, confirmation_region, y_top=n_ch - 0.5, y_bot=-0.5)

    if open_event is not None:
        ch = open_event["channel"]
        try:
            ch_idx = _CHANNELS.index(ch)
        except ValueError:
            ch_idx = None
        if ch_idx is not None:
            bracket_lo = open_event["pending_start"] - 0.5
            bracket_hi = open_event["batch"] + 0.5
            y_top = n_ch - 0.42
            ax.plot(
                [bracket_lo, bracket_lo, bracket_hi, bracket_hi],
                [y_top, y_top + 0.35, y_top + 0.35, y_top],
                color="#c06c00", lw=1.3, clip_on=False,
            )
            ax.text(
                (bracket_lo + bracket_hi) / 2, y_top + 0.6,
                f"3-of-4 confirmed on {_CHANNEL_DISPLAY[ch]} → OPEN",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#c06c00",
                clip_on=False,
            )

    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([_CHANNEL_DISPLAY[ch] for ch in reversed(_CHANNELS)], fontsize=9)
    ax.set_ylim(-0.6, n_ch + 0.4)
    ax.set_ylabel("individual tier\nper-channel", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_group3_ribbon(ax, replay_df, batch_x, *, confirmation_region):
    n = len(batch_x)
    ax.axhspan(-0.45, 0.45, color=_TRACK_BG, zorder=0)

    for b in range(n):
        row = replay_df.iloc[b]
        mode_kind = str(row.get("group_mode_kind") or "")
        is_active = bool(row.get("group_mode_active"))
        promoting = str(row.get("promotion_candidate_kind") or "")
        group_reset = bool(row.get("group_reset"))
        group_opened = bool(row.get("group_opened"))

        if group_reset and mode_kind == "group-3":
            _cell(ax, b, 0, _RESET_COLOR, height=0.9)
        elif mode_kind == "group-3" and (is_active or group_opened):
            _cell(ax, b, 0, _GROUP3_ACTIVE_COLOR, height=0.9)
        elif promoting == "group-3":
            _cell(ax, b, 0, _GROUP3_FORMING_COLOR, height=0.9)

    if confirmation_region is not None:
        _draw_batch_guides(ax, confirmation_region, y_top=0.5, y_bot=-0.5)

    ax.set_yticks([0])
    ax.set_yticklabels(["group-3"], fontsize=9)
    ax.set_ylim(-0.55, 0.55)
    ax.set_ylabel("group-3 tier", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def _draw_group6_ribbon(ax, batch_x, *, alert_params):
    n = len(batch_x)
    ax.add_patch(mpatches.Rectangle(
        (-0.5, -0.45), n, 0.9,
        facecolor=_TRACK_BG, edgecolor="#ddd", linestyle=(0, (4, 3)), linewidth=1.0, zorder=0,
    ))
    label = (
        f"no group-6 events — enable_group6_alerts: {alert_params.enable_group6_alerts}"
        if not alert_params.enable_group6_alerts
        else "group-6 enabled — events would render here"
    )
    ax.text(n / 2, 0, label, ha="center", va="center",
            fontsize=9, style="italic", color="#999")
    ax.set_yticks([0])
    ax.set_yticklabels(["group-6"], fontsize=9)
    ax.set_ylim(-0.55, 0.55)
    ax.set_ylabel("group-6 tier\ndisabled", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def _cell(ax, batch_index, y, color, *, alpha=1.0, height=0.8):
    rect = mpatches.Rectangle(
        (batch_index - 0.5, y - height / 2),
        1.0, height,
        facecolor=color, alpha=alpha, edgecolor="none", zorder=2,
    )
    ax.add_patch(rect)


def _draw_batch_guides(ax, region, *, y_top, y_bot):
    lo, hi = region
    for b in range(lo, hi + 1):
        ax.plot(
            [b, b], [y_bot, y_top],
            color="#bbb", linestyle=(0, (2, 2)), linewidth=0.5, zorder=1,
        )


def _add_numbered_annotations(
    fig,
    *,
    replay_df,
    confirmation_region,
    open_event,
    group3_open_batch,
    group3_reset_batch,
    individual_reset_batch,
    ax_top,
):
    lines = []
    if confirmation_region is not None:
        lines.append((confirmation_region[0],
                      "pending_state buffer accumulates (rolling 4-batch window, orange = active slot)",
                      "#c06c00"))
    if open_event is not None:
        ch_display = _CHANNEL_DISPLAY.get(open_event["channel"], open_event["channel"])
        lines.append((open_event["batch"],
                      f"3-of-4 confirmed on {ch_display} → individual OPEN (green)",
                      "#1a7a1a"))
    if group3_open_batch is not None:
        lines.append((group3_open_batch,
                      "2-of-3 confirmed → group-3 takes over, individuals suppressed",
                      "#4a3f6b"))
    if group3_reset_batch is not None:
        lines.append((group3_reset_batch,
                      "Strict majority < 0 for 12 batches → group-3 RESET (grey)",
                      "#555"))
    if individual_reset_batch is not None:
        lines.append((individual_reset_batch,
                      "18 consecutive batches < 0 on a channel → individual RESET (grey)",
                      "#555"))

    if not lines:
        return

    fig_coord = fig.transFigure.inverted()
    for i, (batch, text, color) in enumerate(lines):
        data_pt = ax_top.transData.transform((batch, ax_top.get_ylim()[1]))
        fig_x, _ = fig_coord.transform(data_pt)
        y_text = 0.215 - 0.025 * i
        fig.add_artist(
            plt.Line2D(
                [fig_x, fig_x], [y_text + 0.005, 0.35],
                transform=fig.transFigure,
                color=color, linestyle=(0, (3, 3)), linewidth=0.9, clip_on=False,
            )
        )
        fig.add_artist(
            mpatches.Circle(
                (fig_x, y_text + 0.005),
                radius=0.004,
                transform=fig.transFigure,
                facecolor=color, edgecolor="none", clip_on=False,
            )
        )
        fig.text(
            0.12, y_text,
            f"({i + 1})  {text}",
            fontsize=10.5, fontweight="bold", color=color,
            transform=fig.transFigure,
        )

    if group3_reset_batch is not None and individual_reset_batch is not None:
        fig.text(
            0.12, 0.215 - 0.025 * len(lines),
            "Group-3 resets first because its threshold is shorter (12 vs 18 batches).",
            fontsize=9.5, style="italic", color="#666",
            transform=fig.transFigure,
        )


def _add_legend(fig):
    handles = [
        mpatches.Patch(facecolor=_PENDING_COLOR, label="pending"),
        mpatches.Patch(facecolor=_OPEN_COLOR, label="individual open"),
        mpatches.Patch(facecolor=_REALERT_COLOR, label="re-alert"),
        mpatches.Patch(facecolor=_GROUP3_FORMING_COLOR, label="group-3 forming"),
        mpatches.Patch(facecolor=_GROUP3_ACTIVE_COLOR, label="group-3 active"),
        mpatches.Patch(facecolor=_RESET_COLOR, label="reset"),
        mpatches.Patch(facecolor=_INCIDENT_COLOR, alpha=_INCIDENT_ALPHA, label="incident window"),
        mpatches.Patch(facecolor=_PENDING_EMPTY_COLOR, label="inactive slot in buffer"),
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=4, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
