"""Notebook-01 calibration widget for pooled-residual sigmoid tuning.

Builds histograms of per-sensor residuals near labeled windows and an
overlayed sigmoid curve so the user can solve for alpha/beta from two
shared anchor points. The widget only calibrates — it does not replay the
pipeline. Consumer: ``notebooks/01_eda.ipynb``.
"""

from __future__ import annotations

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output

from sample_processing.model.scenario_groups import GROUP_DEFINITIONS

from ._helpers import (
    _CURR_FILL_ACCEL,
    _CURR_FILL_VEL,
    _REF_EDGE,
    _REF_FILL,
    _SIGMOID_COLOR,
    _THRESH_COLOR,
    _build_mask,
    _safe_series,
)
from ._sigmoid_math import (
    _recompute_dnorm_frame,
    _sigmoid,
    _solve_alpha_beta_from_two_anchors,
)


def create_sigmoid_global_residual_widget(
    full_scored_df,
    *,
    vel_cols,
    accel_cols,
    scenario_col="scenario_id",
    time_col="sampled_at",
    uptime_col="uptime",
    split_col="split",
    label_col="label",
    normal_label="normal",
    default_grace_hours=2.0,
    default_vel_xlim=10.0,
    default_accel_xlim=10.0,
    default_num_bins=100,
    threshold_vel=3.0,
    threshold_accel=3.0,
    alpha_vel=1.52,
    alpha_accel=1.52,
    beta_vel=2.45,
    beta_accel=2.45,
    default_anchor1_residual_vel=0.5,
    default_anchor1_score_vel=0.10,
    default_anchor2_residual_vel=5.0,
    default_anchor2_score_vel=0.90,
    default_anchor1_residual_acc=0.5,
    default_anchor1_score_acc=0.10,
    default_anchor2_residual_acc=5.0,
    default_anchor2_score_acc=0.90,
):
    """Interactive residual-only sigmoid calibration widget pooled by modality."""
    if not vel_cols:
        raise ValueError("vel_cols is empty.")
    if not accel_cols:
        raise ValueError("accel_cols is empty.")
    if int(default_num_bins) <= 0:
        raise ValueError("default_num_bins must be a positive integer.")

    bins = int(default_num_bins)
    data = full_scored_df.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col]).copy()
    scenario_options = sorted(data[scenario_col].dropna().unique().tolist())
    scored_frame = _recompute_dnorm_frame(
        data,
        scenario_col=scenario_col,
        split_col=split_col,
        uptime_col=uptime_col,
        vel_cols=list(vel_cols),
        accel_cols=list(accel_cols),
    )
    group_options = [("All scenarios", "all")] + [
        (str(group_info["label"]), group_key)
        for group_key, group_info in GROUP_DEFINITIONS.items()
    ]

    def _collect_ref(modality: str, d_cols: list[str]) -> np.ndarray:
        out = scored_frame.copy()
        global_mask_col = "global_mask_vel" if modality == "vel" else "global_mask_accel"
        mask = _build_mask(
            out,
            uptime_col=uptime_col,
            uptime_only=True,
            global_mask_col=global_mask_col,
        )
        out = out[mask].copy()
        if split_col in out.columns and label_col in out.columns:
            out = out[(out[split_col] == "fit") & (out[label_col].astype(str) == str(normal_label))].copy()
        arrays: list[np.ndarray] = []
        for d_col in d_cols:
            vals = _safe_series(out, d_col).dropna().to_numpy(dtype=float)
            if vals.size > 0:
                arrays.append(vals)
        return np.concatenate(arrays) if arrays else np.array([], dtype=float)

    def _collect_by_scenario(*, scenarios: list, modality: str, d_cols: list[str], threshold: float) -> dict[object, np.ndarray]:
        if label_col not in scored_frame.columns:
            return {}
        grace = pd.Timedelta(hours=float(default_grace_hours))
        out: dict[object, np.ndarray] = {}
        for sid in scenarios:
            df_sid = scored_frame[scored_frame[scenario_col] == sid].sort_values(time_col).copy()
            if df_sid.empty:
                continue
            global_mask_col = "global_mask_vel" if modality == "vel" else "global_mask_accel"
            mask = _build_mask(
                df_sid,
                uptime_col=uptime_col,
                uptime_only=True,
                global_mask_col=global_mask_col,
            )
            df_sid = df_sid[mask].copy()
            if df_sid.empty:
                continue
            times = pd.to_datetime(df_sid[time_col], errors="coerce")
            label_flags = df_sid[label_col].astype(str) != str(normal_label)
            if not label_flags.any():
                continue
            keep = pd.Series(False, index=df_sid.index)
            for ts in times[label_flags.fillna(False)]:
                if pd.isna(ts):
                    continue
                keep |= (times >= ts - grace) & (times <= ts + grace)
            arrays: list[np.ndarray] = []
            for d_col in d_cols:
                vals = _safe_series(df_sid.loc[keep], d_col).dropna().to_numpy(dtype=float)
                if vals.size > 0:
                    arrays.append(vals)
            if arrays:
                out[sid] = np.concatenate(arrays) - float(threshold)
        return out

    shared_anchor1_residual = float(0.5 * (default_anchor1_residual_vel + default_anchor1_residual_acc))
    shared_anchor1_score = float(0.5 * (default_anchor1_score_vel + default_anchor1_score_acc))
    shared_anchor2_residual = float(0.5 * (default_anchor2_residual_vel + default_anchor2_residual_acc))
    shared_anchor2_score = float(0.5 * (default_anchor2_score_vel + default_anchor2_score_acc))

    scenario_select = widgets.SelectMultiple(
        options=scenario_options,
        value=tuple(scenario_options),
        rows=min(12, max(6, len(scenario_options))),
        description="",
        layout=widgets.Layout(width="220px", height="320px"),
    )
    group_dropdown = widgets.Dropdown(
        options=group_options,
        value="all",
        description="Group",
        style={"description_width": "55px"},
        layout=widgets.Layout(width="360px"),
    )
    vel_xlim_slider = widgets.FloatRangeSlider(
        value=(-4.0, float(default_vel_xlim)), min=-20.0, max=100.0, step=0.5,
        description="vel resid", continuous_update=False,
        readout_format=".1f", style={"description_width": "70px"}, layout=widgets.Layout(width="340px"),
    )
    accel_xlim_slider = widgets.FloatRangeSlider(
        value=(-4.0, float(default_accel_xlim)), min=-20.0, max=100.0, step=0.5,
        description="acc resid", continuous_update=False,
        readout_format=".1f", style={"description_width": "70px"}, layout=widgets.Layout(width="340px"),
    )
    threshold_vel_slider = widgets.FloatSlider(
        value=float(threshold_vel), min=0.0, max=12.0, step=0.05,
        description="vel thr", continuous_update=False, readout_format=".2f",
        style={"description_width": "60px"}, layout=widgets.Layout(width="250px"),
    )
    threshold_accel_slider = widgets.FloatSlider(
        value=float(threshold_accel), min=0.0, max=12.0, step=0.05,
        description="acc thr", continuous_update=False, readout_format=".2f",
        style={"description_width": "60px"}, layout=widgets.Layout(width="250px"),
    )
    alpha_vel_slider = widgets.FloatSlider(
        value=float(alpha_vel), min=0.05, max=5.0, step=0.01,
        description="vel alpha", continuous_update=False, readout_format=".2f",
        style={"description_width": "70px"}, layout=widgets.Layout(width="250px"),
    )
    alpha_accel_slider = widgets.FloatSlider(
        value=float(alpha_accel), min=0.05, max=5.0, step=0.01,
        description="acc alpha", continuous_update=False, readout_format=".2f",
        style={"description_width": "70px"}, layout=widgets.Layout(width="250px"),
    )
    beta_vel_slider = widgets.FloatSlider(
        value=float(beta_vel), min=-2.0, max=12.0, step=0.05,
        description="vel beta", continuous_update=False, readout_format=".2f",
        style={"description_width": "70px"}, layout=widgets.Layout(width="250px"),
    )
    beta_accel_slider = widgets.FloatSlider(
        value=float(beta_accel), min=-2.0, max=12.0, step=0.05,
        description="acc beta", continuous_update=False, readout_format=".2f",
        style={"description_width": "70px"}, layout=widgets.Layout(width="250px"),
    )
    anchor1_residual = widgets.FloatText(
        value=shared_anchor1_residual, description="anchor 1 resid", step=0.1,
        layout=widgets.Layout(width="210px"), style={"description_width": "110px"},
    )
    anchor1_score = widgets.FloatText(
        value=shared_anchor1_score, description="anchor 1 score", step=0.01,
        layout=widgets.Layout(width="210px"), style={"description_width": "110px"},
    )
    anchor2_residual = widgets.FloatText(
        value=shared_anchor2_residual, description="anchor 2 resid", step=0.01,
        layout=widgets.Layout(width="210px"), style={"description_width": "110px"},
    )
    anchor2_score = widgets.FloatText(
        value=shared_anchor2_score, description="anchor 2 score", step=0.01,
        layout=widgets.Layout(width="210px"), style={"description_width": "110px"},
    )
    solve_btn = widgets.Button(description="Solve alpha,beta", button_style="info", layout=widgets.Layout(width="160px"))
    plot_btn = widgets.Button(description="Plot", button_style="primary", layout=widgets.Layout(width="110px"))
    info_html = widgets.HTML(layout=widgets.Layout(width="100%", margin="4px 0 8px 0"))
    out = widgets.Output(layout=widgets.Layout(width="100%", padding="8px"))

    def _draw_panel(ax, *, ref_residuals, scenario_residuals, alpha, beta, title, color, xlim_values, anchors):
        selected_arrays = [v for v in scenario_residuals.values() if v.size > 0]
        selected = np.concatenate(selected_arrays) if selected_arrays else np.array([], dtype=float)
        ref_residuals = np.asarray(ref_residuals, dtype=float)
        ref_residuals = ref_residuals[np.isfinite(ref_residuals)]
        selected = selected[np.isfinite(selected)]

        lo, hi = (float(xlim_values[0]), float(xlim_values[1]))
        if hi <= lo:
            hi = lo + 1.0
        bin_edges = np.linspace(lo, hi, bins + 1)
        y_candidates: list[float] = []

        if ref_residuals.size > 0:
            ref_counts, _, _ = ax.hist(
                ref_residuals, bins=bin_edges, density=True,
                color=_REF_FILL, edgecolor=_REF_EDGE, alpha=0.55, label="fit healthy",
            )
            ref_ymax = float(np.nanmax(ref_counts)) if len(ref_counts) else 0.0
            y_candidates.append(ref_ymax if np.isfinite(ref_ymax) else 0.0)
        if selected.size > 0:
            selected_counts, _, _ = ax.hist(
                selected, bins=bin_edges, density=True,
                color=color, alpha=0.35, label="selected scenarios",
            )
            selected_ymax = float(np.nanmax(selected_counts)) if len(selected_counts) else 0.0
            y_candidates.append(selected_ymax if np.isfinite(selected_ymax) else 0.0)

        ax.axvline(0.0, color=_THRESH_COLOR, lw=1.4, linestyle="--", label="residual=0 (d_norm threshold)")
        ax2 = ax.twinx()
        xs = np.linspace(lo, hi, 400)
        ax2.plot(xs, _sigmoid(xs, alpha=alpha, beta=beta), color=_SIGMOID_COLOR, lw=2.0, label="sigmoid")
        anchor_colors = ["#7f7f7f", "#9467bd"]
        for idx, (anchor_r, anchor_p) in enumerate(anchors, start=1):
            if np.isfinite(anchor_r) and np.isfinite(anchor_p):
                anchor_color = anchor_colors[(idx - 1) % len(anchor_colors)]
                anchor_curve_score = float(_sigmoid(np.asarray([anchor_r], dtype=float), alpha=alpha, beta=beta)[0])
                ax.axvline(anchor_r, color=anchor_color, lw=1.0, linestyle="--", alpha=0.8, label=f"anchor {idx}")
                ax2.axhline(anchor_p, color=anchor_color, lw=1.0, linestyle="--", alpha=0.6, label=f"anchor {idx} target")
                ax2.scatter([anchor_r], [anchor_curve_score], color=anchor_color, s=34, zorder=6)

        ax.set_title(title)
        ax.set_xlabel("residual (d_norm - threshold)")
        ax.set_ylabel("density")
        ax.set_xlim(lo, hi)
        ymax = max((y for y in y_candidates if np.isfinite(y)), default=0.0)
        ax.set_ylim(0.0, max(ymax * 1.12, 0.1))
        ax2.set_ylabel("sigmoid score", color=_SIGMOID_COLOR)
        ax2.tick_params(axis="y", colors=_SIGMOID_COLOR)
        ax2.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3)

        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        seen = set()
        lines, labels = [], []
        for line, label in list(zip(l1, lab1)) + list(zip(l2, lab2)):
            if label not in seen:
                seen.add(label)
                lines.append(line)
                labels.append(label)
        ax.legend(lines, labels, loc="upper left")

    def _sync_group_selection(change=None):
        if change is not None and change.get("name") != "value":
            return
        group_key = group_dropdown.value
        if group_key == "all":
            scenario_select.value = tuple(scenario_options)
            return
        scenario_ids = tuple(sid for sid in GROUP_DEFINITIONS[group_key]["scenario_ids"] if sid in scenario_options)
        scenario_select.value = scenario_ids

    def _solve_alpha_beta(_=None):
        solved_alpha_vel, solved_beta_vel = _solve_alpha_beta_from_two_anchors(
            residual_1=float(anchor1_residual.value),
            score_1=float(anchor1_score.value),
            residual_2=float(anchor2_residual.value),
            score_2=float(anchor2_score.value),
            fallback_alpha=float(alpha_vel_slider.value),
            fallback_beta=float(beta_vel_slider.value),
        )
        solved_alpha_acc, solved_beta_acc = _solve_alpha_beta_from_two_anchors(
            residual_1=float(anchor1_residual.value),
            score_1=float(anchor1_score.value),
            residual_2=float(anchor2_residual.value),
            score_2=float(anchor2_score.value),
            fallback_alpha=float(alpha_accel_slider.value),
            fallback_beta=float(beta_accel_slider.value),
        )
        alpha_vel_slider.value = float(max(alpha_vel_slider.min, min(alpha_vel_slider.max, solved_alpha_vel)))
        beta_vel_slider.value = float(max(beta_vel_slider.min, min(beta_vel_slider.max, solved_beta_vel)))
        alpha_accel_slider.value = float(max(alpha_accel_slider.min, min(alpha_accel_slider.max, solved_alpha_acc)))
        beta_accel_slider.value = float(max(beta_accel_slider.min, min(beta_accel_slider.max, solved_beta_acc)))
        _render()

    def _render(_=None):
        with out:
            clear_output(wait=True)
            selected = list(scenario_select.value)
            if not selected:
                print("Select at least one scenario.")
                return

            vel_d_cols = [f"d_{col}" for col in vel_cols]
            accel_d_cols = [f"d_{col}" for col in accel_cols]
            ref_vel = _collect_ref("vel", vel_d_cols) - float(threshold_vel_slider.value)
            ref_acc = _collect_ref("acc", accel_d_cols) - float(threshold_accel_slider.value)
            by_scenario_vel = _collect_by_scenario(
                scenarios=selected, modality="vel", d_cols=vel_d_cols,
                threshold=float(threshold_vel_slider.value),
            )
            by_scenario_acc = _collect_by_scenario(
                scenarios=selected, modality="acc", d_cols=accel_d_cols,
                threshold=float(threshold_accel_slider.value),
            )

            fig, (ax_vel, ax_acc) = plt.subplots(2, 1, figsize=(18, 12.5), constrained_layout=True)
            fig.suptitle(f"Global residual calibration near labels (+/-{default_grace_hours:.1f}h) [standard]")
            _draw_panel(
                ax_vel,
                ref_residuals=ref_vel,
                scenario_residuals=by_scenario_vel,
                alpha=float(alpha_vel_slider.value),
                beta=float(beta_vel_slider.value),
                title="Velocity residuals near labels",
                color=_CURR_FILL_VEL,
                xlim_values=vel_xlim_slider.value,
                anchors=[
                    (float(anchor1_residual.value), float(anchor1_score.value)),
                    (float(anchor2_residual.value), float(anchor2_score.value)),
                ],
            )
            _draw_panel(
                ax_acc,
                ref_residuals=ref_acc,
                scenario_residuals=by_scenario_acc,
                alpha=float(alpha_accel_slider.value),
                beta=float(beta_accel_slider.value),
                title="Acceleration residuals near labels",
                color=_CURR_FILL_ACCEL,
                xlim_values=accel_xlim_slider.value,
                anchors=[
                    (float(anchor1_residual.value), float(anchor1_score.value)),
                    (float(anchor2_residual.value), float(anchor2_score.value)),
                ],
            )
            plt.show()
            plt.close(fig)

            info_html.value = (
                "<b>Residual definition:</b> residual = d_norm - threshold &nbsp;|&nbsp; "
                "<b>Sigmoid:</b> score = 1 / (1 + exp(-alpha * (residual - beta)))<br>"
                "<b>Solve rule:</b> fit alpha and beta from two shared anchor points in resid space.<br>"
                f"<b>Selected scenarios:</b> {selected} &nbsp;|&nbsp; "
                f"<b>Group:</b> {dict(group_options).get(group_dropdown.value, group_dropdown.value)} &nbsp;|&nbsp; "
                f"<b>Scaler:</b> standard &nbsp;|&nbsp; "
                f"<b>Grace:</b> +/-{default_grace_hours:.1f}h &nbsp;|&nbsp; "
                f"<b>Bins:</b> {bins} &nbsp;|&nbsp; "
                f"<b>Vel:</b> d_norm threshold={threshold_vel_slider.value:.2f}, alpha={alpha_vel_slider.value:.2f}, beta={beta_vel_slider.value:.2f} &nbsp;|&nbsp; "
                f"<b>Accel:</b> d_norm threshold={threshold_accel_slider.value:.2f}, alpha={alpha_accel_slider.value:.2f}, beta={beta_accel_slider.value:.2f}"
            )

    for w in (
        group_dropdown, scenario_select, vel_xlim_slider, accel_xlim_slider,
        threshold_vel_slider, threshold_accel_slider, alpha_vel_slider, alpha_accel_slider,
        beta_vel_slider, beta_accel_slider, anchor1_residual, anchor1_score,
        anchor2_residual, anchor2_score,
    ):
        w.observe(lambda c: _render() if c["name"] == "value" else None)
    plot_btn.on_click(_render)
    solve_btn.on_click(_solve_alpha_beta)
    group_dropdown.observe(_sync_group_selection, names="value")

    main_block_layout = widgets.Layout(border="1px solid #d9d9d9", padding="10px 12px", margin="0 0 10px 0", width="100%")
    compact_row = widgets.Layout(width="100%", gap="12px", align_items="center", flex_flow="row wrap")
    panel_layout = widgets.Layout(border="1px solid #e3e3e3", padding="10px", margin="0")
    row_layout = widgets.Layout(width="100%", gap="14px", align_items="flex-start")

    scenario_panel = widgets.VBox(
        [widgets.HTML("<b>Scenarios</b>"), scenario_select],
        layout=widgets.Layout(width="230px", min_width="230px"),
    )
    options_panel = widgets.VBox(
        [widgets.HTML("<b>Options</b>"), group_dropdown, widgets.HTML("<b>Plot ranges</b>"), vel_xlim_slider, accel_xlim_slider],
        layout=widgets.Layout(width="420px", min_width="420px"),
    )
    actions_panel = widgets.VBox(
        [
            widgets.HTML("<b>Actions</b>"),
            widgets.HBox([solve_btn, plot_btn], layout=compact_row),
            widgets.HTML("<span style='color:#555'>Use the shared anchors below to solve alpha and beta for both modalities.</span>"),
        ],
        layout=widgets.Layout(width="220px", min_width="220px"),
    )
    selection_row = widgets.HBox(
        [
            widgets.VBox([scenario_panel], layout=panel_layout),
            widgets.VBox([options_panel], layout=panel_layout),
            widgets.VBox([actions_panel], layout=panel_layout),
        ],
        layout=row_layout,
    )

    anchor_panel = widgets.VBox(
        [
            widgets.HTML("<b>Shared anchors</b>"),
            widgets.HBox([anchor1_residual, anchor1_score], layout=compact_row),
            widgets.HBox([anchor2_residual, anchor2_score], layout=compact_row),
        ],
        layout=widgets.Layout(width="420px", min_width="420px"),
    )
    velocity_panel = widgets.VBox(
        [widgets.HTML("<b>Velocity</b>"), widgets.HBox([threshold_vel_slider, alpha_vel_slider, beta_vel_slider], layout=compact_row)],
        layout=widgets.Layout(width="100%", min_width="360px"),
    )
    accel_panel = widgets.VBox(
        [widgets.HTML("<b>Acceleration</b>"), widgets.HBox([threshold_accel_slider, alpha_accel_slider, beta_accel_slider], layout=compact_row)],
        layout=widgets.Layout(width="100%", min_width="360px"),
    )
    params_panel = widgets.VBox([velocity_panel, accel_panel], layout=widgets.Layout(width="100%"))
    calibration_row = widgets.HBox(
        [widgets.VBox([anchor_panel], layout=panel_layout), widgets.VBox([params_panel], layout=panel_layout)],
        layout=row_layout,
    )

    ui = widgets.VBox(
        [
            widgets.VBox([selection_row, calibration_row, info_html], layout=main_block_layout),
            out,
        ],
        layout=widgets.Layout(width="100%"),
    )
    _sync_group_selection()
    _render()

    def get_params():
        return {
            "threshold_vel": threshold_vel_slider.value,
            "threshold_accel": threshold_accel_slider.value,
            "alpha_vel": alpha_vel_slider.value,
            "alpha_accel": alpha_accel_slider.value,
            "beta_vel": beta_vel_slider.value,
            "beta_accel": beta_accel_slider.value,
        }

    return ui, get_params
