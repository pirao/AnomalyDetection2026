"""Group-level residual distribution widget for notebook 01.

Compares fit / pred-normal / per-scenario pred-incident residual
distributions within a single scenario group. Lets the user tune
thresholds and x-ranges while seeing where incident mass sits relative
to the healthy fit baseline. Consumer: ``notebooks/01_eda.ipynb``.
"""

from __future__ import annotations

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output

from ._helpers import (
    _D_ACCEL_COLOR,
    _D_VEL_COLOR,
    _REF_EDGE,
    _REF_FILL,
    _THRESH_COLOR,
    _safe_series,
)


def create_group_distribution_widget(
    full_scored_df,
    *,
    vel_cols,
    accel_cols,
    scenario_col="scenario_id",
    split_col="split",
    label_col="label",
    uptime_col="uptime",
    group_col="scenario_group",
    normal_label="normal",
    default_group="group_1",
    default_num_bins=80,
    default_vel_xlim=(-2.0, 12.0),
    default_accel_xlim=(-2.0, 12.0),
    threshold_vel=3.0,
    threshold_accel=3.0,
):
    """Distribution widget comparing fit / pred-normal / pred-incident residuals for one group.

    Requires ``add_scenario_group_labels`` to have been called on ``full_scored_df``
    so that a ``scenario_group`` column is present.

    Returns a single ``ipywidgets.Widget`` (HBox: plot area left, controls right).
    """
    if not vel_cols:
        raise ValueError("vel_cols is empty.")
    if not accel_cols:
        raise ValueError("accel_cols is empty.")
    if group_col not in full_scored_df.columns:
        raise ValueError(
            f"Column '{group_col}' not found. Call add_scenario_group_labels() first."
        )

    data = full_scored_df.copy()
    data[uptime_col] = data[uptime_col].fillna(False).astype(bool)

    d_vel_cols = [f"d_{c}" for c in vel_cols]
    d_accel_cols = [f"d_{c}" for c in accel_cols]

    available_groups = sorted(data[group_col].dropna().unique().tolist())
    initial_group = default_group if default_group in available_groups else available_groups[0]

    def _scenarios_for_group(group: str) -> list:
        return sorted(
            data.loc[data[group_col] == group, scenario_col].dropna().unique().tolist()
        )

    initial_scenarios = _scenarios_for_group(initial_group)

    group_dropdown = widgets.Dropdown(
        options=available_groups,
        value=initial_group,
        description="Group",
        style={"description_width": "55px"},
        layout=widgets.Layout(width="220px"),
    )
    scenario_select = widgets.SelectMultiple(
        options=initial_scenarios,
        value=tuple(initial_scenarios),
        description="",
        layout=widgets.Layout(width="220px", height="160px"),
    )
    vel_xlim_slider = widgets.FloatRangeSlider(
        value=list(default_vel_xlim), min=-4.0, max=30.0, step=0.5,
        description="vel x",
        continuous_update=False,
        readout_format=".1f",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="220px"),
    )
    accel_xlim_slider = widgets.FloatRangeSlider(
        value=list(default_accel_xlim), min=-4.0, max=30.0, step=0.5,
        description="acc x",
        continuous_update=False,
        readout_format=".1f",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="220px"),
    )
    bins_slider = widgets.IntSlider(
        value=int(default_num_bins), min=10, max=200, step=5,
        description="bins",
        continuous_update=False,
        style={"description_width": "50px"},
        layout=widgets.Layout(width="220px"),
    )
    threshold_vel_slider = widgets.FloatSlider(
        value=float(threshold_vel), min=0.0, max=15.0, step=0.1,
        description="vel thr",
        continuous_update=False,
        readout_format=".2f",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="220px"),
    )
    threshold_accel_slider = widgets.FloatSlider(
        value=float(threshold_accel), min=0.0, max=15.0, step=0.1,
        description="acc thr",
        continuous_update=False,
        readout_format=".2f",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="220px"),
    )
    plot_btn = widgets.Button(
        description="Plot",
        button_style="primary",
        layout=widgets.Layout(width="220px"),
    )

    def _on_group_change(change):
        new_scenarios = _scenarios_for_group(change["new"])
        scenario_select.options = new_scenarios
        scenario_select.value = tuple(new_scenarios)

    group_dropdown.observe(_on_group_change, names="value")

    control_panel = widgets.VBox(
        [
            widgets.HTML("<b>Group</b>"),
            group_dropdown,
            widgets.HTML("<b>Scenarios</b>"),
            scenario_select,
            widgets.HTML("<b>Vel x-range</b>"),
            vel_xlim_slider,
            widgets.HTML("<b>Accel x-range</b>"),
            accel_xlim_slider,
            widgets.HTML("<b>Bins</b>"),
            bins_slider,
            widgets.HTML("<b>Thresholds</b>"),
            threshold_vel_slider,
            threshold_accel_slider,
            plot_btn,
        ],
        layout=widgets.Layout(
            width="240px",
            flex_shrink="0",
            padding="10px",
            gap="4px",
        ),
    )

    output = widgets.Output(
        layout=widgets.Layout(flex="1 1 auto", min_width="800px", padding="10px")
    )

    _TAB10 = plt.cm.tab10.colors

    def _render(_btn=None):
        group = group_dropdown.value
        selected_scenarios = list(scenario_select.value)
        vel_xlim = vel_xlim_slider.value
        accel_xlim = accel_xlim_slider.value
        num_bins = bins_slider.value
        thr_vel = threshold_vel_slider.value
        thr_accel = threshold_accel_slider.value

        group_df = data[data[group_col] == group].copy()
        active_df = group_df[group_df[uptime_col]].copy()

        fit_df = active_df[active_df[split_col] == "fit"] if split_col in active_df.columns else active_df.iloc[0:0]
        pred_df = active_df[active_df[split_col] == "pred"] if split_col in active_df.columns else active_df.iloc[0:0]

        if label_col in pred_df.columns:
            norm_df = pred_df[pred_df[label_col].astype(str) == str(normal_label)]
        else:
            norm_df = pred_df

        scenario_inc: list[tuple[object, pd.DataFrame, tuple]] = []
        for i, sid in enumerate(selected_scenarios):
            sid_pred = pred_df[pred_df[scenario_col] == sid] if scenario_col in pred_df.columns else pred_df
            if label_col in sid_pred.columns:
                sid_inc = sid_pred[sid_pred[label_col].astype(str).str.startswith("incident_")]
            else:
                sid_inc = sid_pred.iloc[0:0]
            color = _TAB10[i % len(_TAB10)]
            scenario_inc.append((sid, sid_inc, color))

        n_rows = max(len(vel_cols), len(accel_cols))
        fig, axes = plt.subplots(
            n_rows, 2,
            figsize=(12, 3 * n_rows),
            constrained_layout=True,
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        col_pairs = list(zip(d_vel_cols, d_accel_cols))

        for row_idx, (d_vel, d_accel) in enumerate(col_pairs):
            for col_idx, (d_col, xlim, thr, modality_label) in enumerate([
                (d_vel, vel_xlim, thr_vel, "vel"),
                (d_accel, accel_xlim, thr_accel, "accel"),
            ]):
                ax = axes[row_idx, col_idx]
                bin_range = (float(xlim[0]), float(xlim[1]))

                fit_vals = _safe_series(fit_df, d_col).dropna().to_numpy(dtype=float)
                if fit_vals.size > 0:
                    ax.hist(
                        fit_vals, bins=num_bins, range=bin_range, density=True,
                        color=_REF_FILL, edgecolor=_REF_EDGE, alpha=0.6,
                        label="fit baseline",
                    )

                norm_vals = _safe_series(norm_df, d_col).dropna().to_numpy(dtype=float)
                if norm_vals.size > 0:
                    ax.hist(
                        norm_vals, bins=num_bins, range=bin_range, density=True,
                        histtype="step",
                        color=_D_VEL_COLOR if modality_label == "vel" else _D_ACCEL_COLOR,
                        linewidth=1.5, alpha=0.85,
                        label="pred-normal (pooled)",
                    )

                for sid, sid_inc_df, color in scenario_inc:
                    inc_vals = _safe_series(sid_inc_df, d_col).dropna().to_numpy(dtype=float)
                    if inc_vals.size > 0:
                        ax.hist(
                            inc_vals, bins=num_bins, range=bin_range, density=True,
                            histtype="stepfilled", color=color, alpha=0.55,
                            label=f"incident s{sid} (n={inc_vals.size})",
                        )

                ax.axvline(thr, color=_THRESH_COLOR, linestyle="--", linewidth=1.2, label=f"threshold={thr:.1f}")
                ax.set_xlim(bin_range)
                ax.set_xlabel("residual (d_norm)")
                ax.set_ylabel("density")
                ax.set_title(d_col)

                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=11, loc="upper right")

        with output:
            clear_output(wait=True)
            plt.show()

    plot_btn.on_click(_render)
    _render()

    return widgets.HBox(
        [output, control_panel],
        layout=widgets.Layout(width="100%"),
    )
