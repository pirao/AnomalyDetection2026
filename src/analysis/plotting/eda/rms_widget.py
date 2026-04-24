"""Per-scenario raw RMS channel inspector.

Pipeline stage: EDA widget. Renders the six RMS channels for one or more
selected scenarios with split boundaries and incident overlays.

Consumer: ``notebooks/01_eda.ipynb`` (``create_rms_scenario_widget``).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ipywidgets as widgets
from IPython.display import clear_output

_LINE_COLOR     = "#1f77b4"
_INCIDENT_COLOR = "#E07B00"
_INCIDENT_ALPHA = 0.15
_SPLIT_COLOR    = "black"
_FIT_VALUE      = "fit"
_PRED_VALUE     = "pred"


def create_rms_scenario_widget(
    fit_df: pd.DataFrame,
    scenario_col: str = "scenario_id",
    time_col: str = "sampled_at",
    uptime_col: str = "uptime",
    split_col: str = "split",
    label_col: str = "label",
    normal_label: str = "normal",
    features_to_plot: list[str] | None = None,
    default_scenarios: tuple = (1,),
):
    """
    Create a widget UI for plotting RMS features by scenario.

    Returns
    -------
    ui : ipywidgets.Widget
        Top-level widget container. Use display(ui) in a notebook.
    """

    if features_to_plot is None:
        features_to_plot = [
            "vel_rms_x", "vel_rms_y", "vel_rms_z",
            "accel_rms_x", "accel_rms_y", "accel_rms_z"
        ]

    required_cols = [scenario_col, time_col, uptime_col, *features_to_plot]
    missing_cols = [c for c in required_cols if c not in fit_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_plot = fit_df.copy()
    df_plot[time_col] = pd.to_datetime(df_plot[time_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[time_col, scenario_col]).copy()
    df_plot[uptime_col] = df_plot[uptime_col].astype(bool)
    has_split = split_col in df_plot.columns
    has_labels = label_col in df_plot.columns

    scenario_options = sorted(df_plot[scenario_col].unique().tolist())
    if not scenario_options:
        raise ValueError(f"No values found in column '{scenario_col}'.")

    default_scenarios = tuple(s for s in default_scenarios if s in scenario_options)
    if not default_scenarios:
        default_scenarios = (scenario_options[0],)

    def shade_uptime_regions(ax, x_vals, states):
        if not x_vals or not states:
            return
        if len(x_vals) == 1:
            ax.axvspan(
                x_vals[0], x_vals[0],
                alpha=0.12,
                color="green" if states[0] else "red",
            )
            return
        start_idx = 0
        current_state = states[0]
        for i in range(1, len(x_vals)):
            if states[i] != current_state:
                ax.axvspan(
                    x_vals[start_idx], x_vals[i],
                    alpha=0.12,
                    color="green" if current_state else "red"
                )
                start_idx = i
                current_state = states[i]
        ax.axvspan(
            x_vals[start_idx], x_vals[-1],
            alpha=0.12,
            color="green" if current_state else "red"
        )

    def shade_incident_regions(ax, x_vals, labels):
        if not x_vals or labels is None:
            return

        label_series = pd.Series(labels).fillna("").astype(str).reset_index(drop=True)
        if normal_label:
            incident_mask = label_series.ne(str(normal_label))
        else:
            incident_mask = label_series.ne("")
        incident_mask &= label_series.ne("")
        if not incident_mask.any():
            return

        x_series = pd.Series(x_vals)
        for _, group_idx in label_series[incident_mask].groupby(
            (incident_mask != incident_mask.shift(fill_value=False)).cumsum()
        ).groups.items():
            x_start = x_series.iloc[group_idx.min()]
            x_end = x_series.iloc[group_idx.max()]
            ax.axvspan(
                x_start,
                x_end,
                alpha=_INCIDENT_ALPHA,
                color=_INCIDENT_COLOR,
                zorder=0,
            )

    def filter_scenario(df, *, split_mode, uptime_mode):
        if df.empty:
            return df

        out = df.copy()
        if has_split and split_mode != "Both":
            desired_split = _FIT_VALUE if split_mode == "Fit" else _PRED_VALUE
            out = out[out[split_col].astype(str).str.lower() == desired_split]

        if uptime_mode == "True":
            out = out[out[uptime_col]]
        elif uptime_mode == "False":
            out = out[~out[uptime_col]]

        return out.sort_values(time_col).reset_index(drop=True)

    def boundary_position(df, *, use_index):
        if not has_split or df.empty:
            return None
        split_series = df[split_col].astype(str).str.lower()
        fit_rows = df[split_series == _FIT_VALUE]
        pred_rows = df[split_series == _PRED_VALUE]
        if fit_rows.empty or pred_rows.empty:
            return None
        if use_index:
            return float(fit_rows.index.max()) + 0.5
        return pred_rows[time_col].iloc[0]

    def x_values_for(df, *, use_index):
        if use_index:
            return df.index.tolist(), "sample index"
        return df[time_col].tolist(), time_col

    def plot_scenarios(selected_scenarios, *, x_axis_mode, uptime_mode, split_mode):
        selected_scenarios = list(selected_scenarios)
        if len(selected_scenarios) == 0:
            print("Please select at least one scenario.")
            return

        fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=False)
        axes = axes.ravel()

        single_selection = len(selected_scenarios) == 1
        use_index = x_axis_mode == "Index"
        plotted_any = False

        for ax, feature in zip(axes, features_to_plot):
            for scenario_id in selected_scenarios:
                df_aux = (
                    df_plot.loc[df_plot[scenario_col] == scenario_id]
                    .sort_values(time_col)
                    .reset_index(drop=True)
                    .copy()
                )
                df_aux = filter_scenario(
                    df_aux,
                    split_mode=split_mode,
                    uptime_mode=uptime_mode,
                )
                if df_aux.empty:
                    continue
                plotted_any = True
                x_vals, x_label = x_values_for(df_aux, use_index=use_index)
                if has_labels:
                    shade_incident_regions(ax, x_vals, df_aux[label_col].tolist())
                if single_selection:
                    if uptime_mode == "Both":
                        shade_uptime_regions(ax, x_vals, df_aux[uptime_col].tolist())
                    if split_mode == "Both":
                        boundary_x = boundary_position(df_aux, use_index=use_index)
                        if boundary_x is not None:
                            ax.axvline(
                                boundary_x,
                                color=_SPLIT_COLOR,
                                linewidth=1.2,
                                linestyle="--",
                                zorder=4,
                            )
                if single_selection and has_split and split_mode == "Both":
                    split_series = df_aux[split_col].astype(str).str.lower()
                    fit_rows = df_aux[split_series == _FIT_VALUE]
                    pred_rows = df_aux[split_series == _PRED_VALUE]
                    fit_label = "fit" if feature == features_to_plot[0] else None
                    pred_label = "pred" if feature == features_to_plot[0] else None
                    if not fit_rows.empty:
                        fit_x, _ = x_values_for(fit_rows, use_index=use_index)
                        ax.plot(
                            fit_x,
                            fit_rows[feature],
                            linewidth=1.0,
                            alpha=0.7,
                            color=_LINE_COLOR,
                            label=fit_label,
                        )
                    if not pred_rows.empty:
                        pred_x, _ = x_values_for(pred_rows, use_index=use_index)
                        ax.plot(
                            pred_x,
                            pred_rows[feature],
                            linewidth=1.3,
                            alpha=0.95,
                            color=_LINE_COLOR,
                            label=pred_label,
                        )
                else:
                    ax.plot(
                        x_vals, df_aux[feature],
                        linewidth=1.2, label=f"Scenario {scenario_id}"
                    )
            ax.set_title(feature)
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            if not use_index:
                locator = mdates.AutoDateLocator()
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            if single_selection and has_split and feature == features_to_plot[0]:
                ax.legend(loc="upper left")

        if not plotted_any:
            plt.close(fig)
            print(
                "No data remains after applying the selected scenario, split, "
                "and uptime filters."
            )
            return

        for ax in axes[-2:]:
            ax.set_xlabel(x_label)
            if not use_index:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        if single_selection:
            fig.suptitle(
                f"Scenario {selected_scenarios[0]} - RMS Features "
                f"(show={split_mode}, uptime={uptime_mode}, x={x_axis_mode})"
            )
        else:
            fig.suptitle(
                f"Scenarios {', '.join(map(str, selected_scenarios))} - RMS Features "
                f"(show={split_mode}, uptime={uptime_mode}, x={x_axis_mode})"
            )

        fig.tight_layout()
        plt.show()

    out = widgets.Output(
        layout=widgets.Layout(flex="1 1 auto", min_width="900px", padding="10px")
    )

    def update_plot(_=None):
        with out:
            clear_output(wait=True)
            plot_scenarios(
                scenario_selector.value,
                x_axis_mode=x_toggle.value,
                uptime_mode=uptime_toggle.value,
                split_mode=split_toggle.value,
            )

    def select_all(_):
        scenario_selector.value = tuple(scenario_options)

    def clear_all(_):
        scenario_selector.value = ()

    title_html = widgets.HTML("<b>Select scenarios</b>")
    help_html = widgets.HTML(
        "<span style='color: gray;'>Use Ctrl+click (or Cmd+click on Mac) to select multiple.</span>"
    )

    scenario_selector = widgets.SelectMultiple(
        options=scenario_options,
        value=default_scenarios,
        description="",
        rows=min(14, len(scenario_options)),
        layout=widgets.Layout(width="220px", height="280px")
    )

    plot_button = widgets.Button(
        description="Plot selected",
        button_style="primary",
        layout=widgets.Layout(width="220px")
    )

    x_toggle = widgets.ToggleButtons(
        options=["Timestamp", "Index"],
        value="Timestamp",
        description="X axis",
        layout=widgets.Layout(width="220px"),
    )

    uptime_toggle = widgets.ToggleButtons(
        options=["Both", "True", "False"],
        value="Both",
        description="Uptime",
        layout=widgets.Layout(width="220px"),
    )

    split_toggle = widgets.ToggleButtons(
        options=["Both", "Fit", "Pred"],
        value="Both",
        description="Show",
        layout=widgets.Layout(width="220px"),
    )

    select_all_button = widgets.Button(
        description="Select all",
        layout=widgets.Layout(width="105px")
    )

    clear_all_button = widgets.Button(
        description="Clear",
        layout=widgets.Layout(width="105px")
    )

    plot_button.on_click(update_plot)
    select_all_button.on_click(select_all)
    clear_all_button.on_click(clear_all)
    x_toggle.observe(lambda c: update_plot() if c["name"] == "value" else None)
    uptime_toggle.observe(lambda c: update_plot() if c["name"] == "value" else None)
    split_toggle.observe(lambda c: update_plot() if c["name"] == "value" else None)

    button_row = widgets.HBox(
        [select_all_button, clear_all_button],
        layout=widgets.Layout(gap="10px")
    )

    legend_html = widgets.HTML("""
        <b>Legend</b><br><br>
        X axis: <code>Timestamp</code> or <code>Index</code><br><br>
        Uptime filter: <code>Both</code>, <code>True</code>, <code>False</code><br><br>
        Split filter: <code>Both</code>, <code>Fit</code>, <code>Pred</code><br><br>
        <span style="background:rgba(224,123,0,0.15);padding:2px 8px;">&nbsp;</span> labeled anomaly / incident window<br><br>
        <span style="background:rgba(0,128,0,0.25);padding:2px 8px;">&nbsp;</span> uptime = True<br><br>
        <span style="background:rgba(255,0,0,0.25);padding:2px 8px;">&nbsp;</span> uptime = False<br><br>
        <span style="border-top:2px dashed black;display:inline-block;width:20px;vertical-align:middle;"></span> fit | pred boundary
        <br><span style="color: gray;">Boundary appears only for a single scenario when Show = Both.</span>
    """)

    controls = widgets.VBox(
        [
            title_html, help_html, scenario_selector, button_row, plot_button,
            widgets.HTML("<b>X axis</b>"), x_toggle,
            widgets.HTML("<b>Uptime</b>"), uptime_toggle,
            widgets.HTML("<b>Show</b>"), split_toggle,
            widgets.HTML("<hr style='margin:8px 0'>"),
            legend_html,
        ],
        layout=widgets.Layout(
            width="250px", min_width="250px", padding="10px",
            border="1px solid #dddddd", align_items="stretch", gap="10px"
        )
    )

    ui = widgets.HBox(
        [out, controls],
        layout=widgets.Layout(width="100%", align_items="flex-start", gap="20px")
    )

    update_plot()
    return ui
