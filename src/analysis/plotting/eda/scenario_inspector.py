"""Multi-panel per-scenario diagnostic view (RMS, uptime, incidents, split boundary).

Pipeline stage: EDA widget. Acts as the main visual entry point when
exploring a single scenario end-to-end.

Consumer: ``notebooks/01_eda.ipynb`` (``create_scenario_inspector``).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ipywidgets as widgets
from IPython.display import clear_output

_FEATURES = [
    "vel_rms_x", "vel_rms_y", "vel_rms_z",
    "accel_rms_x", "accel_rms_y", "accel_rms_z",
]

_INCIDENT_COLOR    = "#E07B00"
_UPTIME_TRUE_COLOR = "green"
_UPTIME_FALSE_COLOR = "red"
_SPLIT_LINE_COLOR  = "black"


def _shade_uptime(ax: plt.Axes, x: list, states: list[bool]) -> None:
    start_idx = 0
    current = states[0]
    for i in range(1, len(x)):
        if states[i] != current:
            ax.axvspan(x[start_idx], x[i], alpha=0.10,
                       color=_UPTIME_TRUE_COLOR if current else _UPTIME_FALSE_COLOR, zorder=1)
            start_idx, current = i, states[i]
    ax.axvspan(x[start_idx], x[-1], alpha=0.10,
               color=_UPTIME_TRUE_COLOR if current else _UPTIME_FALSE_COLOR, zorder=1)


def _shade_incidents(ax: plt.Axes, x: list, labels: pd.Series) -> None:
    incident_mask = labels.str.contains("incident", na=False)
    if not incident_mask.any():
        return
    x_series = pd.Series(x)
    for label, group_idx in labels[incident_mask].groupby(labels[incident_mask]).groups.items():
        x_start = x_series.iloc[group_idx.min()]
        x_end   = x_series.iloc[group_idx.max()]
        ax.axvspan(x_start, x_end, alpha=0.08, color=_INCIDENT_COLOR, zorder=0)
        ax.axvspan(x_start, x_end, ymin=0.95, ymax=1.0, alpha=0.85, color=_INCIDENT_COLOR, zorder=5)
        ax.text(
            x_start + (x_end - x_start) / 2, 1.0,
            label.replace("_", " "),
            ha="center", va="top",
            transform=ax.get_xaxis_transform(),
            color=_INCIDENT_COLOR, fontweight="bold", clip_on=True,
        )


def _draw_split_boundary(ax: plt.Axes, boundary_x) -> None:
    ax.axvline(boundary_x, color=_SPLIT_LINE_COLOR, linewidth=1.2, linestyle="--", zorder=3)


def _plot_scenario(
    df: pd.DataFrame,
    scenario_id: int,
    scenario_col: str,
    time_col: str,
    uptime_col: str,
    label_col: str,
    split_col: str,
    features: list[str],
    use_index: bool,
    x_range: tuple[int, int] | None,
    y_range: tuple[float, float] | None,
    uptime_filter: str = "Both",
) -> None:
    df_sc = (
        df[df[scenario_col] == scenario_id]
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    if df_sc.empty:
        print(f"No data for scenario {scenario_id}.")
        return

    if uptime_filter == "True":
        df_sc = df_sc[df_sc[uptime_col]].reset_index(drop=True)
    elif uptime_filter == "False":
        df_sc = df_sc[~df_sc[uptime_col]].reset_index(drop=True)

    if df_sc.empty:
        print(f"No data for scenario {scenario_id} with uptime={uptime_filter}.")
        return

    if use_index:
        x_vals  = df_sc.index.tolist()
        x_label = "sample index"
        fit_rows   = df_sc[df_sc[split_col] == "fit"]
        boundary_x = fit_rows.index.max() if not fit_rows.empty else None
    else:
        x_vals  = df_sc[time_col].tolist()
        x_label = time_col
        fit_rows   = df_sc[df_sc[split_col] == "fit"]
        boundary_x = fit_rows[time_col].max() if not fit_rows.empty else None

    uptime_states = df_sc[uptime_col].tolist()
    labels        = df_sc[label_col].reset_index(drop=True)

    fig, axes = plt.subplots(6, 1, figsize=(18, 22), sharex=True)
    axes = axes.ravel()

    for ax, feature in zip(axes, features):
        if feature not in df_sc.columns:
            ax.text(0.5, 0.5, f"Missing: {feature}", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_title(feature)
            continue

        _shade_uptime(ax, x_vals, uptime_states)
        ax.plot(x_vals, df_sc[feature].tolist(), linewidth=1.2, color="#1f77b4", zorder=4)
        ax.set_title(feature)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        if not use_index:
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        if boundary_x is not None:
            _draw_split_boundary(ax, boundary_x)

        _shade_incidents(ax, x_vals, labels)

        if y_range is not None and y_range[1] > y_range[0]:
            ax.set_ylim(y_range)

    # Apply x range via xlim on the shared axis
    if x_range is not None:
        lo, hi = x_range
        lo = max(0, lo)
        hi = min(len(x_vals) - 1, hi)
        axes[0].set_xlim(x_vals[lo], x_vals[hi])

    axes[-1].set_xlabel(x_label)
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")

    fig.suptitle(f"Scenario {scenario_id} — RMS Features  (uptime={uptime_filter})")
    fig.tight_layout()
    plt.show()


def create_scenario_inspector(
    full_df: pd.DataFrame,
    scenario_col: str = "scenario_id",
    time_col: str = "sampled_at",
    uptime_col: str = "uptime",
    label_col: str = "label",
    split_col: str = "split",
    features: list[str] | None = None,
    default_scenario: int | None = None,
    default_x_axis: str = "Index",
) -> widgets.Widget:
    """
    Interactive single-scenario inspector with x/y range sliders for zooming.

    Controls
    --------
    Scenario dropdown, Time/Index toggle, Plot button.
    X range slider  — restrict the visible sample window (maps to timestamps
                      in Time mode automatically).
    Y range slider  — clip all six subplots to the same y limits.
    """
    if features is None:
        features = _FEATURES

    required = [scenario_col, time_col, uptime_col, label_col, split_col, *features]
    missing  = [c for c in required if c not in full_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = full_df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, scenario_col]).copy()
    df[uptime_col] = df[uptime_col].astype(bool)

    scenario_options = sorted(df[scenario_col].unique().tolist())
    if not scenario_options:
        raise ValueError("No scenarios found in DataFrame.")

    if default_scenario is None or default_scenario not in scenario_options:
        default_scenario = scenario_options[0]

    # ── range sliders ──────────────────────────────────────────────────────
    def _scenario_ranges(sid: int) -> tuple[int, float, float]:
        """Return (n_samples, y_min, y_max) for a scenario."""
        sub = df[df[scenario_col] == sid].sort_values(time_col)
        n   = len(sub)
        vals = sub[features].select_dtypes("number").values.ravel()
        vals = vals[~np.isnan(vals)]
        y_lo = float(vals.min()) if len(vals) else 0.0
        y_hi = float(vals.max()) if len(vals) else 1.0
        return n, y_lo, y_hi

    _n0, _y0, _y1 = _scenario_ranges(default_scenario)

    x_slider = widgets.IntRangeSlider(
        value=[0, _n0 - 1], min=0, max=max(_n0 - 1, 1), step=1,
        description="X range",
        layout=widgets.Layout(width="360px"),
        style={"description_width": "60px"},
        continuous_update=False,
    )

    _y_step = max((_y1 - _y0) / 200, 1e-6)
    y_slider = widgets.FloatRangeSlider(
        value=[_y0, _y1], min=_y0, max=max(_y1, _y0 + _y_step), step=_y_step,
        description="Y range",
        layout=widgets.Layout(width="360px"),
        style={"description_width": "60px"},
        continuous_update=False,
        readout_format=".2f",
    )

    x_full_btn = widgets.Button(
        description="Full X", layout=widgets.Layout(width="80px", height="28px"),
        button_style="",
    )
    y_full_btn = widgets.Button(
        description="Full Y", layout=widgets.Layout(width="80px", height="28px"),
        button_style="",
    )

    # ── other controls ─────────────────────────────────────────────────────
    dropdown = widgets.Dropdown(
        options=scenario_options,
        value=default_scenario,
        description="",
        layout=widgets.Layout(width="80px"),
    )

    x_toggle = widgets.ToggleButtons(
        options=["Time", "Index"],
        value=default_x_axis if default_x_axis in ("Time", "Index") else "Index",
        description="",
        layout=widgets.Layout(width="160px"),
        style={"button_width": "70px"},
    )

    uptime_toggle = widgets.ToggleButtons(
        options=["True", "False", "Both"],
        value="Both",
        description="",
        layout=widgets.Layout(width="220px"),
        style={"button_width": "66px"},
    )

    plot_button = widgets.Button(
        description="Plot",
        button_style="primary",
        layout=widgets.Layout(width="160px"),
    )

    out = widgets.Output(
        layout=widgets.Layout(flex="1 1 auto", min_width="900px", padding="10px")
    )

    # ── slider reset when scenario changes ────────────────────────────────
    def _reset_sliders(sid: int) -> None:
        n, y_lo, y_hi = _scenario_ranges(sid)
        step = max((y_hi - y_lo) / 200, 1e-6)
        # X slider: safe update order
        x_slider.value = [0, 0]
        x_slider.max   = max(n - 1, 1)
        x_slider.value = [0, n - 1]
        # Y slider: safe update order
        y_slider.value = [y_slider.min, y_slider.min]
        y_slider.step  = step
        y_slider.min   = y_lo
        y_slider.max   = max(y_hi, y_lo + step)
        y_slider.value = [y_lo, y_hi]

    def _reset_x(_=None):
        n, _, _ = _scenario_ranges(dropdown.value)
        x_slider.value = [0, n - 1]

    def _reset_y(_=None):
        n, y_lo, y_hi = _scenario_ranges(dropdown.value)
        y_slider.value = [y_lo, y_hi]

    x_full_btn.on_click(_reset_x)
    y_full_btn.on_click(_reset_y)

    # ── plot callback ──────────────────────────────────────────────────────
    def update_plot(_=None):
        x_lo, x_hi = x_slider.value
        y_lo, y_hi = y_slider.value
        # Only pass range if user has zoomed (not full extent)
        n = x_slider.max + 1
        x_rng = (x_lo, x_hi) if (x_lo > 0 or x_hi < n - 1) else None
        y_rng = (y_lo, y_hi) if (y_lo > y_slider.min or y_hi < y_slider.max) else None
        with out:
            clear_output(wait=True)
            _plot_scenario(
                df=df,
                scenario_id=dropdown.value,
                scenario_col=scenario_col,
                time_col=time_col,
                uptime_col=uptime_col,
                label_col=label_col,
                split_col=split_col,
                features=features,
                use_index=(x_toggle.value == "Index"),
                x_range=x_rng,
                y_range=y_rng,
                uptime_filter=uptime_toggle.value,
            )

    def on_scenario_change(change):
        if change["name"] == "value":
            _reset_sliders(change["new"])
            update_plot()

    plot_button.on_click(update_plot)
    dropdown.observe(on_scenario_change)
    x_slider.observe(lambda c: update_plot() if c["name"] == "value" else None)
    y_slider.observe(lambda c: update_plot() if c["name"] == "value" else None)
    x_toggle.observe(lambda c: update_plot() if c["name"] == "value" else None)
    uptime_toggle.observe(lambda c: update_plot() if c["name"] == "value" else None)

    # ── layout ────────────────────────────────────────────────────────────
    legend_html = widgets.HTML("""
        <b>Legend</b><br><br>
        <span style="background:rgba(0,128,0,0.25);padding:2px 8px;">&nbsp;</span> uptime = True<br><br>
        <span style="background:rgba(255,0,0,0.25);padding:2px 8px;">&nbsp;</span> uptime = False<br><br>
        <span style="background:rgba(224,123,0,0.35);padding:2px 8px;">&nbsp;</span> incident window<br><br>
        <span style="border-top:2px dashed black;display:inline-block;width:20px;vertical-align:middle;"></span> fit | pred boundary
    """)

    controls = widgets.VBox(
        [
            widgets.HTML("<b>Scenario</b>"),
            dropdown,
            widgets.HTML("<b>X axis</b>"),
            x_toggle,
            widgets.HTML("<b>Uptime</b>"),
            uptime_toggle,
            plot_button,
            widgets.HTML("<hr style='margin:8px 0'>"),
            widgets.HTML("<b>X range</b> (sample index)"),
            widgets.HBox([x_slider, x_full_btn]),
            widgets.HTML("<b>Y range</b>"),
            widgets.HBox([y_slider, y_full_btn]),
            widgets.HTML("<hr style='margin:8px 0'>"),
            legend_html,
        ],
        layout=widgets.Layout(
            width="500px",
            min_width="500px",
            padding="8px",
            border="1px solid #dddddd",
            align_items="stretch",
            gap="6px",
        ),
    )

    ui = widgets.HBox(
        [out, controls],
        layout=widgets.Layout(width="100%", align_items="flex-start", gap="20px"),
    )

    update_plot()
    return ui
