"""Notebook-02 sigmoid scoring widget — the per-scenario replay debugger.

Drives ``_build_sigmoid_scoring_figure`` with cached model fits and
pre-scored batches so sliders for alpha/beta/threshold respond without a
refit. Exposes a "Export all defaults" action that renders one figure per
scenario using the current slider values. Consumer: ``notebooks/02_model_debugging.ipynb``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output, display as _ipy_display

from sample_processing.model.anomaly_model import load_model_params
from sample_processing.model.scenario_groups import (
    get_scenario_group_key,
    get_scenario_group_label,
)

from ._helpers import DEFAULT_WIDGET_EXPORT_DIR, _ACCEL_COLS, _VEL_COLS, _scenario_slug
from ._scoring_figure import _build_sigmoid_scoring_figure
from ._sigmoid_math import _FastPayloadKey, build_sigmoid_scoring_payload_fast


def create_sigmoid_scoring_widget(
    full_df,
    scenario_col="scenario_id",
    split_col="split",
    fit_value="fit",
    pred_value="pred",
    time_col="sampled_at",
    label_col="label",
    normal_label="normal",
    default_x_axis="Index",
    uptime_col="uptime",
    cyclic_col=None,
    residual_ymin: float = -2.0,
    default_state: dict[str, Any] | None = None,
    export_dir: str | Path = DEFAULT_WIDGET_EXPORT_DIR,
):
    """Optimized scoring widget with cached model fits."""
    default_state = dict(default_state or {})
    df = full_df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[time_col]).copy()
    scenario_options = sorted(df[scenario_col].dropna().unique().tolist())
    initial_scenario_id = default_state.get("scenario_id", scenario_options[0] if scenario_options else None)
    mp = load_model_params(scenario_id=initial_scenario_id)
    initial_alpha = float(default_state.get("alpha", default_state.get("alpha_vel", default_state.get("alpha_accel", mp.alpha_vel))))
    initial_beta = float(default_state.get("beta", default_state.get("beta_vel", default_state.get("beta_accel", mp.beta_vel))))
    _desc = {"description_width": "120px"}
    _sl = widgets.Layout(width="340px")

    dropdown = widgets.Dropdown(
        options=scenario_options, value=initial_scenario_id, description="Scenario",
        style={"description_width": "80px"}, layout=widgets.Layout(width="200px"),
    )
    vel_col_dropdown = widgets.Dropdown(
        options=_VEL_COLS, value=default_state.get("vel_col", _VEL_COLS[0]), description="Vel feat",
        style={"description_width": "70px"}, layout=widgets.Layout(width="270px"),
    )
    accel_col_dropdown = widgets.Dropdown(
        options=_ACCEL_COLS, value=default_state.get("accel_col", _ACCEL_COLS[0]), description="Accel feat",
        style={"description_width": "70px"}, layout=widgets.Layout(width="280px"),
    )
    x_toggle = widgets.ToggleButtons(
        options=["Time", "Index"],
        value=default_state.get("x_axis", default_x_axis if default_x_axis in ("Time", "Index") else "Index"),
        description="x-axis",
        style={"description_width": "55px", "button_width": "78px"},
        layout=widgets.Layout(width="220px"),
    )
    split_view_btn = widgets.ToggleButtons(
        options=["pred", "fit", "both"], value=default_state.get("show", "both"),
        description="show", style={"description_width": "45px", "button_width": "78px"},
        layout=widgets.Layout(width="280px"),
    )
    uptime_btn = widgets.ToggleButtons(
        options=["All data", "Uptime only"], value=default_state.get("uptime_mode", "Uptime only"),
        description="Uptime", style={"description_width": "60px", "button_width": "90px"},
        layout=widgets.Layout(width="270px"),
    )
    cyclic_btn = widgets.ToggleButtons(
        options=["All", "Cyclic ON"], value=default_state.get("cyclic_mode", "All"),
        description="Cyclic", style={"description_width": "55px", "button_width": "78px"},
        layout=widgets.Layout(width="220px", display="none"),
    )
    on_mask_btn = widgets.ToggleButtons(
        options=["Both", "ON only", "OFF only"], value=default_state.get("on_mask_mode", "Both"),
        description="ON mask", style={"description_width": "65px", "button_width": "78px"},
        layout=widgets.Layout(width="320px", display="none"),
    )
    alpha_sl = widgets.FloatSlider(value=initial_alpha, min=0.1, max=10.0, step=0.05,
        description="alpha", continuous_update=False, readout_format=".2f", style=_desc, layout=_sl)
    beta_sl = widgets.FloatSlider(value=initial_beta, min=-2.0, max=12.0, step=0.05,
        description="beta", continuous_update=False, readout_format=".2f", style=_desc, layout=_sl)
    threshold_vel_sl = widgets.FloatSlider(value=default_state.get("threshold_vel", mp.threshold_vel), min=0.0, max=8.0, step=0.25,
        description="vel thr", continuous_update=False, readout_format=".2f", style=_desc, layout=_sl)
    threshold_accel_sl = widgets.FloatSlider(value=default_state.get("threshold_accel", mp.threshold_accel), min=0.0, max=8.0, step=0.25,
        description="acc thr", continuous_update=False, readout_format=".2f", style=_desc, layout=_sl)
    top_k_sl = widgets.IntSlider(value=default_state.get("window_top_k", mp.window_top_k), min=1, max=20,
        description="window_top_k", continuous_update=False, style=_desc, layout=_sl)
    fusion_sl = widgets.FloatSlider(value=default_state.get("fusion_threshold", mp.fusion_threshold), min=0.01, max=1.0, step=0.01,
        description="fusion_threshold", continuous_update=False, readout_format=".2f", style=_desc, layout=_sl)
    plot_btn = widgets.Button(description="Plot", button_style="primary", layout=widgets.Layout(width="110px"))
    export_btn = widgets.Button(description="Export all defaults", button_style="", layout=widgets.Layout(width="170px"))
    group_html = widgets.HTML(layout=widgets.Layout(width="100%", margin="4px 0 6px 0"))
    export_status = widgets.HTML(layout=widgets.Layout(width="100%", margin="4px 0 8px 0"))
    out = widgets.Output(layout=widgets.Layout(width="100%", padding="6px"))

    payload_cache: dict[_FastPayloadKey, dict[str, Any]] = {}
    ui_state: dict[str, Any] = {"last_payload": None}

    def _state_from_controls() -> dict[str, Any]:
        return {
            "scenario_id": dropdown.value,
            "show": split_view_btn.value,
            "uptime_mode": uptime_btn.value,
            "cyclic_mode": cyclic_btn.value,
            "on_mask_mode": on_mask_btn.value,
            "vel_col": vel_col_dropdown.value,
            "accel_col": accel_col_dropdown.value,
            "alpha": alpha_sl.value,
            "beta": beta_sl.value,
            "threshold_vel": threshold_vel_sl.value,
            "threshold_accel": threshold_accel_sl.value,
            "window_top_k": top_k_sl.value,
            "fusion_threshold": fusion_sl.value,
            "x_axis": x_toggle.value,
        }

    def _payload_key_from_state(state: dict[str, Any]) -> _FastPayloadKey:
        return _FastPayloadKey(
            scenario_id=state["scenario_id"],
            show=state["show"],
            uptime_only=state["uptime_mode"] == "Uptime only",
            cyclic_only=state["cyclic_mode"] == "Cyclic ON",
            on_mask_mode=state["on_mask_mode"],
            is_cyclic=False,
            vel_col=state["vel_col"],
            accel_col=state["accel_col"],
            alpha=round(float(state["alpha"]), 6),
            beta=round(float(state["beta"]), 6),
            threshold_vel=round(float(state["threshold_vel"]), 6),
            threshold_accel=round(float(state["threshold_accel"]), 6),
            window_top_k=int(state["window_top_k"]),
            fusion_thr=round(float(state["fusion_threshold"]), 6),
        )

    def _get_payload_for_state(state: dict[str, Any]) -> dict[str, Any]:
        key = _payload_key_from_state(state)
        payload = payload_cache.get(key)
        if payload is None:
            payload = build_sigmoid_scoring_payload_fast(
                full_df=df,
                scenario_id=state["scenario_id"],
                vel_col=state["vel_col"],
                accel_col=state["accel_col"],
                show=state["show"],
                uptime_only=state["uptime_mode"] == "Uptime only",
                cyclic_only=state["cyclic_mode"] == "Cyclic ON",
                on_mask_mode=state["on_mask_mode"],
                alpha_vel=float(state["alpha"]),
                beta_vel=float(state["beta"]),
                threshold_vel=float(state["threshold_vel"]),
                alpha_accel=float(state["alpha"]),
                beta_accel=float(state["beta"]),
                threshold_accel=float(state["threshold_accel"]),
                window_top_k=int(state["window_top_k"]),
                fusion_thr=float(state["fusion_threshold"]),
                scenario_col=scenario_col,
                split_col=split_col,
                fit_value=fit_value,
                pred_value=pred_value,
                time_col=time_col,
                label_col=label_col,
                normal_label=normal_label,
                uptime_col=uptime_col,
                cyclic_col=cyclic_col,
            )
            payload_cache[key] = payload
        return payload

    def _get_payload() -> dict[str, Any]:
        payload = _get_payload_for_state(_state_from_controls())
        ui_state["last_payload"] = payload
        return payload

    def _apply_group_defaults_for_scenario(scenario_id: Any) -> None:
        params = load_model_params(scenario_id=scenario_id)
        group_html.value = (
            f"<b>Scenario group</b>: {get_scenario_group_label(scenario_id)} "
            f"(<code>{get_scenario_group_key(scenario_id)}</code>)"
        )
        alpha_sl.value = float(params.alpha_vel)
        beta_sl.value = float(params.beta_vel)
        threshold_vel_sl.value = float(params.threshold_vel)
        threshold_accel_sl.value = float(params.threshold_accel)
        top_k_sl.value = int(params.window_top_k)
        fusion_sl.value = float(params.fusion_threshold)

    def _sync_zoom_controls(*_args) -> None:
        sid = dropdown.value
        if sid is None:
            return
        cyclic_btn.layout.display = "none"
        on_mask_btn.layout.display = "none"
        try:
            payload = _get_payload()
        except Exception as exc:
            with out:
                clear_output(wait=True)
                import traceback
                traceback.print_exc()
                print(f"Payload error: {exc}")
            return
        ui_state["last_payload"] = payload
        _plot()

    def _plot(_=None) -> None:
        sid = dropdown.value
        if sid is None:
            return

        with out:
            clear_output(wait=True)
            try:
                payload = _get_payload()
            except Exception as exc:
                print(f"Unable to build scoring payload: {exc}")
                return

            use_index = x_toggle.value == "Index"
            fig = _build_sigmoid_scoring_figure(payload, use_index=use_index, residual_ymin=residual_ymin)
            _ipy_display(fig)
            plt.close(fig)

    def _export_all_defaults(_=None) -> None:
        export_root = Path(export_dir) / "sigmoid_scoring"
        export_root.mkdir(parents=True, exist_ok=True)
        template = {
            "show": default_state.get("show", "both"),
            "uptime_mode": default_state.get("uptime_mode", "Uptime only"),
            "cyclic_mode": default_state.get("cyclic_mode", "All"),
            "on_mask_mode": default_state.get("on_mask_mode", "Both"),
            "vel_col": default_state.get("vel_col", _VEL_COLS[0]),
            "accel_col": default_state.get("accel_col", _ACCEL_COLS[0]),
            "alpha": float(default_state.get("alpha", default_state.get("alpha_vel", default_state.get("alpha_accel", mp.alpha_vel)))),
            "beta": float(default_state.get("beta", default_state.get("beta_vel", default_state.get("beta_accel", mp.beta_vel)))),
            "threshold_vel": float(default_state.get("threshold_vel", mp.threshold_vel)),
            "threshold_accel": float(default_state.get("threshold_accel", mp.threshold_accel)),
            "window_top_k": int(default_state.get("window_top_k", mp.window_top_k)),
            "fusion_threshold": float(default_state.get("fusion_threshold", mp.fusion_threshold)),
            "x_axis": default_state.get("x_axis", default_x_axis if default_x_axis in ("Time", "Index") else "Index"),
        }
        exported = 0
        errors: list[str] = []
        for scenario_id in scenario_options:
            state = dict(template)
            state["scenario_id"] = scenario_id
            try:
                payload = _get_payload_for_state(state)
                fig = _build_sigmoid_scoring_figure(
                    payload,
                    use_index=state["x_axis"] == "Index",
                    residual_ymin=residual_ymin,
                )
                out_path = export_root / f"scenario_{_scenario_slug(scenario_id)}.png"
                fig.savefig(out_path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                exported += 1
            except Exception as exc:
                errors.append(f"{scenario_id}: {exc}")
        if errors:
            export_status.value = f"<b>Exported {exported} scenario(s)</b><br>{'<br>'.join(errors[:5])}"
        else:
            export_status.value = f"<b>Exported {exported} scenario(s)</b> to {export_root}"

    def _on_scenario_change(change) -> None:
        if change["name"] != "value":
            return
        sid = change["new"]
        if sid is None:
            return
        _apply_group_defaults_for_scenario(sid)

    plot_btn.on_click(_plot)
    export_btn.on_click(_export_all_defaults)
    dropdown.observe(_on_scenario_change, names="value")
    for ctl in (
        vel_col_dropdown, accel_col_dropdown, x_toggle, uptime_btn, cyclic_btn,
        on_mask_btn, split_view_btn, alpha_sl, beta_sl, threshold_vel_sl,
        threshold_accel_sl, top_k_sl, fusion_sl,
    ):
        ctl.observe(_sync_zoom_controls, names="value")
    _apply_group_defaults_for_scenario(initial_scenario_id)
    _sync_zoom_controls()
    row_layout = widgets.Layout(width="100%", align_items="center")
    return widgets.VBox(
        [
            widgets.HBox([dropdown, vel_col_dropdown, accel_col_dropdown, x_toggle, split_view_btn], layout=row_layout),
            widgets.HBox([uptime_btn, cyclic_btn, on_mask_btn], layout=row_layout),
            widgets.HBox([alpha_sl, beta_sl], layout=row_layout),
            widgets.HBox([threshold_vel_sl, threshold_accel_sl], layout=row_layout),
            widgets.HBox([top_k_sl, fusion_sl], layout=row_layout),
            widgets.HBox([plot_btn, export_btn], layout=row_layout),
            group_html,
            export_status,
            out,
        ],
        layout=widgets.Layout(width="100%"),
    )
