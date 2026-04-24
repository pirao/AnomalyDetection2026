"""Internal API replay widget aligned with the evaluation/test protocol."""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output, display

from analysis.api_replay import (
    df_to_timeseries,
    diagnose_replay_against_incidents,
    get_incident_spans,
    simulate_api_replay_one_scenario,
)
from sample_processing.model.anomaly_model import (
    AnomalyModel,
    load_alert_params,
    load_model_params,
    load_pipeline_params,
)
from sample_processing.model.interface import ModelParams
from sample_processing.model.scenario_groups import (
    get_scenario_group_key,
    get_scenario_group_label,
)

from ._helpers import DEFAULT_WIDGET_EXPORT_DIR, _scenario_slug
from ._replay_rendering import _compute_replay_plot_state, _plot_replay_column, _with_split_view

# Module-level cache for fitted models (keyed by scenario_id).
# Avoids re-fitting AnomalyModel on every slider change when models=None.
_api_replay_fit_cache: dict[object, AnomalyModel] = {}


def _effective_model_params(
    base_params: ModelParams,
    *,
    alpha: float,
    beta: float,
    threshold_vel: float,
    threshold_accel: float,
    window_top_k: int,
    fusion_threshold: float,
) -> ModelParams:
    return ModelParams(
        **{
            **base_params.model_dump(),
            "baseline_scaler": "standard",
            "alpha_vel": float(alpha),
            "alpha_accel": float(alpha),
            "beta_vel": float(beta),
            "beta_accel": float(beta),
            "threshold_vel": float(threshold_vel),
            "threshold_accel": float(threshold_accel),
            "window_top_k": int(window_top_k),
            "fusion_threshold": float(fusion_threshold),
        }
    )


def _format_timestamp(ts):
    ts = pd.to_datetime(ts, errors="coerce", utc=True)
    return "—" if pd.isna(ts) else ts.strftime("%Y-%m-%d %H:%M")


def _format_delta_hours(value):
    if pd.isna(value):
        return "—"
    return f"{float(value):+.2f}h"


def _diagnosis_html_v2(title: str, diag: dict) -> str:
    summary = diag.get("summary", {})
    alerts_df = diag.get("alerts_df", pd.DataFrame())
    incidents_df = diag.get("incidents_df", pd.DataFrame())
    if alerts_df.empty:
        alerts_txt = "API alerts: -"
    else:
        bits = []
        for row in alerts_df.itertuples(index=False):
            if row.classification == "in-window":
                suffix = "in-window"
            elif row.classification == "spurious":
                suffix = "spurious"
            else:
                suffix = f"{row.classification}, delta={_format_delta_hours(row.delta_to_grace_hours)}"
            bits.append(f"{_format_timestamp(row.timestamp)} ({suffix})")
        alerts_txt = "API alerts: " + "; ".join(bits)
    if incidents_df.empty:
        incidents_txt = "incidents: -"
    else:
        bits = []
        for row in incidents_df.itertuples(index=False):
            if row.hit:
                bits.append(f"{_format_timestamp(row.start)}->{_format_timestamp(row.end)} hit")
            else:
                bits.append(
                    f"{_format_timestamp(row.start)}->{_format_timestamp(row.end)} miss "
                    f"(nearest={_format_timestamp(row.nearest_alert)}, "
                    f"delta={_format_delta_hours(row.nearest_alert_delta_hours)})"
                )
        incidents_txt = "incidents: " + "; ".join(bits)
    return (
        f"<b>{title}</b>: "
        f"API alerts={summary.get('alerts', 0)} "
        f"in-window={summary.get('in_window', 0)} "
        f"early={summary.get('early', 0)} "
        f"late={summary.get('late', 0)} "
        f"spurious={summary.get('spurious', 0)} "
        f"| incidents={summary.get('incidents', 0)} "
        f"covered={summary.get('covered_incidents', 0)} "
        f"missed={summary.get('missed_incidents', 0)}"
        f"<br>{alerts_txt}"
        f"<br>{incidents_txt}"
        f"<br><i>Evaluation is based on emitted API alerts, not raw engine candidates.</i>"
    )


def _counts_line(plot_state: dict[str, object] | None) -> str:
    raw_counts = plot_state["raw"] if plot_state is not None else {}
    emitted_counts = plot_state["emitted"] if plot_state is not None else {}
    return (
        f"<b>Diagnostics vs API output</b>: "
        f"engine pending={len(raw_counts.get('pending_xs', []))} "
        f"engine channel candidates={len(raw_counts.get('open_candidate_xs', [])) + len(raw_counts.get('realert_candidate_xs', []))} "
        f"engine group emits={len(raw_counts.get('group3_emitted_xs', [])) + len(raw_counts.get('group6_emitted_xs', []))} "
        f"| API channel alerts={len(emitted_counts.get('open_xs', [])) + len(emitted_counts.get('realert_xs', []))} "
        f"API grouped alerts={len(emitted_counts.get('group3_xs', [])) + len(emitted_counts.get('group6_xs', []))}"
    )


def _build_replay_info_html(
    *,
    scenario_id,
    show,
    model_params,
    pipeline_params,
    alert_params,
    stride_hours: float,
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    api_pred_df: pd.DataFrame,
    api_diag: dict,
    plot_state: dict[str, object] | None,
) -> str:
    return (
        f"<b>Scenario {scenario_id}</b> &nbsp;|&nbsp; "
        f"group={get_scenario_group_label(scenario_id)} &nbsp;|&nbsp; "
        f"view={show} &nbsp;|&nbsp; "
        f"protocol={pipeline_params.model_window_size_hours:.1f}h window / {stride_hours:.1f}h stride<br>"
        f"<b>Model</b>: alpha={model_params.alpha_vel:.2f}, beta={model_params.beta_vel:.2f}, "
        f"vel_thr={model_params.threshold_vel:.2f}, acc_thr={model_params.threshold_accel:.2f}, "
        f"top_k={model_params.window_top_k}, fusion_thr={model_params.fusion_threshold:.2f}<br>"
        f"<b>Pipeline</b>: window={pipeline_params.model_window_size_hours}h overlap={pipeline_params.window_overlap_hours}h stride={stride_hours}h &nbsp;|&nbsp; "
        f"<b>Alert</b>: rel_thr={alert_params.relative_threshold:.2f}, cooldown={alert_params.min_cooldown_windows}, "
        f"confirm={alert_params.confirmation_count}/{alert_params.confirmation_window}, "
        f"group_confirm={alert_params.group_confirmation_count}/{alert_params.group_confirmation_window}, "
        f"group6_enabled={alert_params.enable_group6_alerts}<br>"
        f"<b>Replay sizes</b>: fit rows={len(fit_df)} pred rows={len(pred_df)} "
        f"| windows={len(api_pred_df)} API alerts={int(api_pred_df['alert'].sum()) if not api_pred_df.empty else 0}<br>"
        f"{_counts_line(plot_state)}<br>"
        f"{_diagnosis_html_v2('API/Test replay', api_diag)}"
    )


def create_api_replay_widget_ui(
    *,
    full_df,
    scenario_col="scenario_id",
    split_col="split",
    fit_value="fit",
    pred_value="pred",
    time_col="sampled_at",
    label_col="label",
    normal_label="normal",
    default_x_axis="Time",
    default_state: dict | None = None,
    export_dir: str | Path = DEFAULT_WIDGET_EXPORT_DIR,
    models: dict | None = None,
):
    """Single-view widget for the API/test replay protocol.

    Parameters
    ----------
    models :
        Optional ``{scenario_id: AnomalyModel}`` mapping produced by
        ``load_fitted_models()``. When provided the widget skips the internal
        fit step and uses the pre-fitted model directly. Alarm-rule sliders
        still work normally. Pass ``None`` (default) to keep the original
        fit-from-data behaviour.
    """
    default_state = dict(default_state or {})
    df = full_df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[time_col]).copy()
    scenario_options = sorted(df[scenario_col].dropna().unique().tolist())
    initial_scenario_id = default_state.get("scenario_id", scenario_options[0] if scenario_options else None)

    # Load once at init — avoids repeated YAML reads on every slider change.
    _pipeline_params = load_pipeline_params()
    _alert_params = load_alert_params()

    dropdown = widgets.Dropdown(
        options=scenario_options,
        value=initial_scenario_id,
        description="Scenario",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="200px"),
    )
    x_toggle = widgets.ToggleButtons(
        options=["Time", "Index"],
        value=default_state.get("x_axis", default_x_axis if default_x_axis in ("Time", "Index") else "Time"),
        description="x-axis",
        style={"description_width": "55px", "button_width": "78px"},
        layout=widgets.Layout(width="220px"),
    )
    split_view_btn = widgets.ToggleButtons(
        options=["pred", "fit", "both"],
        value=default_state.get("show", "pred"),
        description="show",
        style={"description_width": "45px", "button_width": "78px"},
        layout=widgets.Layout(width="280px"),
    )
    initial_scenario_id = default_state.get("scenario_id", scenario_options[0] if scenario_options else None)
    default_model_params = load_model_params(scenario_id=initial_scenario_id)
    initial_alpha = float(default_state.get("alpha", default_state.get("alpha_vel", default_state.get("alpha_accel", default_model_params.alpha_vel))))
    initial_beta = float(default_state.get("beta", default_state.get("beta_vel", default_state.get("beta_accel", default_model_params.beta_vel))))
    top_k_sl = widgets.IntSlider(
        value=int(default_state.get("window_top_k", default_model_params.window_top_k)),
        min=1,
        max=20,
        step=1,
        description="window_top_k",
        continuous_update=False,
        style={"description_width": "110px"},
        layout=widgets.Layout(width="300px"),
    )
    fusion_sl = widgets.FloatSlider(
        value=float(default_state.get("fusion_threshold", default_model_params.fusion_threshold)),
        min=0.01,
        max=1.0,
        step=0.01,
        description="fusion_threshold",
        continuous_update=False,
        readout_format=".2f",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="300px"),
    )
    threshold_vel_sl = widgets.FloatSlider(
        value=float(default_state.get("threshold_vel", default_model_params.threshold_vel)),
        min=0.0,
        max=8.0,
        step=0.25,
        description="thr vel",
        continuous_update=False,
        readout_format=".2f",
        style={"description_width": "65px"},
        layout=widgets.Layout(width="260px"),
    )
    alpha_sl = widgets.FloatSlider(
        value=initial_alpha,
        min=0.05,
        max=8.0,
        step=0.05,
        description="alpha",
        continuous_update=False,
        readout_format=".2f",
        style={"description_width": "65px"},
        layout=widgets.Layout(width="260px"),
    )
    beta_sl = widgets.FloatSlider(
        value=initial_beta,
        min=-2.0,
        max=12.0,
        step=0.05,
        description="beta",
        continuous_update=False,
        readout_format=".2f",
        style={"description_width": "65px"},
        layout=widgets.Layout(width="260px"),
    )
    threshold_accel_sl = widgets.FloatSlider(
        value=float(default_state.get("threshold_accel", default_model_params.threshold_accel)),
        min=0.0,
        max=8.0,
        step=0.25,
        description="thr acc",
        continuous_update=False,
        readout_format=".2f",
        style={"description_width": "65px"},
        layout=widgets.Layout(width="260px"),
    )
    plot_btn = widgets.Button(description="Plot", button_style="primary", layout=widgets.Layout(width="110px"))
    export_btn = widgets.Button(description="Export all defaults", layout=widgets.Layout(width="170px"))
    group_html = widgets.HTML(layout=widgets.Layout(width="100%", margin="4px 0 6px 0"))
    info_html = widgets.HTML(layout=widgets.Layout(width="100%", margin="4px 0 8px 0"))
    export_status = widgets.HTML(layout=widgets.Layout(width="100%", margin="4px 0 8px 0"))
    out = widgets.Output(layout=widgets.Layout(width="100%", padding="6px"))
    replay_state: dict[str, object] = {
        "df": pd.DataFrame(),
        "api_pred_df": pd.DataFrame(),
        "diagnostics": {},
        "incidents": [],
        "model_params": None,
        "scenario_id": None,
    }
    interaction_state = {"applying_defaults": False}
    anomaly_fill_color = "#9ecae1"

    def _apply_group_defaults_for_scenario(scenario_id) -> ModelParams:
        params = load_model_params(scenario_id=scenario_id)
        group_html.value = (
            f"<b>Scenario group</b>: {get_scenario_group_label(scenario_id)} "
            f"(<code>{get_scenario_group_key(scenario_id)}</code>)"
        )
        interaction_state["applying_defaults"] = True
        try:
            top_k_sl.value = int(params.window_top_k)
            fusion_sl.value = float(params.fusion_threshold)
            alpha_sl.value = float(params.alpha_vel)
            beta_sl.value = float(params.beta_vel)
            threshold_vel_sl.value = float(params.threshold_vel)
            threshold_accel_sl.value = float(params.threshold_accel)
        finally:
            interaction_state["applying_defaults"] = False
        return params

    def _run_replay(
        scenario_id,
        *,
        show: str,
        alpha: float,
        beta: float,
        threshold_vel: float,
        threshold_accel: float,
        window_top_k: int,
        fusion_threshold: float,
    ):
        model_params = _effective_model_params(
            load_model_params(scenario_id=scenario_id),
            alpha=alpha,
            beta=beta,
            threshold_vel=threshold_vel,
            threshold_accel=threshold_accel,
            window_top_k=window_top_k,
            fusion_threshold=fusion_threshold,
        )
        stride_hours = float(_pipeline_params.model_window_size_hours - _pipeline_params.window_overlap_hours)

        df_sid = df[df[scenario_col] == scenario_id].sort_values(time_col).copy()
        fit_df = df_sid[df_sid[split_col] == fit_value].copy()
        pred_df = df_sid[df_sid[split_col] == pred_value].copy()
        incidents = get_incident_spans(df_sid, label_col=label_col, time_col=time_col, normal_label=normal_label) if label_col in df_sid.columns else []
        if fit_df.empty or pred_df.empty:
            raise ValueError(f"Scenario {scenario_id}: missing fit or pred split.")

        if models is not None:
            _export_model = models.get(scenario_id)
        elif scenario_id in _api_replay_fit_cache:
            _export_model = _api_replay_fit_cache[scenario_id]
        else:
            _export_model = AnomalyModel(is_cyclic=False)
            _export_model.params = model_params
            _export_model.params.baseline_scaler = "standard"
            if hasattr(_export_model._backend, "baseline_scaler"):
                _export_model._backend.baseline_scaler = "standard"
            _export_model.fit(df_to_timeseries(fit_df, time_col=time_col))
            _api_replay_fit_cache[scenario_id] = _export_model

        api_pred_df = simulate_api_replay_one_scenario(
            fit_df=fit_df,
            pred_df=pred_df,
            model=_export_model,
            sensor_id=f"sensor_{scenario_id}",
            baseline_scaler=None,
            model_params_override=model_params,
            alert_params=_alert_params,
            time_col=time_col,
        )
        api_fit_df = simulate_api_replay_one_scenario(
            fit_df=fit_df,
            pred_df=fit_df,
            model=_export_model,
            sensor_id=f"sensor_{scenario_id}",
            baseline_scaler=None,
            model_params_override=model_params,
            alert_params=_alert_params,
            time_col=time_col,
        ) if show in ("fit", "both") else pd.DataFrame()
        api_plot_df = _with_split_view(show, api_pred_df, api_fit_df)
        api_diag = diagnose_replay_against_incidents(api_pred_df, incidents, tolerance_hours=2.0)
        plot_state = _compute_replay_plot_state(api_plot_df, use_index=False, alert_params=_alert_params) if not api_plot_df.empty else None
        info_value = _build_replay_info_html(
            scenario_id=scenario_id,
            show=show,
            model_params=model_params,
            pipeline_params=_pipeline_params,
            alert_params=_alert_params,
            stride_hours=stride_hours,
            fit_df=fit_df,
            pred_df=pred_df,
            api_pred_df=api_pred_df,
            api_diag=api_diag,
            plot_state=plot_state,
        )
        return {
            "model_params": model_params,
            "pipeline_params": _pipeline_params,
            "alert_params": _alert_params,
            "fit_df": fit_df,
            "pred_df": pred_df,
            "incidents": incidents,
            "api_pred_df": api_pred_df,
            "api_plot_df": api_plot_df,
            "api_diag": api_diag,
            "info_html": info_value,
        }

    def _build_api_figure(*, api_plot_df, incidents, use_index, show, model_params, alert_params, scenario_id):
        plt.ioff()
        fig, axes = plt.subplots(6, 1, figsize=(18, 28), sharex=True, gridspec_kw={"height_ratios": [2.6, 1.8, 1.4, 1.5, 1.8, 3.0]})
        _plot_replay_column(
            axes,
            api_plot_df,
            incidents=incidents,
            use_index=use_index,
            split_view_value=show,
            model_params=model_params,
            alert_params=alert_params,
            title="API/Test replay",
            anomaly_fill_color=anomaly_fill_color,
            time_col=time_col,
        )
        fig.suptitle(f"Scenario {scenario_id} — API replay (fit-trained, {show} windows)", y=0.995)
        plt.tight_layout()
        return fig

    def _render_current_plot() -> None:
        sid = replay_state.get("scenario_id")
        api_plot_df = replay_state.get("df", pd.DataFrame())
        incidents = replay_state.get("incidents", [])
        model_params = replay_state.get("model_params")
        if sid is None or model_params is None:
            return
        with out:
            clear_output(wait=True)
            if api_plot_df.empty:
                print("Replay returned no windows.")
                return
            plt.ioff()
            use_index = (x_toggle.value == "Index")
            fig, axes = plt.subplots(6, 1, figsize=(18, 28), sharex=True, gridspec_kw={"height_ratios": [2.6, 1.8, 1.4, 1.5, 1.8, 3.0]})
            _plot_replay_column(
                axes,
                api_plot_df,
                incidents=incidents,
                use_index=use_index,
                split_view_value=split_view_btn.value,
                model_params=model_params,
                alert_params=_alert_params,
                title="API/Test replay",
                anomaly_fill_color=anomaly_fill_color,
                time_col=time_col,
            )
            fig.suptitle(f"Scenario {sid} â€” API replay (fit-trained, {split_view_btn.value} windows)", y=0.995)
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            plt.ion()

    def _plot(_=None, *, reset_range: bool = False):
        sid = dropdown.value
        if sid is None:
            return
        model_params = _effective_model_params(
            load_model_params(scenario_id=sid),
            alpha=alpha_sl.value,
            beta=beta_sl.value,
            threshold_vel=threshold_vel_sl.value,
            threshold_accel=threshold_accel_sl.value,
            window_top_k=top_k_sl.value,
            fusion_threshold=fusion_sl.value,
        )
        stride_hours = float(_pipeline_params.model_window_size_hours - _pipeline_params.window_overlap_hours)

        df_sid = df[df[scenario_col] == sid].sort_values(time_col).copy()
        fit_df = df_sid[df_sid[split_col] == fit_value].copy()
        pred_df = df_sid[df_sid[split_col] == pred_value].copy()
        incidents = get_incident_spans(df_sid, label_col=label_col, time_col=time_col, normal_label=normal_label) if label_col in df_sid.columns else []
        with out:
            clear_output(wait=True)
            if fit_df.empty or pred_df.empty:
                print(f"Scenario {sid}: missing fit or pred split.")
                return

            if models is not None:
                _prefit_model = models.get(sid)
            elif sid in _api_replay_fit_cache:
                _prefit_model = _api_replay_fit_cache[sid]
            else:
                _prefit_model = AnomalyModel(is_cyclic=False)
                _prefit_model.params = model_params
                _prefit_model.params.baseline_scaler = "standard"
                if hasattr(_prefit_model._backend, "baseline_scaler"):
                    _prefit_model._backend.baseline_scaler = "standard"
                _prefit_model.fit(df_to_timeseries(fit_df, time_col=time_col))
                _api_replay_fit_cache[sid] = _prefit_model

            api_pred_df = simulate_api_replay_one_scenario(
                fit_df=fit_df,
                pred_df=pred_df,
                model=_prefit_model,
                sensor_id=f"sensor_{sid}",
                baseline_scaler=None,
                model_params_override=model_params,
                alert_params=_alert_params,
                time_col=time_col,
            )
            api_fit_df = simulate_api_replay_one_scenario(
                fit_df=fit_df,
                pred_df=fit_df,
                model=_prefit_model,
                sensor_id=f"sensor_{sid}",
                baseline_scaler=None,
                model_params_override=model_params,
                alert_params=_alert_params,
                time_col=time_col,
            ) if split_view_btn.value in ("fit", "both") else pd.DataFrame()
            api_plot_df = _with_split_view(split_view_btn.value, api_pred_df, api_fit_df)

            api_diag = diagnose_replay_against_incidents(api_pred_df, incidents, tolerance_hours=2.0)

            replay_state["df"] = api_plot_df.copy()
            replay_state["api_pred_df"] = api_pred_df.copy()
            replay_state["diagnostics"] = {"api": api_diag}
            replay_state["incidents"] = incidents
            replay_state["model_params"] = model_params
            replay_state["scenario_id"] = sid
            ui.replay_df = api_plot_df.copy()

            plot_state = _compute_replay_plot_state(api_plot_df, use_index=False, alert_params=_alert_params) if not api_plot_df.empty else None
            info_html.value = _build_replay_info_html(
                scenario_id=sid,
                show=split_view_btn.value,
                model_params=model_params,
                pipeline_params=_pipeline_params,
                alert_params=_alert_params,
                stride_hours=stride_hours,
                fit_df=fit_df,
                pred_df=pred_df,
                api_pred_df=api_pred_df,
                api_diag=api_diag,
                plot_state=plot_state,
            )

            if api_plot_df.empty:
                print("Replay returned no windows.")
                return

            plt.ioff()
            use_index = (x_toggle.value == "Index")
            height_ratios = [2.6, 1.8, 1.4, 1.5, 1.8, 3.0]
            fig, axes = plt.subplots(6, 1, figsize=(18, 28), sharex=True, gridspec_kw={"height_ratios": height_ratios})
            _plot_replay_column(axes, api_plot_df, incidents=incidents, use_index=use_index, split_view_value=split_view_btn.value, model_params=model_params, alert_params=_alert_params, title="API/Test replay", anomaly_fill_color=anomaly_fill_color, time_col=time_col)
            fig.suptitle(f"Scenario {sid} — API replay (fit-trained, {split_view_btn.value} windows)", y=0.995)
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            plt.ion()

    def _export_all_defaults(_=None):
        export_root = Path(export_dir) / "api_replay"
        export_root.mkdir(parents=True, exist_ok=True)
        template = {
            "show": default_state.get("show", "pred"),
            "x_axis": default_state.get("x_axis", default_x_axis if default_x_axis in ("Time", "Index") else "Time"),
            "alpha": float(default_state.get("alpha", default_state.get("alpha_vel", default_state.get("alpha_accel", default_model_params.alpha_vel)))),
            "beta": float(default_state.get("beta", default_state.get("beta_vel", default_state.get("beta_accel", default_model_params.beta_vel)))),
            "threshold_vel": float(default_state.get("threshold_vel", default_model_params.threshold_vel)),
            "threshold_accel": float(default_state.get("threshold_accel", default_model_params.threshold_accel)),
        }
        exported = 0
        errors: list[str] = []
        for scenario_id in scenario_options:
            try:
                group_params = load_model_params(scenario_id=scenario_id)
                replay = _run_replay(
                    scenario_id,
                    show=template["show"],
                    alpha=template["alpha"],
                    beta=template["beta"],
                    threshold_vel=template["threshold_vel"],
                    threshold_accel=template["threshold_accel"],
                    window_top_k=int(group_params.window_top_k),
                    fusion_threshold=float(group_params.fusion_threshold),
                )
                fig = _build_api_figure(
                    api_plot_df=replay["api_plot_df"],
                    incidents=replay["incidents"],
                    use_index=template["x_axis"] == "Index",
                    show=template["show"],
                    model_params=replay["model_params"],
                    alert_params=replay["alert_params"],
                    scenario_id=scenario_id,
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

    def _on_scenario_change(change):
        if change["name"] != "value" or change["new"] is None:
            return
        _apply_group_defaults_for_scenario(change["new"])
        _plot(reset_range=True)

    def _on_slider_change(change):
        if change["name"] != "value" or interaction_state["applying_defaults"]:
            return
        _plot()

    plot_btn.on_click(_plot)
    export_btn.on_click(_export_all_defaults)
    dropdown.observe(_on_scenario_change, names="value")
    split_view_btn.observe(lambda c: _plot(reset_range=True) if c["name"] == "value" else None, names="value")
    for ctl in (alpha_sl, beta_sl, threshold_vel_sl, threshold_accel_sl, top_k_sl, fusion_sl):
        ctl.observe(_on_slider_change, names="value")
    x_toggle.observe(lambda c: _render_current_plot() if c["name"] == "value" else None, names="value")
    _apply_group_defaults_for_scenario(initial_scenario_id)
    ui = widgets.VBox([
        widgets.HBox([dropdown, x_toggle, split_view_btn, plot_btn, export_btn]),
        widgets.HBox([alpha_sl, beta_sl, threshold_vel_sl, threshold_accel_sl, top_k_sl, fusion_sl]),
        group_html,
        info_html,
        export_status,
        out,
    ])
    ui.replay_df = replay_state["df"]
    ui.compare_replay_df = pd.DataFrame()
    ui.get_replay_df = lambda: replay_state["df"].copy()
    ui.get_compare_replay_df = lambda: pd.DataFrame()
    ui.get_replay_diagnostics = lambda: dict(replay_state["diagnostics"])
    _plot()
    return ui, ui.get_replay_df


create_api_replay_widget_compare = create_api_replay_widget_ui


def create_api_replay_widget(
    full_df,
    scenario_col="scenario_id",
    split_col="split",
    fit_value="fit",
    pred_value="pred",
    time_col="sampled_at",
    label_col="label",
    normal_label="normal",
    default_x_axis="Time",
    default_state=None,
    export_dir=DEFAULT_WIDGET_EXPORT_DIR,
    models=None,
):
    """Widget that mirrors the actual API lifecycle.

    Parameters
    ----------
    models :
        Optional ``{scenario_id: AnomalyModel}`` from ``load_fitted_models()``.
        When provided the widget skips the internal fit step. Pass ``None``
        (default) to keep fit-from-data behaviour.
    """
    return create_api_replay_widget_ui(
        full_df=full_df,
        scenario_col=scenario_col,
        split_col=split_col,
        fit_value=fit_value,
        pred_value=pred_value,
        time_col=time_col,
        label_col=label_col,
        normal_label=normal_label,
        default_x_axis=default_x_axis,
        default_state=default_state,
        export_dir=export_dir,
        models=models,
    )
