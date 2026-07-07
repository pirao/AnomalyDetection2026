"""Render a GIF of the deployed anomaly detector serving a live sensor stream.

This is the *deployment* demo: it drives the FastAPI service exactly as production
does - replaying one sensor's ``pred`` split in 2h windows on a 1h stride through
``POST /predict`` - while the service serves the model currently promoted to
``@production`` in the MLflow Registry (loaded once at startup, **no runtime fit**).

The figure shows the six raw channels (3 components x velocity / acceleration) for
the sensor, with the **fit (training) split** drawn first as context and the
**pred split** revealed window by window as it streams. No masking is applied:
both uptime and downtime are shown (downtime reads near zero - those are the flat
dips). The labelled incident window is shaded and a red line marks the timestamp
where the service raised an alert; the caption names the served registry version.

Lives in ``analysis.mlflow`` next to the registry code it demonstrates. Run in an
environment with the analysis/notebook stack (mlflow + matplotlib), e.g.
``uv run --extra notebooks``:

    python -m analysis.mlflow.deploy_demo --sensor 9
    python -m analysis.mlflow.deploy_demo --sensor 9 --http http://localhost:8000

Output: ``reports/figures/mlflow/deploy_demo.gif``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# src/analysis/mlflow/deploy_demo.py -> repo root is 3 parents up; src is 2 up.
_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sample_processing.serving.registry import REGISTERED_MODEL_NAME  # noqa: E402

_CANONICAL_DATA_DIR = _REPO / "data" / "raw"
_LEGACY_DATA_DIR = _REPO / "data"
DATA_DIR = (
    _CANONICAL_DATA_DIR
    if any(_CANONICAL_DATA_DIR.glob("sensor_data_fit_*.parquet"))
    else _LEGACY_DATA_DIR
)

_CANONICAL_LABELS_PATH = _REPO / "data" / "raw" / "labels" / "incidents.yaml"
_LEGACY_LABELS_PATH = _REPO / "labels" / "incidents.yaml"
LABELS_PATH = (
    _CANONICAL_LABELS_PATH
    if _CANONICAL_LABELS_PATH.exists()
    else _LEGACY_LABELS_PATH
)
DEFAULT_OUT = _REPO / "reports" / "figures" / "mlflow" / "deploy_demo.gif"

WINDOW = timedelta(hours=2.0)
STRIDE = timedelta(hours=1.0)
MAX_FRAMES = 60  # cap animation frames so the GIF stays short

# Component (row) x modality (column) layout for the 3x2 channel grid.
COMPONENTS = ["x", "y", "z"]
MODALITIES = [("vel_rms", "Velocity RMS"), ("accel_rms", "Acceleration RMS")]


# -- Data --------------------------------------------------------------------


def _load(sensor: int):
    fit = pd.read_parquet(DATA_DIR / f"sensor_data_fit_{sensor}.parquet").sort_values("sampled_at").reset_index(drop=True)
    pred = pd.read_parquet(DATA_DIR / f"sensor_data_pred_{sensor}.parquet").sort_values("sampled_at").reset_index(drop=True)
    raw = yaml.safe_load(LABELS_PATH.read_text()).get(sensor, []) or []
    windows = [
        (pd.Timestamp(w["start"]), pd.Timestamp(w["end"]))
        for w in raw
        if w.get("start") and w.get("end") and w["start"] < w["end"]
    ]
    return fit, pred, windows


def _payload(batch: pd.DataFrame) -> list[dict]:
    return [
        {
            "timestamp": r.sampled_at.isoformat(),
            "uptime": bool(r.uptime),
            "vel_rms_x": float(r.vel_rms_x),
            "vel_rms_y": float(r.vel_rms_y),
            "vel_rms_z": float(r.vel_rms_z),
            "accel_rms_x": float(r.accel_rms_x),
            "accel_rms_y": float(r.accel_rms_y),
            "accel_rms_z": float(r.accel_rms_z),
        }
        for r in batch.itertuples(index=False)
    ]


def _windows(pred: pd.DataFrame):
    """Yield (window_end, batch_df) over the pred series, 2h window / 1h stride."""
    cur = pred["sampled_at"].iloc[0]
    end = pred["sampled_at"].iloc[-1]
    while cur <= end:
        batch = pred[(pred["sampled_at"] >= cur) & (pred["sampled_at"] < cur + WINDOW)]
        if not batch.empty:
            yield cur + WINDOW, batch
        cur += STRIDE


# -- Serving -----------------------------------------------------------------


def _served_version() -> str:
    """Resolve the @production version for the caption (best effort)."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(f"sqlite:///{(_REPO / 'mlflow.db').as_posix()}")
        mv = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")
        return f"v{mv.version}"
    except Exception:  # noqa: BLE001 - caption is cosmetic
        return "@production"


def stream(sensor: int, pred: pd.DataFrame, http: str | None) -> list[pd.Timestamp]:
    """Replay the sensor through the deployed API; return alert timestamps."""
    sensor_id = f"sensor_{sensor}"
    alert_ts: list[pd.Timestamp] = []

    def consume(call) -> None:
        for _end, batch in _windows(pred):
            resp = call(_payload(batch))
            if resp.get("alert"):
                alert_ts.append(pd.Timestamp(resp["timestamp"]))

    if http:
        import requests

        requests.get(f"{http}/health", timeout=5).raise_for_status()

        def call(data):
            r = requests.post(
                f"{http}/predict", json={"sensor_id": sensor_id, "data": data}, timeout=60
            )
            r.raise_for_status()
            return r.json()

        consume(call)
    else:
        from fastapi.testclient import TestClient

        import sample_processing.api.main as m

        with TestClient(m.app) as client:  # lifespan loads @production
            if not m._bundle:
                raise SystemExit(
                    "Registry @production bundle did not load. Run in an env with "
                    "mlflow installed and an @production alias set, or pass --http "
                    "to target a running server."
                )

            def call(data):
                r = client.post("/predict", json={"sensor_id": sensor_id, "data": data})
                r.raise_for_status()
                return r.json()

            consume(call)

    return alert_ts


# -- Animation ---------------------------------------------------------------


def render(
    sensor: int,
    fit: pd.DataFrame,
    pred: pd.DataFrame,
    incidents: list[tuple[pd.Timestamp, pd.Timestamp]],
    alerts: list[pd.Timestamp],
    served: str,
    out_path: Path,
    fps: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from analysis.plotting.style import set_plot_style

    set_plot_style()  # repo-wide Arial rcParams

    nf, npd = len(fit), len(pred)
    x_fit = pd.to_datetime(fit["sampled_at"], utc=True)
    x_pred = pd.to_datetime(pred["sampled_at"], utc=True)
    boundary = x_fit.iloc[-1]
    pred_ns = x_pred.astype("int64").to_numpy()  # UTC ns for searchsorted in animation

    def _ns(ts) -> int:
        return pd.to_datetime(ts, utc=True).value

    inc_spans = list(incidents)  # already (Timestamp, Timestamp) tuples
    alert_x = list(alerts)       # already Timestamps
    pred_start = x_pred.iloc[0]  # first pred timestamp; gap = [boundary, pred_start]

    # Subsample window-ends to keep the GIF short; reveal still shows all data so far.
    win_ends = [we for we, _ in _windows(pred)]
    step = max(1, len(win_ends) // MAX_FRAMES)
    frame_ends = win_ends[::step]
    frame_ends += [frame_ends[-1]] * max(1, fps * 2)  # hold on the final result

    fig, axes = plt.subplots(3, 2, figsize=(14, 9.5), sharex=True)
    fig.suptitle(
        f"Deployed anomaly detector - live stream of sensor_{sensor}",
        fontsize=20, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.92,
        f"served by {REGISTERED_MODEL_NAME}@production ({served})  |  MLflow Registry -> FastAPI /predict  |  pre-fitted, no runtime training",
        ha="center", fontsize=12, style="italic", color="0.3",
    )

    panels = []  # (pred_values, pred_line, now_line, [alert_lines])
    for r, comp in enumerate(COMPONENTS):
        for c, (prefix, col_title) in enumerate(MODALITIES):
            ax = axes[r][c]
            col = f"{prefix}_{comp}"
            fit_vals = fit[col].to_numpy()
            pred_vals = pred[col].to_numpy()
            ax.set_xlim(x_fit.iloc[0], x_pred.iloc[-1])
            both = np.concatenate([fit_vals, pred_vals])
            lo, hi = float(np.nanmin(both)), float(np.nanmax(both))
            pad = (hi - lo) * 0.1 or 1.0
            ax.set_ylim(lo - pad, hi + pad)
            ax.grid(alpha=0.3)
            for i0, i1 in inc_spans:
                ax.axvspan(i0, i1, color="orange", alpha=0.18, zorder=0)
            ax.axvline(boundary, color="0.5", ls=":", lw=1.2)  # fit | pred divider
            if pred_start > boundary:
                ax.axvspan(boundary, pred_start, color="0.92", alpha=1.0, zorder=0)
            ax.plot(x_fit, fit_vals, color="0.6", lw=0.7)       # fit context (static)
            (pred_line,) = ax.plot([], [], color="tab:blue" if c == 0 else "tab:green", lw=0.8)
            now = ax.axvline(boundary, color="0.35", ls="--", lw=1.0)
            alines = [ax.axvline(ax_x, color="red", lw=1.6, visible=False) for ax_x in alert_x]
            if r == 0:
                ax.set_title(col_title, fontsize=16, pad=12)
                # The fit|pred divider and the "now" cursor are explained in the legend,
                # so the only text left inside the grey span is the "no data" marker.
                if pred_start > boundary:
                    y_lo, y_hi = ax.get_ylim()
                    gap_mid = boundary + (pred_start - boundary) / 2
                    ax.text(
                        gap_mid, y_hi - (y_hi - y_lo) * 0.04,
                        "no data", ha="center", va="top",
                        fontsize=9, color="0.5", style="italic",
                    )
            if c == 0:
                ax.set_ylabel(comp.upper(), fontsize=18, rotation=0, labelpad=18, va="center")
            panels.append((pred_vals, pred_line, now, alines))

    for c in range(2):
        axes[2][c].set_xlabel("timestamp", fontsize=12)
        loc = mdates.AutoDateLocator()
        axes[2][c].xaxis.set_major_locator(loc)
        axes[2][c].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    legend_handles = [
        Line2D([], [], color="0.6", lw=1.2, label="fit (training)"),
        Line2D([], [], color="tab:blue", lw=1.4, label="velocity (pred)"),
        Line2D([], [], color="tab:green", lw=1.4, label="acceleration (pred)"),
        Line2D([], [], color="0.5", ls=":", lw=1.2, label="fit | pred divider"),
        Line2D([], [], color="0.35", ls="--", lw=1.0, label="now — current stream position"),
        Patch(facecolor="0.92", label="no data (train/deploy gap)"),
        Patch(facecolor="orange", alpha=0.25, label="labelled incident"),
        Line2D([], [], color="red", lw=1.6, label="alert raised by API"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 0.008), framealpha=0.95)

    status = axes[0][1].text(
        0.985, 0.95, "", transform=axes[0][1].transAxes, ha="right", va="top",
        fontsize=11, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )
    fig.subplots_adjust(top=0.85, bottom=0.17, hspace=0.2, wspace=0.16, left=0.08, right=0.985)

    def update(i):
        fe = frame_ends[i]
        cnt = int(np.searchsorted(pred_ns, _ns(fe), side="right"))
        pos = x_pred.iloc[cnt - 1] if cnt > 0 else boundary
        for pred_vals, line, now, alines in panels:
            line.set_data(x_pred.iloc[:cnt], pred_vals[:cnt])
            now.set_xdata([pos, pos])
            for ax_x, al in zip(alert_x, alines):
                al.set_visible(pos >= ax_x)
        n_alerts = sum(1 for ax_x in alert_x if pos >= ax_x)
        flag = "  [!] ALERT" if n_alerts else ""
        status.set_text(f"stream: {pd.Timestamp(fe):%Y-%m-%d %H:%M}\nalerts: {n_alerts}{flag}")
        return ()

    ani = FuncAnimation(fig, update, frames=len(frame_ends), interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# -- CLI ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sensor", type=int, default=9, help="sensor/scenario id (default: 9)")
    ap.add_argument("--http", default=None, help="base URL of a running API, e.g. http://localhost:8000")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()

    fit, pred, incidents = _load(args.sensor)
    served = _served_version()
    print(f"streaming sensor_{args.sensor} through {'HTTP ' + args.http if args.http else 'in-process app'} "
          f"(serving {REGISTERED_MODEL_NAME}@production {served}) ...")
    alerts = stream(args.sensor, pred, args.http)
    print(f"  windows alerted: {len(alerts)} -> {[str(a) for a in alerts]}")
    in_window = [a for a in alerts if any(s <= a <= e for s, e in incidents)]
    print(f"  alerts inside a labelled incident window: {len(in_window)}/{len(alerts)}")
    render(args.sensor, fit, pred, incidents, alerts, served, args.out, args.fps)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
