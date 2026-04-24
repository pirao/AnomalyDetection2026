"""Per-sensor anomaly detection pipeline on raw RMS channels.

The preprocessing contract is intentionally small:

1. Convert ``TimeSeries`` -> DataFrame.
2. Clip gross RMS spikes.
3. Trust ``uptime=True`` as the operational gating signal.
4. Fit / score per-axis baselines directly on raw RMS channels.

The replay/helper code defines the outer 2 h batch with 1 h stride; this model
scores each submitted batch once and does not create a second inner window.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .baselines import apply_norm_scores, fit_norm_baselines
from .preprocessing import clip_rms_spikes

from .interface import TimeSeries

_SCENARIO_ID = 0

_VEL_RAW = ["vel_rms_x", "vel_rms_y", "vel_rms_z"]
_ACCEL_RAW = ["accel_rms_x", "accel_rms_y", "accel_rms_z"]
_DEFAULT_CADENCE_MINUTES = 10.0


class SensorModel:
    """Per-axis raw-RMS anomaly detector for a single sensor."""

    def __init__(self, is_cyclic: bool = False, baseline_scaler: str = "standard") -> None:
        if baseline_scaler != "standard":
            raise ValueError("baseline_scaler must be 'standard'")
        self.is_cyclic = is_cyclic
        self.baseline_scaler = baseline_scaler
        self._state: dict[str, Any] = {}

    @staticmethod
    def _to_df(samples: TimeSeries) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "sampled_at": p.timestamp,
                    "uptime": p.uptime,
                    "vel_rms_x": p.vel_x,
                    "vel_rms_y": p.vel_y,
                    "vel_rms_z": p.vel_z,
                    "accel_rms_x": p.acc_x,
                    "accel_rms_y": p.acc_y,
                    "accel_rms_z": p.acc_z,
                    "scenario_id": _SCENARIO_ID,
                }
                for p in sorted(samples.data, key=lambda x: x.timestamp)
            ]
        )

    @staticmethod
    def _build_masks(df: pd.DataFrame) -> pd.DataFrame:
        uptime = df["uptime"].astype(bool)
        df["global_mask_vel"] = ~uptime
        df["global_mask_accel"] = ~uptime
        return df

    def fit(self, samples: TimeSeries) -> tuple[float, float]:
        """Fit the detection pipeline on healthy data."""
        df = self._to_df(samples)
        df = clip_rms_spikes(df, vel_threshold=100, accel_threshold=10)
        df = self._build_masks(df)

        fit_df = df[df["uptime"].astype(bool)].copy()
        fit_df["global_mask_vel"] = False
        fit_df["global_mask_accel"] = False

        score_baselines = fit_norm_baselines(
            df=fit_df,
            vel_cols=_VEL_RAW,
            accel_cols=_ACCEL_RAW,
            scaler=self.baseline_scaler,
        )

        self._state = {
            "score_baselines": score_baselines,
        }

        vel_vals = fit_df[_VEL_RAW[0]].dropna()
        mean = float(vel_vals.mean()) if not vel_vals.empty else 1.0
        std = float(vel_vals.std(ddof=1)) if len(vel_vals) > 1 else 1e-3
        return round(mean, 3), round(max(std, 1e-3), 3)

    def _score_df(self, samples: TimeSeries) -> pd.DataFrame:
        """Run the full preprocessing/scoring pipeline and return the scored DataFrame."""
        df = self._to_df(samples)
        df = clip_rms_spikes(df, vel_threshold=100, accel_threshold=10)
        df = self._build_masks(df)

        return apply_norm_scores(
            df=df,
            baselines=self._state["score_baselines"],
            vel_cols=_VEL_RAW,
            accel_cols=_ACCEL_RAW,
        )

    @staticmethod
    def _sigmoid(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        z = np.clip(alpha * (x - beta), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _residuals_from_dnorm(d_vals: np.ndarray, dnorm_ref: float) -> np.ndarray:
        return np.asarray(d_vals, dtype=float) - float(dnorm_ref)

    @staticmethod
    def _topk_mean(scores: np.ndarray, top_k: int) -> float:
        top_k = max(int(top_k), 1)
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return 0.0

        top = np.sort(scores)[::-1][:top_k]
        if top.size < top_k:
            top = np.pad(top, (0, top_k - top.size), constant_values=0.0)
        return float(top.mean())

    @staticmethod
    def _infer_cadence_minutes(
        timestamps: pd.Series,
        *,
        default_minutes: float = _DEFAULT_CADENCE_MINUTES,
    ) -> float:
        ts = pd.to_datetime(timestamps, errors="coerce")
        ts = ts.dropna().sort_values().drop_duplicates()
        if len(ts) < 2:
            return default_minutes

        deltas = ts.diff().dt.total_seconds().div(60.0).dropna()
        deltas = deltas[deltas > 0]
        if deltas.empty:
            return default_minutes

        mode = deltas.mode()
        cadence = float(mode.iloc[0]) if not mode.empty else float(deltas.median())
        return cadence if cadence > 0 else default_minutes

    @classmethod
    def _infer_expected_samples_per_window(
        cls,
        timestamps: pd.Series,
        *,
        window_hours: float,
        default_minutes: float = _DEFAULT_CADENCE_MINUTES,
    ) -> int:
        cadence_minutes = cls._infer_cadence_minutes(
            timestamps,
            default_minutes=default_minutes,
        )
        expected = int(round((window_hours * 60.0) / cadence_minutes))
        return max(expected, 1)

    def _modality_channel_details(
        self,
        df: pd.DataFrame,
        *,
        cols: list[str],
        alpha: float,
        beta: float,
        threshold: float,
        window_top_k: int,
        expected_samples_per_window: int,
    ) -> dict[str, dict[str, float | int | bool]]:
        details: dict[str, dict[str, float | int | bool]] = {}
        if df.empty:
            return details

        batch_rows = int(len(df))
        for col in cols:
            d_col = f"d_{col}"
            if d_col not in df.columns:
                continue

            d_vals = df[d_col].fillna(0.0).to_numpy(dtype=float)
            residual_vals = self._residuals_from_dnorm(d_vals, dnorm_ref=threshold)
            scores = self._sigmoid(residual_vals, alpha=alpha, beta=beta)
            window_score_raw = self._topk_mean(scores, top_k=window_top_k)
            window_score = float(window_score_raw)

            max_residual = float(residual_vals.max()) if residual_vals.size else 0.0
            max_score = float(scores.max()) if scores.size else 0.0

            details[col] = {
                "batch_rows": batch_rows,
                "topk_contributors": int(min(len(scores), max(int(window_top_k), 1))),
                "occupancy_fixed": float(window_score_raw),
                "occupancy_observed": float(window_score),
                "occupancy_raw": float(window_score_raw),
                "max_residual": max_residual,
                "max_residual_active": max_residual if window_score_raw > 0.0 else 0.0,
                "max_score": max_score,
                "is_active": bool(window_score_raw > 0.0),
            }

        return details

    def _score_batch(
        self,
        samples: TimeSeries,
        *,
        model_window_size_hours: float,
        expected_samples_per_window: int | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
        """Shared preprocessing: score df → uptime-gated vel/accel splits.

        Returns (scored_df, up_vel, up_accel, expected_samples_per_window).
        Sigmoid, flagging, and persistency are applied later in _modality_channel_details.
        """
        scored = self._score_df(samples).copy()
        scored["sampled_at"] = pd.to_datetime(scored["sampled_at"], errors="coerce")
        scored = scored.dropna(subset=["sampled_at"]).sort_values("sampled_at").reset_index(drop=True)

        if expected_samples_per_window is None:
            expected_samples_per_window = self._infer_expected_samples_per_window(
                scored["sampled_at"],
                window_hours=model_window_size_hours,
            )

        up_vel = (
            scored.loc[~scored["global_mask_vel"]]
            .copy()
            .sort_values("sampled_at")
            .reset_index(drop=True)
        )
        up_accel = (
            scored.loc[~scored["global_mask_accel"]]
            .copy()
            .sort_values("sampled_at")
            .reset_index(drop=True)
        )
        return scored, up_vel, up_accel, int(expected_samples_per_window)

    def predict_batch_details(
        self,
        samples: TimeSeries,
        alpha_vel: float,
        alpha_accel: float,
        beta_vel: float,
        beta_accel: float,
        threshold_vel: float,
        threshold_accel: float,
        window_top_k: int,
        model_window_size_hours: float,
        window_overlap_hours: float,
        fusion_threshold: float,
        expected_samples_per_window: int | None = None,
    ) -> dict[str, Any]:
        """Score one submitted API batch once and return rich diagnostics.

        Notes
        -----
        ``window_overlap_hours`` is kept in the signature for compatibility with
        the shared hyperparameter object, but it is not used here because the
        batch has already been defined by the outer replay/API logic.
        """
        del window_overlap_hours

        scored, up_vel, up_accel, expected_samples_per_window = self._score_batch(
            samples,
            model_window_size_hours=model_window_size_hours,
            expected_samples_per_window=expected_samples_per_window,
        )

        return self._build_batch_details_from_scored(
            scored=scored,
            up_vel=up_vel,
            up_accel=up_accel,
            timestamp=samples.data[-1].timestamp,
            alpha_vel=alpha_vel,
            alpha_accel=alpha_accel,
            beta_vel=beta_vel,
            beta_accel=beta_accel,
            threshold_vel=threshold_vel,
            threshold_accel=threshold_accel,
            window_top_k=window_top_k,
            fusion_threshold=fusion_threshold,
            expected_samples_per_window=expected_samples_per_window,
        )

    def _build_batch_details_from_scored(
        self,
        *,
        scored: pd.DataFrame,
        up_vel: pd.DataFrame,
        up_accel: pd.DataFrame,
        timestamp,
        alpha_vel: float,
        alpha_accel: float,
        beta_vel: float,
        beta_accel: float,
        threshold_vel: float,
        threshold_accel: float,
        window_top_k: int,
        fusion_threshold: float,
        expected_samples_per_window: int,
    ) -> dict[str, Any]:
        del scored

        vel_channel_details = self._modality_channel_details(
            up_vel,
            cols=_VEL_RAW,
            alpha=alpha_vel,
            beta=beta_vel,
            threshold=threshold_vel,
            window_top_k=window_top_k,
            expected_samples_per_window=expected_samples_per_window,
        )
        accel_channel_details = self._modality_channel_details(
            up_accel,
            cols=_ACCEL_RAW,
            alpha=alpha_accel,
            beta=beta_accel,
            threshold=threshold_accel,
            window_top_k=window_top_k,
            expected_samples_per_window=expected_samples_per_window,
        )

        vel_occupancy_fixed = max(
            (float(v["occupancy_raw"]) for v in vel_channel_details.values()),
            default=0.0,
        )
        accel_occupancy_fixed = max(
            (float(v["occupancy_raw"]) for v in accel_channel_details.values()),
            default=0.0,
        )
        vel_occupancy_observed = max(
            (float(v["occupancy_observed"]) for v in vel_channel_details.values()),
            default=0.0,
        )
        accel_occupancy_observed = max(
            (float(v["occupancy_observed"]) for v in accel_channel_details.values()),
            default=0.0,
        )

        vel_occupancy = vel_occupancy_observed
        accel_occupancy = accel_occupancy_observed
        occupancy_score = max(vel_occupancy, accel_occupancy)
        occupancy_score_fixed = max(vel_occupancy_fixed, accel_occupancy_fixed)
        anomaly_status = bool(occupancy_score >= fusion_threshold)

        active_modalities: list[str] = []
        active_channels: list[str] = []
        if vel_occupancy >= fusion_threshold:
            active_modalities.append("vel")
            active_channels.extend(
                col
                for col, info in vel_channel_details.items()
                if float(info["occupancy_observed"]) >= fusion_threshold
            )
        if accel_occupancy >= fusion_threshold:
            active_modalities.append("accel")
            active_channels.extend(
                col
                for col, info in accel_channel_details.items()
                if float(info["occupancy_observed"]) >= fusion_threshold
            )

        max_residual_active = 0.0
        if active_channels:
            all_details = {**vel_channel_details, **accel_channel_details}
            max_residual_active = max(float(all_details[col]["max_residual"]) for col in active_channels)

        return {
            "timestamp": timestamp,
            "expected_samples_per_window": int(expected_samples_per_window),
            "batch_rows_vel": int(len(up_vel)),
            "batch_rows_accel": int(len(up_accel)),
            "vel_occupancy": float(vel_occupancy),
            "accel_occupancy": float(accel_occupancy),
            "vel_occupancy_fixed": float(vel_occupancy_fixed),
            "accel_occupancy_fixed": float(accel_occupancy_fixed),
            "vel_occupancy_observed": float(vel_occupancy_observed),
            "accel_occupancy_observed": float(accel_occupancy_observed),
            "occupancy_score": float(occupancy_score),
            "occupancy_score_fixed": float(occupancy_score_fixed),
            "occupancy_score_observed": float(occupancy_score),
            "alert_score": float(occupancy_score),
            "max_residual_active": float(max_residual_active),
            "anomaly_status": anomaly_status,
            "active_modalities": active_modalities,
            "active_channels": active_channels,
            "vel_channel_details": vel_channel_details,
            "accel_channel_details": accel_channel_details,
        }

    def predict(
        self,
        samples: TimeSeries,
        alpha_vel: float,
        alpha_accel: float,
        beta_vel: float,
        beta_accel: float,
        threshold_vel: float,
        threshold_accel: float,
        window_top_k: int,
        model_window_size_hours: float,
        window_overlap_hours: float,
        fusion_threshold: float,
    ) -> tuple[bool, float]:
        """Return one anomaly decision for one submitted API batch."""
        details = self.predict_batch_details(
            samples,
            alpha_vel=alpha_vel,
            alpha_accel=alpha_accel,
            beta_vel=beta_vel,
            beta_accel=beta_accel,
            threshold_vel=threshold_vel,
            threshold_accel=threshold_accel,
            window_top_k=window_top_k,
            model_window_size_hours=model_window_size_hours,
            window_overlap_hours=window_overlap_hours,
            fusion_threshold=fusion_threshold,
        )
        return bool(details["anomaly_status"]), float(details["occupancy_score"])
