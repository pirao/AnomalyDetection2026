"""Grouped-residual detector facade around the scoring engine and YAML parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..shared.interface import PredictOutput, TimeSeries, Weights
from .params import DEFAULT_PARAMS_PATH, load_model_params
from .scoring import Scorer


class GroupedResidualDetector:
    """Scenario-aware anomaly detector backed by the grouped-residual scoring pipeline."""

    def __init__(
        self,
        params_path: Path | None = None,
        is_cyclic: bool = False,
        scenario_id: object | None = None,
        group_key: str | None = None,
    ):
        self.weights = Weights()
        resolved_params_path = params_path or DEFAULT_PARAMS_PATH
        # Stored as a string, not a Path: a fitted model is pickled and may be
        # unpickled on a different OS (fitted on Windows, served in a Linux
        # container). A pickled WindowsPath cannot be reconstructed on POSIX.
        self.params_path = str(resolved_params_path)
        self.params = load_model_params(
            resolved_params_path,
            scenario_id=scenario_id,
            group_key=group_key,
        )
        self._backend = Scorer(
            is_cyclic=is_cyclic,
            baseline_scaler=self.params.baseline_scaler,
        )

    def fit(self, fitting_samples: TimeSeries) -> None:
        """Fit baseline weights from normal samples for this model instance."""
        mean, std = self._backend.fit(fitting_samples)
        self.weights = Weights(fitted=True, mean=mean, std=std)
        if hasattr(self._backend, "weights"):
            self._backend.weights = self.weights

    def predict_batch_details(
        self,
        samples: TimeSeries,
        *,
        expected_samples_per_window: int | None = None,
    ) -> dict[str, Any]:
        """Return one rich diagnostic dictionary for one submitted API batch."""
        if not self.weights.fitted:
            raise RuntimeError("Model not fitted")
        if not samples.data:
            raise ValueError("Cannot predict on empty TimeSeries")

        return self._backend.predict_batch_details(
            samples,
            alpha_vel=self.params.alpha_vel,
            alpha_accel=self.params.alpha_accel,
            beta_vel=self.params.beta_vel,
            beta_accel=self.params.beta_accel,
            threshold_vel=self.params.threshold_vel,
            threshold_accel=self.params.threshold_accel,
            window_top_k=self.params.window_top_k,
            model_window_size_hours=self.params.model_window_size_hours,
            window_overlap_hours=self.params.window_overlap_hours,
            fusion_threshold=self.params.fusion_threshold,
            expected_samples_per_window=expected_samples_per_window,
        )

    def predict(self, samples: TimeSeries) -> PredictOutput:
        """Score one batch and return the compact output consumed by alerting."""
        if not self.weights.fitted:
            raise RuntimeError("Model not fitted")
        if not samples.data:
            raise ValueError("Cannot predict on empty TimeSeries")

        details = self.predict_batch_details(samples)
        all_channel_details = {
            **details["vel_channel_details"],
            **details["accel_channel_details"],
        }
        return PredictOutput(
            anomaly_status=bool(details["anomaly_status"]),
            timestamp=samples.data[-1].timestamp,
            occupancy_score=float(details["occupancy_score"]),
            alert_score=float(details["alert_score"]),
            mean_d_score=float(details["alert_score"]),
            active_channels=list(details["active_channels"]),
            active_modalities=list(details["active_modalities"]),
            channel_max_residual={
                col: float(info["max_residual"])
                for col, info in all_channel_details.items()
            },
        )
