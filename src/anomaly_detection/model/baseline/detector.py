"""Velocity-norm baseline detector used for MLflow comparison runs."""

from pathlib import Path

import numpy as np

from ..shared.interface import PredictOutput, TimeSeries, Weights
from .params import DEFAULT_PARAMS_PATH, load_model_params


class BaselineDetector:
    """Simple baseline detector using velocity norm z-scores."""

    def __init__(self, params_path: Path | None = None):
        self.weights = Weights()
        self.params = load_model_params(params_path or DEFAULT_PARAMS_PATH)

    def _featuring(self, samples: TimeSeries) -> np.ndarray:
        if not samples.data:
            return np.array([], dtype=float)

        ordered = sorted(samples.data, key=lambda p: p.timestamp)

        vel = np.array([(p.vel_x, p.vel_y, p.vel_z) for p in ordered], dtype=float)

        return np.linalg.norm(vel, axis=1)

    def fit(self, fitting_samples: TimeSeries) -> None:
        """Fit scalar mean and standard deviation on baseline samples."""
        X = self._featuring(fitting_samples)

        self.weights = Weights(
            fitted=True,
            mean=round(X.mean(), 3),
            std=round(X.std(), 3),
        )

    def _zscore(self, X: np.ndarray) -> np.ndarray:
        w = self.weights
        return np.abs((X - w.mean) / w.std)

    def _window_deviation_ratio(self, X: np.ndarray) -> float:
        z = self._zscore(X)
        anomalous_points = z > self.params.z_threshold
        return float(np.sum(anomalous_points) / len(z))

    def predict(self, samples: TimeSeries) -> PredictOutput:
        """Return whether a batch exceeds the configured deviation ratio."""
        if not self.weights.fitted:
            raise RuntimeError("Model not fitted")
        if not samples.data:
            raise ValueError("Cannot predict on empty TimeSeries")

        X = self._featuring(samples)
        deviation_ratio = self._window_deviation_ratio(X)
        is_anomalous = deviation_ratio >= self.params.window_anomaly_ratio

        return PredictOutput(
            anomaly_status=is_anomalous,
            timestamp=samples.data[-1].timestamp,
        )
