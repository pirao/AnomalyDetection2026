from pathlib import Path

import numpy as np
import yaml

from .interface import ModelParams, PipelineParams, PredictOutput, TimeSeries, Weights

# Model-specific hyperparameters live alongside this file in
# baseline/hyperparameters/. The pipeline (windowing) config is shared across
# every model, so it lives one level up in model/shared/.
_HYPERPARAMS_DIR = Path(__file__).resolve().parent / "hyperparameters"
_SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
DEFAULT_PARAMS_PATH = _HYPERPARAMS_DIR / "model_hyperparams.yaml"
DEFAULT_PIPELINE_PARAMS_PATH = _SHARED_DIR / "pipeline_hyperparams.yaml"


def load_model_params(path: Path = DEFAULT_PARAMS_PATH) -> ModelParams:
    """Load model hyperparameters from YAML. Falls back to ModelParams defaults if file missing."""
    if not path.exists():
        return ModelParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ModelParams(**data)


def load_pipeline_params(path: Path = DEFAULT_PIPELINE_PARAMS_PATH) -> PipelineParams:
    """Load pipeline hyperparameters from YAML. Falls back to PipelineParams defaults if file missing."""
    if not path.exists():
        return PipelineParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return PipelineParams(**data)


class AnomalyModel:
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
