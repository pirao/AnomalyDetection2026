from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .interface import AlertParams, ModelParams, PipelineParams, PredictOutput, TimeSeries, Weights
from .scenario_groups import get_scenario_group_key
from .sensor_model import SensorModel

_HYPERPARAMS_DIR = Path(__file__).parent.parent / "hyperparameters"
DEFAULT_PARAMS_PATH = _HYPERPARAMS_DIR / "norm_model_hyperparams.yaml"
DEFAULT_PIPELINE_PARAMS_PATH = _HYPERPARAMS_DIR / "pipeline_hyperparams.yaml"
DEFAULT_ALERT_PARAMS_PATH = _HYPERPARAMS_DIR / "alert_hyperparams.yaml"


def load_model_params(
    path: Path = DEFAULT_PARAMS_PATH,
    *,
    group_key: str | None = None,
    scenario_id: object | None = None,
) -> ModelParams:
    """Load model hyperparameters from YAML. Falls back to defaults if file missing."""
    def _normalize_sigmoid_params(raw: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(raw)
        shared_alpha = normalized.pop("alpha", None)
        shared_beta = normalized.pop("beta", None)
        if shared_alpha is not None:
            normalized.setdefault("alpha_vel", shared_alpha)
            normalized.setdefault("alpha_accel", shared_alpha)
        if shared_beta is not None:
            normalized.setdefault("beta_vel", shared_beta)
            normalized.setdefault("beta_accel", shared_beta)
        return normalized

    if not path.exists():
        return ModelParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if "defaults" not in data and "groups" not in data:
        return ModelParams(**_normalize_sigmoid_params(data))

    effective_group = group_key
    if effective_group is None and scenario_id is not None:
        effective_group = get_scenario_group_key(scenario_id)

    merged = dict(data.get("defaults", {}))
    group_overrides = (data.get("groups", {}) or {}).get(effective_group or "", {})
    if isinstance(group_overrides, dict):
        merged.update(group_overrides)
    return ModelParams(**_normalize_sigmoid_params(merged))


def load_pipeline_params(path: Path = DEFAULT_PIPELINE_PARAMS_PATH) -> PipelineParams:
    """Load pipeline hyperparameters from YAML. Falls back to defaults if file missing."""
    if not path.exists():
        return PipelineParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return PipelineParams(**data)


def load_alert_params(path: Path = DEFAULT_ALERT_PARAMS_PATH) -> AlertParams:
    """Load alert engine hyperparameters from YAML. Falls back to defaults if file missing."""
    if not path.exists():
        return AlertParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return AlertParams(**data)


class AnomalyModel:
    def __init__(
        self,
        params_path: Path | None = None,
        is_cyclic: bool = False,
        scenario_id: object | None = None,
        group_key: str | None = None,
    ):
        self.weights = Weights()
        resolved_params_path = params_path or DEFAULT_PARAMS_PATH
        self.params_path = resolved_params_path
        self.params = load_model_params(
            resolved_params_path,
            scenario_id=scenario_id,
            group_key=group_key,
        )
        self._backend = SensorModel(
            is_cyclic=is_cyclic,
            baseline_scaler=self.params.baseline_scaler,
        )

    def fit(self, fitting_samples: TimeSeries) -> None:
        mean, std = self._backend.fit(fitting_samples)
        self.weights = Weights(fitted=True, mean=mean, std=std)
        if hasattr(self._backend, "weights"):
            self._backend.weights = self.weights

    def predict_detailed(self, samples: TimeSeries) -> "pd.DataFrame":
        """Return the per-sample scored DataFrame with d_* columns for diagnostics."""
        if not self.weights.fitted:
            raise RuntimeError("Model not fitted")
        import pandas as pd  # local import to keep module-level imports minimal

        return self._backend._score_df(samples)

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
