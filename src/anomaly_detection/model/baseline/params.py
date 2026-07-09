"""Baseline detector parameters (contract + loader).

The pipeline (windowing) config is shared across detectors and lives in
``model/shared/config.py``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ..shared.config import load_yaml_or_empty

_HYPERPARAMS_DIR = Path(__file__).resolve().parent / "hyperparameters"
DEFAULT_PARAMS_PATH = _HYPERPARAMS_DIR / "model_hyperparams.yaml"


class ModelParams(BaseModel):
    """Baseline detector thresholds loaded from YAML or defaults."""

    z_threshold: int = 3
    window_anomaly_ratio: float = 0.2


def load_model_params(path: Path = DEFAULT_PARAMS_PATH) -> ModelParams:
    """Load model hyperparameters from YAML. Falls back to ModelParams defaults if file missing."""
    return ModelParams(**load_yaml_or_empty(path))
