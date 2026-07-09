"""Grouped-residual detector parameters (contracts + YAML loaders).

Defines this detector's tunable parameter contracts (``ModelParams``,
``AlertParams``) and the loaders that build them from YAML. Model- and
alert-specific hyperparameters live alongside this file in
``grouped_residual/hyperparameters/``. The shared pipeline (windowing) config
lives in ``model/shared/`` and is loaded via ``model.shared.config``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from ..shared.config import load_yaml_or_empty
from ..shared.scenario_groups import get_scenario_group_key

_HYPERPARAMS_DIR = Path(__file__).resolve().parent / "hyperparameters"
DEFAULT_PARAMS_PATH = _HYPERPARAMS_DIR / "norm_model_hyperparams.yaml"
DEFAULT_ALERT_PARAMS_PATH = _HYPERPARAMS_DIR / "alert_hyperparams.yaml"


class ModelParams(BaseModel):
    """Grouped-residual detector and fusion hyperparameters loaded from YAML."""

    model_config = {"protected_namespaces": ()}

    alpha_vel: float = 1.52
    alpha_accel: float = 1.52
    beta_vel: float = 2.45
    beta_accel: float = 2.45
    threshold_vel: float = 3.0
    threshold_accel: float = 3.0
    baseline_scaler: str = "standard"

    window_top_k: int = 6

    model_window_size_hours: float = 2.0
    window_overlap_hours: float = 1.0

    fusion_threshold: float = 0.5


class AlertParams(BaseModel):
    """State-machine thresholds and holdback settings for alerting."""

    individual_alert_mode: Literal["exclusive", "shared"] = "exclusive"
    relative_threshold: float = 0.20
    min_channel_delta: float = 0.0
    min_cooldown_windows: int = 2
    reset_batches_below_zero: int = 12
    confirmation_count: int = 2
    confirmation_window: int = 3
    inter_alert_cooldown_windows: int = 0
    group_relative_threshold: float = 0.20
    group_min_cooldown_windows: int = 2
    group_reset_batches_below_zero: int = 12
    group_confirmation_count: int = 2
    group_confirmation_window: int = 3
    channel_to_group_holdback_windows: int = 2
    group3_to_group6_holdback_windows: int = 2
    suppress_lower_priority_during_group_candidate: bool = True
    suppress_group3_during_group6_candidate: bool = True
    enable_group6_alerts: bool = False


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

    data = load_yaml_or_empty(path)
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


def load_alert_params(path: Path = DEFAULT_ALERT_PARAMS_PATH) -> AlertParams:
    """Load alert engine hyperparameters from YAML. Falls back to defaults if file missing."""
    return AlertParams(**load_yaml_or_empty(path))
