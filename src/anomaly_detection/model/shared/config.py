"""Shared replay-window (pipeline) configuration loader.

The pipeline windowing config is shared across every detector, so both the
baseline and grouped-residual detectors load it from here. Also hosts
``load_yaml_or_empty``, the read-with-fallback primitive every ``params.py``
YAML loader in ``model/`` is built on.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .interface import PipelineParams

DEFAULT_PIPELINE_PARAMS_PATH = Path(__file__).resolve().parent / "pipeline_hyperparams.yaml"


def load_yaml_or_empty(path: Path) -> dict:
    """Read a YAML file into a dict, or ``{}`` if the file is missing or empty."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_pipeline_params(path: Path = DEFAULT_PIPELINE_PARAMS_PATH) -> PipelineParams:
    """Load pipeline hyperparameters from YAML. Falls back to defaults if file missing."""
    return PipelineParams(**load_yaml_or_empty(path))
