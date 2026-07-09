"""Data-transfer contracts shared by every detector.

These types describe the data flowing into a model (`DataPoint`, `TimeSeries`),
generic fitted state (`Weights`), evaluation labels (`TrueIncident`), the batch
output every detector emits (`PredictOutput`), the alert-engine decision
(`AlertDecision`), and the shared replay-window config (`PipelineParams`). The
grouped-residual detector uses the full field set; the baseline detector fills
only the required fields and leaves the rest at their defaults. Detector-specific
tuning parameters live in each detector's ``params.py``.
"""

from datetime import datetime
from typing import Sequence

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    """Single timestamped sensor reading in the internal model format."""

    timestamp: datetime = Field(..., description="UTC datetime of the time the data point was collected")
    uptime: bool = Field(..., description="Whether the data point is during uptime")
    vel_x: float = Field(..., description="Vibration velocity component along the X axis")
    vel_y: float = Field(..., description="Vibration velocity component along the Y axis")
    vel_z: float = Field(..., description="Vibration velocity component along the Z axis")
    acc_x: float = Field(..., description="Vibration acceleration component along the X axis")
    acc_y: float = Field(..., description="Vibration acceleration component along the Y axis")
    acc_z: float = Field(..., description="Vibration acceleration component along the Z axis")


def reading_kwargs_from_row(row) -> dict:
    """Map a wire-format reading (``vel_rms_*``/``accel_rms_*``) to ``DataPoint`` kwargs.

    ``row`` is duck-typed: any object exposing ``.uptime``, ``.vel_rms_x``, ``.vel_rms_y``,
    ``.vel_rms_z``, ``.accel_rms_x``, ``.accel_rms_y``, ``.accel_rms_z`` by attribute access
    works - a pydantic API request row and a ``DataFrame.itertuples()`` row both qualify.
    Excludes ``timestamp``, since its source field and parsing differ by caller.
    """
    return {
        "uptime": bool(row.uptime),
        "vel_x": float(row.vel_rms_x),
        "vel_y": float(row.vel_rms_y),
        "vel_z": float(row.vel_rms_z),
        "acc_x": float(row.accel_rms_x),
        "acc_y": float(row.accel_rms_y),
        "acc_z": float(row.accel_rms_z),
    }


class TimeSeries(BaseModel):
    """Ordered collection of model-format sensor readings."""

    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints ordered in time.",
    )

    @property
    def length(self) -> int:
        """Number of points in the time series."""
        return len(self.data)

    @property
    def last_timestamp(self) -> datetime:
        """Timestamp of the final point, or ValueError when empty."""
        if not self.data:
            raise ValueError("TimeSeries has no data")
        return self.data[-1].timestamp

    @property
    def first_timestamp(self) -> datetime:
        """Timestamp of the first point, or ValueError when empty."""
        if not self.data:
            raise ValueError("TimeSeries has no data")
        return self.data[0].timestamp


class Weights(BaseModel):
    """Fitted scalar baseline statistics retained for model state."""

    fitted: bool = False
    mean: float = 0.0
    std: float = 1.0


class TrueIncident(BaseModel):
    """Labeled incident window used by evaluation helpers."""

    start: datetime
    end: datetime


class PipelineParams(BaseModel):
    """Shared replay-window parameters for benchmark-style evaluation."""

    model_config = {"protected_namespaces": ()}

    model_window_size_hours: float = 2.0
    window_overlap_hours: float = 1.0


class PredictOutput(BaseModel):
    """Batch-level anomaly features passed from a detector to alerting.

    The baseline detector sets only ``anomaly_status`` and ``timestamp``; the
    grouped-residual detector fills the scoring fields as well.
    """

    anomaly_status: bool
    timestamp: datetime
    occupancy_score: float = 0.0
    alert_score: float = 0.0
    mean_d_score: float = 0.0
    active_channels: list[str] = Field(default_factory=list)
    active_modalities: list[str] = Field(default_factory=list)
    channel_max_residual: dict[str, float] = Field(default_factory=dict)


class AlertDecision(BaseModel):
    """Final alert decision emitted by an alert engine.

    The baseline lock engine sets only ``alert``/``timestamp``/``message``; the
    grouped-residual engine fills the ownership/provenance fields as well.
    """

    alert: bool
    timestamp: datetime
    message: str
    decision_timestamp: datetime | None = None
    anchored_timestamp: datetime | None = None
    owner_level: int = -1
    owner_kind: str = ""
    group_family: str = ""
