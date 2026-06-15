"""Pydantic contracts for the baseline model and alert engine."""

from datetime import datetime
from typing import Sequence

from pydantic import BaseModel, Field

# --- Domain / data transfer types ---


class DataPoint(BaseModel):
    """Single timestamped sensor reading used by the baseline model."""

    timestamp: datetime = Field(
        ..., description="UTC datetime of the time the data point was collected"
    )
    uptime: bool = Field(..., description="Whether the data point is during uptime")
    vel_x: float = Field(
        ..., description="Vibration velocity component along the X axis"
    )
    vel_y: float = Field(
        ..., description="Vibration velocity component along the Y axis"
    )
    vel_z: float = Field(
        ..., description="Vibration velocity component along the Z axis"
    )

    acc_x: float = Field(
        ..., description="Vibration acceleration component along the X axis"
    )
    acc_y: float = Field(
        ..., description="Vibration acceleration component along the Y axis"
    )
    acc_z: float = Field(
        ..., description="Vibration acceleration component along the Z axis"
    )


class TimeSeries(BaseModel):
    """Ordered collection of sensor readings passed to fit and predict."""

    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints, ordered in time, of subsequent measurements of some quantity",
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


# --- Model / pipeline types (shared across modules) ---


class Weights(BaseModel):
    """Fitted scalar baseline statistics for the baseline detector."""

    fitted: bool = False
    mean: float = 0.0
    std: float = 1.0


class ModelParams(BaseModel):
    """Baseline detector thresholds loaded from YAML or defaults."""

    z_threshold: int = 3
    window_anomaly_ratio: float = 0.2


class PipelineParams(BaseModel):
    """Shared replay-window parameters for benchmark-style evaluation."""

    model_config = {"protected_namespaces": ()}

    model_window_size_hours: float = 4.0
    window_overlap_hours: float = 0.0


class PredictOutput(BaseModel):
    """Minimal anomaly output consumed by the baseline alert engine."""

    anomaly_status: bool
    timestamp: datetime


class AlertDecision(BaseModel):
    """Baseline alert decision returned after applying lockout state."""

    alert: bool
    timestamp: datetime
    message: str


class TrueIncident(BaseModel):
    """Labeled incident window used by evaluation helpers."""

    start: datetime
    end: datetime
