from datetime import datetime
from typing import Literal, Sequence

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    timestamp: datetime = Field(..., description="UTC datetime of the time the data point was collected")
    uptime: bool = Field(..., description="Whether the data point is during uptime")
    vel_x: float = Field(..., description="Vibration velocity component along the X axis")
    vel_y: float = Field(..., description="Vibration velocity component along the Y axis")
    vel_z: float = Field(..., description="Vibration velocity component along the Z axis")
    acc_x: float = Field(..., description="Vibration acceleration component along the X axis")
    acc_y: float = Field(..., description="Vibration acceleration component along the Y axis")
    acc_z: float = Field(..., description="Vibration acceleration component along the Z axis")


class TimeSeries(BaseModel):
    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints ordered in time.",
    )

    @property
    def length(self) -> int:
        return len(self.data)

    @property
    def last_timestamp(self) -> datetime:
        if not self.data:
            raise ValueError("TimeSeries has no data")
        return self.data[-1].timestamp

    @property
    def first_timestamp(self) -> datetime:
        if not self.data:
            raise ValueError("TimeSeries has no data")
        return self.data[0].timestamp


class Weights(BaseModel):
    fitted: bool = False
    mean: float = 0.0
    std: float = 1.0


class ModelParams(BaseModel):
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


class PipelineParams(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_window_size_hours: float = 2.0
    window_overlap_hours: float = 1.0


class PredictOutput(BaseModel):
    anomaly_status: bool
    timestamp: datetime
    occupancy_score: float = 0.0
    alert_score: float = 0.0
    mean_d_score: float = 0.0
    active_channels: list[str] = Field(default_factory=list)
    active_modalities: list[str] = Field(default_factory=list)
    channel_max_residual: dict[str, float] = Field(default_factory=dict)


class AlertParams(BaseModel):
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


class AlertDecision(BaseModel):
    alert: bool
    timestamp: datetime
    message: str
    decision_timestamp: datetime | None = None
    anchored_timestamp: datetime | None = None
    owner_level: int = -1
    owner_kind: str = ""
    group_family: str = ""


class TrueIncident(BaseModel):
    start: datetime
    end: datetime
