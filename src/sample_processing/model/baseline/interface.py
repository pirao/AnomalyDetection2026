from datetime import datetime
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field

# --- Domain / data transfer types ---


class DataPoint(BaseModel):
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
    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints, ordered in time, of subsequent measurements of some quantity",
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


# --- Model / pipeline types (shared across modules) ---


class Weights(BaseModel):
    fitted: bool = False
    mean: float = 0.0
    std: float = 1.0


class ModelParams(BaseModel):
    z_threshold: int = 3
    window_anomaly_ratio: float = 0.2


class PipelineParams(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_window_size_hours: float = 4.0
    window_overlap_hours: float = 0.0


class PredictOutput(BaseModel):
    anomaly_status: bool
    timestamp: datetime


class AlertDecision(BaseModel):
    alert: bool
    timestamp: datetime
    message: str


class TrueIncident(BaseModel):
    start: datetime
    end: datetime
