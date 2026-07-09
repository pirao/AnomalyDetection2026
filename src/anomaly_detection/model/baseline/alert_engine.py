"""Locking baseline alert engine used by comparison experiments."""

from ..shared.interface import AlertDecision, PredictOutput


class AlertEngine:
    """Emit one alert for the first anomalous baseline prediction."""

    def __init__(
        self,
    ):
        self.locked = False

    def _has_alert(self, prediction: PredictOutput) -> bool:
        return prediction.anomaly_status

    def predict(self, prediction: PredictOutput) -> AlertDecision:
        """Apply one-shot lockout to a baseline anomaly prediction."""
        if self.locked:
            return AlertDecision(
                alert=False,
                timestamp=prediction.timestamp,
                message="System already entered abnormal state earlier.",
            )
        if self._has_alert(prediction):
            self.locked = True
            return AlertDecision(
                alert=True,
                timestamp=prediction.timestamp,
                message="Abnormal vibration detected.",
            )

        return AlertDecision(
            alert=False,
            timestamp=prediction.timestamp,
            message="No persistent abnormal vibration.",
        )
