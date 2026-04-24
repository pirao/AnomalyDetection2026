"""
Model unit tests — shared DS / MLE evaluation track.

Tests the core model components directly (bypassing the HTTP layer):
- AnomalyModel: feature extraction, fitting, prediction
- AlertEngine: alert suppression and re-arming logic

Candidates must not change the public interface (method signatures, return types)
but are expected to improve internal logic.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sample_processing.model.alert_engine import AlertEngine
from sample_processing.model.anomaly_model import AnomalyModel
from sample_processing.model.interface import (
    AlertDecision,
    DataPoint,
    PredictOutput,
    TimeSeries,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_UTC = timezone.utc
_BASE = datetime(2024, 1, 1, tzinfo=_UTC)


def _make_ts(
    n: int, vel_base: float = 1.0, vel_std: float = 0.1, seed: int = 42
) -> TimeSeries:
    """Create a synthetic TimeSeries of *n* points centred around *vel_base*."""
    rng = np.random.default_rng(seed)
    points = []
    for i in range(n):
        v = rng.normal(vel_base, vel_std, 3)
        a = rng.normal(0.5, 0.05, 3)
        points.append(
            DataPoint(
                timestamp=_BASE + timedelta(minutes=10 * i),
                uptime=True,
                vel_x=float(v[0]),
                vel_y=float(v[1]),
                vel_z=float(v[2]),
                acc_x=float(a[0]),
                acc_y=float(a[1]),
                acc_z=float(a[2]),
            )
        )
    return TimeSeries(data=points)


def _make_pred(*, anomalous: bool, offset: int = 0) -> PredictOutput:
    return PredictOutput(
        anomaly_status=anomalous,
        timestamp=_BASE + timedelta(hours=offset),
    )


# ── AnomalyModel ──────────────────────────────────────────────────────────────


class TestAnomalyModelFit:
    def test_fit_marks_model_as_fitted(self):
        model = AnomalyModel()
        assert model.weights.fitted is False
        model.fit(_make_ts(200))
        assert model.weights.fitted is True

    def test_fit_stores_mean_and_std(self):
        model = AnomalyModel()
        model.fit(_make_ts(200))
        assert model.weights.mean > 0
        assert model.weights.std > 0

    def test_fit_accepts_large_timeseries(self):
        model = AnomalyModel()
        model.fit(_make_ts(5000))
        assert model.weights.fitted is True

    def test_fit_can_be_called_multiple_times(self):
        """Retraining should update weights without error."""
        model = AnomalyModel()
        model.fit(_make_ts(100))
        mean_first = model.weights.mean
        model.fit(_make_ts(100, vel_base=5.0))
        assert model.weights.mean != mean_first


class TestAnomalyModelPredict:
    def test_predict_raises_if_not_fitted(self):
        model = AnomalyModel()
        with pytest.raises(RuntimeError):
            model.predict(_make_ts(10))

    def test_predict_raises_on_empty_timeseries(self):
        model = AnomalyModel()
        model.fit(_make_ts(200))
        with pytest.raises((ValueError, Exception)):
            model.predict(TimeSeries(data=[]))

    def test_predict_returns_predict_output(self):
        model = AnomalyModel()
        model.fit(_make_ts(200))
        result = model.predict(_make_ts(50))
        assert isinstance(result, PredictOutput)

    def test_predict_output_has_anomaly_status(self):
        model = AnomalyModel()
        model.fit(_make_ts(200))
        result = model.predict(_make_ts(50))
        assert isinstance(result.anomaly_status, bool)

    def test_predict_output_has_timestamp(self):
        model = AnomalyModel()
        model.fit(_make_ts(200))
        ts = _make_ts(50)
        result = model.predict(ts)
        assert isinstance(result.timestamp, datetime)

    def test_normal_data_is_not_anomalous(self):
        """Data from the same distribution as training should not be anomalous."""
        model = AnomalyModel()
        model.fit(_make_ts(500, vel_base=1.0, vel_std=0.1, seed=0))
        result = model.predict(_make_ts(100, vel_base=1.0, vel_std=0.1, seed=1))
        assert result.anomaly_status is False

    def test_clearly_anomalous_data_is_flagged(self):
        """Data far from the training distribution should be flagged anomalous."""
        model = AnomalyModel()
        model.fit(_make_ts(500, vel_base=1.0, vel_std=0.1, seed=0))
        # 50× the training mean → well beyond any reasonable threshold
        result = model.predict(_make_ts(100, vel_base=50.0, vel_std=0.5, seed=99))
        assert result.anomaly_status is True

    def test_predict_output_timestamp_is_last_point(self):
        """PredictOutput.timestamp must equal the last DataPoint's timestamp."""
        model = AnomalyModel()
        ts = _make_ts(200)
        model.fit(ts)
        batch = _make_ts(30)
        result = model.predict(batch)
        assert result.timestamp == batch.data[-1].timestamp


# ── AlertEngine ───────────────────────────────────────────────────────────────


class TestAlertEngineBasicBehavior:
    def test_normal_prediction_does_not_alert(self):
        engine = AlertEngine()
        decision = engine.predict(_make_pred(anomalous=False))
        assert decision.alert is False

    def test_first_anomaly_fires_alert(self):
        engine = AlertEngine()
        decision = engine.predict(_make_pred(anomalous=True))
        assert decision.alert is True

    def test_second_anomaly_does_not_re_alert(self):
        """After the first alert the engine must suppress duplicate alerts."""
        engine = AlertEngine()
        engine.predict(_make_pred(anomalous=True, offset=0))
        decision = engine.predict(_make_pred(anomalous=True, offset=1))
        assert decision.alert is False

    def test_alert_decision_is_alert_decision_type(self):
        engine = AlertEngine()
        result = engine.predict(_make_pred(anomalous=False))
        assert isinstance(result, AlertDecision)

    def test_alert_decision_has_timestamp(self):
        engine = AlertEngine()
        pred = _make_pred(anomalous=False, offset=5)
        result = engine.predict(pred)
        assert result.timestamp == pred.timestamp

    def test_alert_decision_has_message(self):
        engine = AlertEngine()
        result = engine.predict(_make_pred(anomalous=True))
        assert isinstance(result.message, str)
        assert len(result.message) > 0

    def test_alert_bool_field_is_bool(self):
        engine = AlertEngine()
        result = engine.predict(_make_pred(anomalous=False))
        assert isinstance(result.alert, bool)


class TestAlertEngineReArming:
    def test_alert_re_arms_after_normal_observation(self):
        """
        After a normal batch clears the alert state, the engine should be
        able to fire again on a subsequent anomalous batch.

        The baseline implementation FAILS this test — it locks permanently.
        Candidates are expected to fix this behaviour.
        """
        engine = AlertEngine()

        # First anomaly → alert fires
        d1 = engine.predict(_make_pred(anomalous=True, offset=0))
        assert d1.alert is True

        # Normal observation → clears the alert state
        engine.predict(_make_pred(anomalous=False, offset=1))

        # Second anomaly → must alert again
        d3 = engine.predict(_make_pred(anomalous=True, offset=2))
        assert d3.alert is True, (
            "AlertEngine must re-arm after a normal observation so it can "
            "detect a second incident."
        )

    def test_multiple_normal_periods_re_arm_engine(self):
        """Engine should re-arm consistently across multiple normal periods."""
        engine = AlertEngine()

        for cycle in range(3):
            d_anom = engine.predict(_make_pred(anomalous=True, offset=cycle * 3))
            assert d_anom.alert is True, f"Should alert on anomaly in cycle {cycle}"
            engine.predict(_make_pred(anomalous=False, offset=cycle * 3 + 1))

    def test_consecutive_anomalies_only_fire_one_alert(self):
        """
        During a single anomaly episode (no normal period between them),
        only the first batch should produce alert=True.
        """
        engine = AlertEngine()
        results = [
            engine.predict(_make_pred(anomalous=True, offset=i)) for i in range(5)
        ]
        assert results[0].alert is True
        for r in results[1:]:
            assert r.alert is False, "Duplicate alerts within the same episode"

    def test_alert_not_active_after_normal_period(self):
        """Observing a normal batch after an alert must clear the locked state."""
        engine = AlertEngine()
        engine.predict(_make_pred(anomalous=True, offset=0))  # fire

        d_normal = engine.predict(_make_pred(anomalous=False, offset=1))
        assert d_normal.alert is False  # no alert on normal batch

        # State should now be clear — next anomaly fires
        d_re = engine.predict(_make_pred(anomalous=True, offset=2))
        assert d_re.alert is True


# ── AnomalyModel + AlertEngine integration ───────────────────────────────────


class TestModelEngineIntegration:
    def test_normal_then_anomalous_batch(self):
        model = AnomalyModel()
        engine = AlertEngine()
        model.fit(_make_ts(500, vel_base=1.0, vel_std=0.1, seed=0))

        pred_normal = model.predict(_make_ts(50, vel_base=1.0, vel_std=0.1, seed=1))
        d_normal = engine.predict(pred_normal)
        assert d_normal.alert is False

        pred_anomaly = model.predict(_make_ts(50, vel_base=50.0, vel_std=0.5, seed=99))
        d_anomaly = engine.predict(pred_anomaly)
        assert d_anomaly.alert is True

    def test_multiple_sensors_have_independent_engines(self):
        models = {sid: AnomalyModel() for sid in ("a", "b")}
        engines = {sid: AlertEngine() for sid in ("a", "b")}
        fit_ts = _make_ts(200)

        for m in models.values():
            m.fit(fit_ts)

        # Fire alert on sensor "a"
        engines["a"].predict(models["a"].predict(_make_ts(50, vel_base=50.0, seed=99)))

        # Sensor "b" must still fire on its own anomaly
        pred_b = models["b"].predict(_make_ts(50, vel_base=50.0, seed=99))
        d_b = engines["b"].predict(pred_b)
        assert d_b.alert is True
