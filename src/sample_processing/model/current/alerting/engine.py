from __future__ import annotations

from ..interface import AlertDecision, AlertParams, PredictOutput
from ._utils import (
    check_confirmation,
    confirmation_anchor_entry,
    l2_norm,
    strict_majority_count,
)
from .group_logic import build_group_candidate
from .priority_queue import (
    drop_pending_events_covered_by_group,
    pending_event_resolution_state,
    pending_events_debug,
    queue_pending_event,
)


class AlertEngine:
    """Priority-aware alert engine driven by active causal channels."""

    def __init__(self, params: AlertParams | None = None) -> None:
        if params is None:
            params = AlertParams()
        self.params = params
        self.channel_state: dict[str, dict[str, object]] = {}
        self.group_state: dict[str, object] = {
            "is_open": False,
            "mode_type": "",
            "mode_kind": "",
            "group_family": "",
            "metric_label": "",
            "group_channels": [],
            "active_channels": [],
            "reference_severity": 0.0,
            "current_severity": 0.0,
            "cooldown_windows": self.params.group_min_cooldown_windows,
            "reset_streak": 0,
            "opened_at": None,
            "last_alert_at": None,
            "last_reset_at": None,
        }
        self.generic_open: bool = False
        self.batch_index: int = -1
        self.last_realert_reason: str = ""
        self.last_debug: dict[str, object] = {}
        self.pending_state: dict[str, list[dict[str, object]]] = {}
        self.group3_pending_buffer: list[bool] = []
        self.group6_pending_buffer: list[bool] = []
        self.pending_priority_events: dict[str, dict[str, object]] = {}
        self.global_alert_cooldown: int = 0
        self.last_release_reason: str = ""

    def _ensure_channel_state(self, channel: str) -> dict[str, object]:
        if channel not in self.channel_state:
            self.channel_state[channel] = {
                "is_open": False,
                "reference_max_residual": 0.0,
                "latest_max_residual": 0.0,
                "cooldown_windows": self.params.min_cooldown_windows,
                "reset_streak_below_zero": 0,
                "opened_at": None,
                "last_alert_at": None,
                "last_reset_at": None,
            }
        return self.channel_state[channel]

    def _snapshot_states(self) -> tuple[list[str], dict[str, float], dict[str, float], dict[str, int], dict[str, int]]:
        watched_channels = sorted(
            ch for ch, state in self.channel_state.items() if bool(state["is_open"])
        )
        reference_max_by_channel = {
            ch: float(self.channel_state[ch]["reference_max_residual"])
            for ch in watched_channels
        }
        current_max_by_channel = {
            ch: float(self.channel_state[ch]["latest_max_residual"])
            for ch in watched_channels
        }
        reset_streak_by_channel = {
            ch: int(self.channel_state[ch]["reset_streak_below_zero"])
            for ch in watched_channels
        }
        cooldown_by_channel = {
            ch: int(self.channel_state[ch]["cooldown_windows"])
            for ch in watched_channels
        }
        return (
            watched_channels,
            reference_max_by_channel,
            current_max_by_channel,
            reset_streak_by_channel,
            cooldown_by_channel,
        )

    def _apply_group_candidate(self, candidate: dict[str, object], timestamp) -> None:
        self.group_state["is_open"] = True
        self.group_state["mode_type"] = str(candidate["mode_type"])
        self.group_state["mode_kind"] = str(candidate["mode_kind"])
        self.group_state["group_family"] = str(candidate.get("group_family", ""))
        self.group_state["metric_label"] = str(candidate["metric_label"])
        self.group_state["group_channels"] = list(candidate["group_channels"])
        self.group_state["active_channels"] = list(candidate["active_channels"])
        self.group_state["reference_severity"] = float(candidate["severity"])
        self.group_state["current_severity"] = float(candidate["severity"])
        self.group_state["cooldown_windows"] = 0
        self.group_state["reset_streak"] = 0
        self.group_state["opened_at"] = timestamp
        self.group_state["last_alert_at"] = timestamp

    def _update_pending(self, channel: str, active: bool, timestamp) -> None:
        buf = self.pending_state.setdefault(channel, [])
        buf.append(
            {
                "active": bool(active),
                "timestamp": timestamp,
                "batch_index": int(self.batch_index),
            }
        )
        if len(buf) > self.params.confirmation_window:
            buf[:] = buf[-self.params.confirmation_window:]

    def _update_group_pending(self, active: bool) -> None:
        self.group3_pending_buffer.append(active)
        if len(self.group3_pending_buffer) > self.params.group_confirmation_window:
            self.group3_pending_buffer[:] = self.group3_pending_buffer[-self.params.group_confirmation_window:]

    def _update_group6_pending(self, active: bool) -> None:
        self.group6_pending_buffer.append(active)
        if len(self.group6_pending_buffer) > self.params.group_confirmation_window:
            self.group6_pending_buffer[:] = self.group6_pending_buffer[-self.params.group_confirmation_window:]

    def _group_priority(self) -> int:
        if not bool(self.group_state["is_open"]):
            return -1
        return 2 if str(self.group_state["mode_kind"]) == "group-6" else 1

    def _individual_case_active(self) -> bool:
        return (
            not bool(self.group_state["is_open"])
            and any(bool(state["is_open"]) for state in self.channel_state.values())
        )

    def _exclusive_individual_mode(self) -> bool:
        return str(self.params.individual_alert_mode) == "exclusive"

    def _group_severity(
        self,
        channels: list[str],
        current_max_by_channel: dict[str, float],
    ) -> float:
        return l2_norm([float(current_max_by_channel.get(ch, 0.0)) for ch in channels])

    def _build_debug(
        self,
        *,
        event: str,
        prediction: PredictOutput,
        triggered_channels: list[str],
        opened_channels: list[str],
        realerted_channels: list[str],
        reset_channels: list[str],
        triggered_reasons_by_channel: dict[str, str],
        pending_channels: list[str] | None = None,
        suppressed_by_priority: list[str] | None = None,
        suppression_targets: list[str] | None = None,
        promotion_candidate_kind: str = "",
        promotion_holdback_remaining: int = 0,
        promotion_resolution_state: str = "",
        pending_event_release_reason: str = "",
        suppression_window_expires_at: int = -1,
        emitted_event_scope: str = "",
        decision_timestamp=None,
        alert_timestamp=None,
    ) -> dict[str, object]:
        (
            watched_channels,
            reference_max_by_channel,
            current_max_by_channel,
            reset_streak_by_channel,
            cooldown_by_channel,
        ) = self._snapshot_states()
        return {
            "event": event,
            "batch_index": int(self.batch_index),
            "anomaly_status": bool(prediction.anomaly_status),
            "active_modalities": list(prediction.active_modalities),
            "active_channels": sorted(prediction.active_channels),
            "current_channels": sorted(prediction.channel_max_residual.keys()),
            "watched_channels": watched_channels,
            "triggered_channels": triggered_channels,
            "opened_channels": opened_channels,
            "realerted_channels": realerted_channels,
            "reset_channels": reset_channels,
            "triggered_reasons_by_channel": triggered_reasons_by_channel,
            "reference_max_by_channel": reference_max_by_channel,
            "current_channel_max_residual": {
                ch: float(v) for ch, v in prediction.channel_max_residual.items()
            },
            "current_max_by_channel": current_max_by_channel,
            "reset_streak_by_channel": reset_streak_by_channel,
            "cooldown_by_channel": cooldown_by_channel,
            "group_mode_active": bool(self.group_state["is_open"]),
            "group_mode_type": str(self.group_state["mode_type"]),
            "group_mode_kind": str(self.group_state["mode_kind"]),
            "group_family": str(self.group_state["group_family"]),
            "group_metric_label": str(self.group_state["metric_label"]),
            "group_event": event if event.startswith("group_") else "",
            "group_opened": event == "group_opened",
            "group_realerted": event == "group_realerted",
            "group_reset": event == "group_reset",
            "group_channels": list(self.group_state["group_channels"]),
            "group_active_channels": list(self.group_state["active_channels"]),
            "group_reference_severity": float(self.group_state["reference_severity"]),
            "group_current_severity": float(self.group_state["current_severity"]),
            "group_cooldown_windows": int(self.group_state["cooldown_windows"]),
            "group_reset_streak": int(self.group_state["reset_streak"]),
            "suppressed_channel_alerts": sorted(
                [
                    ch
                    for ch in prediction.active_channels
                    if ch not in triggered_channels
                ]
            ),
            "pending_channels": sorted(pending_channels or []),
            "pending_lower_priority_events": pending_events_debug(self.pending_priority_events),
            "suppressed_by_priority": sorted(set(suppressed_by_priority or [])),
            "suppression_target": sorted(set(suppression_targets or [])),
            "promotion_candidate_kind": promotion_candidate_kind,
            "promotion_holdback_remaining": int(promotion_holdback_remaining),
            "promotion_resolution_state": promotion_resolution_state,
            "pending_event_release_reason": pending_event_release_reason,
            "suppression_window_expires_at": int(suppression_window_expires_at),
            "emitted_event_scope": emitted_event_scope,
            "individual_alert_mode": str(self.params.individual_alert_mode),
            "decision_timestamp": decision_timestamp,
            "alert_timestamp": alert_timestamp,
            "owner_level": int(self._group_priority() if self._group_priority() >= 0 else (0 if watched_channels else -1)),
            "owner_kind": str(self.group_state["mode_kind"]) if bool(self.group_state["is_open"]) else ("channel" if watched_channels else ""),
        }

    def predict(self, prediction: PredictOutput) -> AlertDecision:
        self.batch_index += 1
        if self.global_alert_cooldown > 0:
            self.global_alert_cooldown -= 1
        if not prediction.anomaly_status:
            self.generic_open = False

        current_max_by_channel = {
            ch: float(v) for ch, v in prediction.channel_max_residual.items()
        }
        active_channels = sorted(set(prediction.active_channels))
        active_modalities = set(prediction.active_modalities)
        group_candidate = build_group_candidate(
            active_channels,
            current_max_by_channel,
            enable_group6=bool(self.params.enable_group6_alerts),
        )

        for state in self.channel_state.values():
            if bool(state["is_open"]):
                state["cooldown_windows"] = int(state["cooldown_windows"]) + 1
        if bool(self.group_state["is_open"]):
            self.group_state["cooldown_windows"] = int(self.group_state["cooldown_windows"]) + 1

        reset_channels: list[str] = []
        for channel, state in self.channel_state.items():
            if not bool(state["is_open"]):
                continue
            current_value = float(current_max_by_channel.get(channel, 0.0))
            state["latest_max_residual"] = current_value
            if current_value < 0.0:
                state["reset_streak_below_zero"] = int(state["reset_streak_below_zero"]) + 1
            else:
                state["reset_streak_below_zero"] = 0

            if int(state["reset_streak_below_zero"]) >= self.params.reset_batches_below_zero:
                state["is_open"] = False
                state["last_reset_at"] = prediction.timestamp
                state["cooldown_windows"] = self.params.min_cooldown_windows
                state["reset_streak_below_zero"] = 0
                reset_channels.append(channel)

        group_reset = False
        if bool(self.group_state["is_open"]):
            group_channels = list(self.group_state["group_channels"])
            active_group_channels = sorted(set(ch for ch in active_channels if ch in group_channels))
            self.group_state["active_channels"] = active_group_channels
            self.group_state["current_severity"] = self._group_severity(
                active_group_channels,
                current_max_by_channel,
            )
            majority_needed = strict_majority_count(len(group_channels))
            negative_count = sum(
                1
                for ch in group_channels
                if float(current_max_by_channel.get(ch, 0.0)) < 0.0
            )
            if majority_needed > 0 and negative_count >= majority_needed:
                self.group_state["reset_streak"] = int(self.group_state["reset_streak"]) + 1
            else:
                self.group_state["reset_streak"] = 0
            if int(self.group_state["reset_streak"]) >= self.params.group_reset_batches_below_zero:
                self.group_state["is_open"] = False
                self.group_state["last_reset_at"] = prediction.timestamp
                self.group_state["cooldown_windows"] = self.params.group_min_cooldown_windows
                self.group_state["reset_streak"] = 0
                self.group_state["active_channels"] = []
                self.group_state["current_severity"] = 0.0
                self.group_state["reference_severity"] = 0.0
                self.group_state["mode_type"] = ""
                self.group_state["mode_kind"] = ""
                self.group_state["group_family"] = ""
                self.group_state["metric_label"] = ""
                self.group_state["group_channels"] = []
                group_reset = True

        opened_channels: list[str] = []
        realerted_channels: list[str] = []
        triggered_channels: list[str] = []
        triggered_reasons_by_channel: dict[str, str] = {}
        group_opened = False
        group_realerted = False
        pending_channels: list[str] = []
        suppressed_by_priority: list[str] = []
        suppression_targets: list[str] = []
        chosen_event: dict[str, object] | None = None
        promotion_resolution_state = ""
        suppression_window_expires_at = -1
        emitted_event_scope = ""
        self.last_release_reason = ""
        decision_timestamp = prediction.timestamp
        alert_timestamp = prediction.timestamp

        for channel in active_channels:
            current_value = float(current_max_by_channel.get(channel, 0.0))
            state = self._ensure_channel_state(channel)
            state["latest_max_residual"] = current_value
        group3_now = bool(prediction.anomaly_status) and group_candidate is not None and int(group_candidate["priority"]) >= 1
        group6_enabled = bool(self.params.enable_group6_alerts)
        group6_now = (
            group6_enabled
            and bool(prediction.anomaly_status)
            and group_candidate is not None
            and int(group_candidate["priority"]) == 2
        )
        self._update_group_pending(group3_now)
        self._update_group6_pending(group6_now if group6_enabled else False)
        group3_confirmed = check_confirmation(
            self.group3_pending_buffer,
            self.params.group_confirmation_count,
            self.params.group_confirmation_window,
        )
        group6_confirmed = (
            check_confirmation(
                self.group6_pending_buffer,
                self.params.group_confirmation_count,
                self.params.group_confirmation_window,
            )
            if group6_enabled
            else False
        )
        group6_forming = (
            group6_enabled
            and self.params.suppress_group3_during_group6_candidate
            and group6_now
            and not group6_confirmed
            and self._group_priority() < 2
        )
        group6_precursor = (
            group6_enabled
            and self.params.suppress_group3_during_group6_candidate
            and bool(prediction.anomaly_status)
            and {"vel", "accel"}.issubset(active_modalities)
            and self._group_priority() == 1
        )
        group6_escalation_watch = bool(group6_forming or group6_precursor)
        group3_forming = (
            self.params.suppress_lower_priority_during_group_candidate
            and group3_now
            and not group3_confirmed
            and not group6_escalation_watch
            and self._group_priority() < 1
        )
        promotion_candidate_kind = "group-6" if (group6_now or group6_precursor) else "group-3" if group3_now else ""
        promotion_holdback_remaining = (
            self.params.group3_to_group6_holdback_windows
            if group6_escalation_watch
            else self.params.channel_to_group_holdback_windows if group3_forming else 0
        )

        current_group_priority = self._group_priority()
        candidate_priority = int(group_candidate["priority"]) if group_candidate is not None else -1
        candidate_confirmed = (
            group6_confirmed if candidate_priority == 2 else group3_confirmed if candidate_priority == 1 else False
        )
        if group_candidate is not None and candidate_confirmed and candidate_priority > current_group_priority:
            hold_for_group6 = candidate_priority == 1 and group6_escalation_watch
            event = {
                "key": f"group:{group_candidate['mode_kind']}:open",
                "event_type": "group_open",
                "priority": candidate_priority,
                "channels": list(group_candidate["active_channels"]),
                "candidate": group_candidate,
                "suppressed_by_candidate": "group-6" if hold_for_group6 else "",
                "suppression_target": str(group_candidate["mode_kind"]),
                "hold_until_batch": self.batch_index + (self.params.group3_to_group6_holdback_windows if hold_for_group6 else 0),
                "suppression_window_expires_at": self.batch_index + self.params.group_confirmation_window,
                "event_scope": str(group_candidate["mode_kind"]),
            }
            if hold_for_group6:
                queue_pending_event(
                    self.pending_priority_events,
                    event,
                    pending_channels,
                    suppressed_by_priority,
                    suppression_targets,
                    batch_index=self.batch_index,
                )
            else:
                chosen_event = event
        elif bool(self.group_state["is_open"]):
            active_group_channels = sorted(
                ch for ch in active_channels if ch in self.group_state["group_channels"]
            )
            current_severity = self._group_severity(
                active_group_channels,
                current_max_by_channel,
            )
            self.group_state["current_severity"] = current_severity
            reference_value = float(self.group_state["reference_severity"])
            rel_ok = current_severity >= reference_value * (1 + self.params.group_relative_threshold)
            cool_ok = int(self.group_state["cooldown_windows"]) >= self.params.group_min_cooldown_windows
            if active_group_channels and rel_ok and cool_ok:
                hold_for_group6 = self._group_priority() == 1 and group6_escalation_watch
                event = {
                    "key": f"group:{self.group_state['mode_kind']}:realert",
                    "event_type": "group_realert",
                    "priority": self._group_priority(),
                    "channels": list(active_group_channels),
                    "candidate": {
                        "severity": current_severity,
                        "active_channels": list(active_group_channels),
                    },
                    "suppressed_by_candidate": "group-6" if hold_for_group6 else "",
                    "suppression_target": str(self.group_state["mode_kind"]),
                    "hold_until_batch": self.batch_index + (self.params.group3_to_group6_holdback_windows if hold_for_group6 else 0),
                    "suppression_window_expires_at": self.batch_index + self.params.group_confirmation_window,
                    "event_scope": str(self.group_state["mode_kind"]),
                }
                if hold_for_group6:
                    queue_pending_event(
                        self.pending_priority_events,
                        event,
                        pending_channels,
                        suppressed_by_priority,
                        suppression_targets,
                        batch_index=self.batch_index,
                    )
                else:
                    chosen_event = event

        if not bool(self.group_state["is_open"]):
            self.group_state["active_channels"] = []
            self.group_state["current_severity"] = 0.0

        active_set = set(active_channels)
        for ch in list(self.pending_state.keys()):
            if ch not in active_set:
                self._update_pending(ch, False, prediction.timestamp)
        individual_case_active = self._individual_case_active()

        covered_channels = (
            set(self.group_state["group_channels"]) if bool(self.group_state["is_open"]) else set()
        )
        if chosen_event is None:
            for channel in active_channels:
                if channel in covered_channels:
                    continue
                current_value = float(current_max_by_channel.get(channel, 0.0))
                state = self._ensure_channel_state(channel)
                state["latest_max_residual"] = current_value

                if not bool(state["is_open"]):
                    self._update_pending(channel, True, prediction.timestamp)
                    confirmed = check_confirmation(
                        self.pending_state[channel],
                        self.params.confirmation_count,
                        self.params.confirmation_window,
                    )
                    if not confirmed:
                        if channel not in pending_channels:
                            pending_channels.append(channel)
                        continue
                    anchor_entry = confirmation_anchor_entry(
                        self.pending_state[channel],
                        window=self.params.confirmation_window,
                    )
                    anchor_timestamp = (
                        anchor_entry.get("timestamp", prediction.timestamp)
                        if isinstance(anchor_entry, dict)
                        else prediction.timestamp
                    )
                    anchor_batch_index = (
                        int(anchor_entry.get("batch_index", self.batch_index))
                        if isinstance(anchor_entry, dict)
                        else int(self.batch_index)
                    )
                    suppressor = (
                        "group-6"
                        if group6_escalation_watch
                        else "group-3"
                        if group3_forming or bool(self.group_state["is_open"])
                        else ""
                    )
                    # A single individual case owns the episode until it resets
                    # or grouped ownership takes over.
                    if not suppressor and self._exclusive_individual_mode() and individual_case_active:
                        if channel not in pending_channels:
                            pending_channels.append(channel)
                        continue
                    # New individual opens are muted briefly so grouped ownership
                    # has time to confirm before another channel creates noise.
                    if not suppressor and self.global_alert_cooldown > 0:
                        if channel not in pending_channels:
                            pending_channels.append(channel)
                        continue
                    event = {
                        "key": f"channel:{channel}:open",
                        "event_type": "channel_open",
                        "priority": 0,
                        "channel": channel,
                        "channels": [channel],
                        "current_value": current_value,
                        "suppressed_by_candidate": suppressor,
                        "suppression_target": channel,
                        "hold_until_batch": self.batch_index + (self.params.group3_to_group6_holdback_windows if suppressor == "group-6" else self.params.channel_to_group_holdback_windows if suppressor == "group-3" else 0),
                        "suppression_window_expires_at": self.batch_index + self.params.group_confirmation_window,
                        "event_scope": "channel",
                        "alert_timestamp": anchor_timestamp,
                        "anchor_batch_index": anchor_batch_index,
                    }
                else:
                    state["reset_streak_below_zero"] = 0
                    reference_value = float(state["reference_max_residual"])
                    rel_ok = current_value >= reference_value * (1 + self.params.relative_threshold)
                    abs_ok = (current_value - reference_value) >= self.params.min_channel_delta
                    cool_ok = int(state["cooldown_windows"]) >= self.params.min_cooldown_windows
                    if not (rel_ok and abs_ok and cool_ok):
                        continue
                    suppressor = (
                        "group-6"
                        if group6_escalation_watch
                        else "group-3"
                        if group3_forming or bool(self.group_state["is_open"])
                        else ""
                    )
                    event = {
                        "key": f"channel:{channel}:realert",
                        "event_type": "channel_realert",
                        "priority": 0,
                        "channel": channel,
                        "channels": [channel],
                        "current_value": current_value,
                        "suppressed_by_candidate": suppressor,
                        "suppression_target": channel,
                        "hold_until_batch": self.batch_index + (self.params.group3_to_group6_holdback_windows if suppressor == "group-6" else self.params.channel_to_group_holdback_windows if suppressor == "group-3" else 0),
                        "suppression_window_expires_at": self.batch_index + self.params.group_confirmation_window,
                        "event_scope": "channel",
                    }
                if suppressor:
                    queue_pending_event(
                        self.pending_priority_events,
                        event,
                        pending_channels,
                        suppressed_by_priority,
                        suppression_targets,
                        batch_index=self.batch_index,
                    )
                elif chosen_event is None:
                    chosen_event = event

        if chosen_event is None and self.pending_priority_events:
            current_group_priority = self._group_priority()
            releasable: list[dict[str, object]] = []
            kept: dict[str, dict[str, object]] = {}
            for key, event in self.pending_priority_events.items():
                if (
                    str(event.get("event_type", "")) == "channel_open"
                    and self._exclusive_individual_mode()
                    and individual_case_active
                ):
                    continue
                suppressor = str(event["suppressed_by_candidate"])
                if int(event["hold_until_batch"]) > self.batch_index:
                    kept[key] = event
                    continue
                resolution_state, release_reason = pending_event_resolution_state(
                    suppressor=suppressor,
                    current_group_priority=current_group_priority,
                    group3_forming=group3_forming,
                    group6_forming=group6_escalation_watch,
                    anomaly_status=bool(prediction.anomaly_status),
                )
                if resolution_state == "superseded":
                    continue
                if resolution_state == "blocked":
                    kept[key] = event
                    continue
                event["pending_event_release_reason"] = release_reason
                releasable.append(event)
            self.pending_priority_events = kept
            if releasable:
                releasable.sort(
                    key=lambda event: (-int(event["priority"]), int(event["first_eligible_at"]))
                )
                chosen_event = releasable[0]
                self.last_release_reason = str(
                    chosen_event.get("pending_event_release_reason", "release-window-expired")
                )
                promotion_resolution_state = "released_pending"
                suppression_window_expires_at = int(
                    chosen_event.get(
                        "suppression_window_expires_at",
                        chosen_event.get("hold_until_batch", -1),
                    )
                )

        if chosen_event is None and prediction.anomaly_status and not active_channels and not bool(self.group_state["is_open"]):
            if not self.generic_open:
                self.generic_open = True
                chosen_event = {"event_type": "generic_open"}
                emitted_event_scope = "generic"

        if chosen_event is not None and str(chosen_event["event_type"]) == "group_open":
            candidate = dict(chosen_event["candidate"])
            for channel in candidate["active_channels"]:
                current_value = float(current_max_by_channel.get(channel, 0.0))
                state = self._ensure_channel_state(channel)
                if not bool(state["is_open"]):
                    state["opened_at"] = prediction.timestamp
                state["is_open"] = True
                state["reference_max_residual"] = current_value
                state["latest_max_residual"] = current_value
                state["cooldown_windows"] = 0
                state["reset_streak_below_zero"] = 0
                state["last_alert_at"] = prediction.timestamp
                self.pending_state.pop(channel, None)
            self._apply_group_candidate(candidate, prediction.timestamp)
            triggered_channels = list(candidate["active_channels"])
            triggered_reasons_by_channel = {ch: "group_opened" for ch in triggered_channels}
            group_opened = True
            promotion_resolution_state = f"{candidate['mode_kind']}-confirmed"
            emitted_event_scope = str(chosen_event.get("event_scope", candidate["mode_kind"]))
            self.pending_priority_events = drop_pending_events_covered_by_group(
                self.pending_priority_events,
                current_priority=self._group_priority(),
                group_channels=list(self.group_state["group_channels"]),
            )
        elif chosen_event is not None and str(chosen_event["event_type"]) == "group_realert":
            current_severity = float(chosen_event["candidate"]["severity"])
            active_group_channels = list(chosen_event["candidate"]["active_channels"])
            self.group_state["reference_severity"] = current_severity
            self.group_state["current_severity"] = current_severity
            self.group_state["active_channels"] = active_group_channels
            self.group_state["cooldown_windows"] = 0
            self.group_state["last_alert_at"] = prediction.timestamp
            triggered_channels = list(active_group_channels)
            triggered_reasons_by_channel = {
                ch: "group_relative_worsening" for ch in active_group_channels
            }
            group_realerted = True
            promotion_resolution_state = f"{self.group_state['mode_kind']}-realerted"
            emitted_event_scope = str(chosen_event.get("event_scope", self.group_state["mode_kind"]))
        elif chosen_event is not None and str(chosen_event["event_type"]) == "channel_open":
            channel = str(chosen_event["channel"])
            current_value = float(chosen_event["current_value"])
            state = self._ensure_channel_state(channel)
            state["is_open"] = True
            state["reference_max_residual"] = current_value
            state["latest_max_residual"] = current_value
            state["cooldown_windows"] = 0
            state["reset_streak_below_zero"] = 0
            state["opened_at"] = decision_timestamp
            state["last_alert_at"] = decision_timestamp
            self.pending_state.pop(channel, None)
            opened_channels.append(channel)
            triggered_channels.append(channel)
            triggered_reasons_by_channel[channel] = "opened"
            emitted_event_scope = str(chosen_event.get("event_scope", "channel"))
            alert_timestamp = chosen_event.get("alert_timestamp", prediction.timestamp)
        elif chosen_event is not None and str(chosen_event["event_type"]) == "channel_realert":
            channel = str(chosen_event["channel"])
            current_value = float(chosen_event["current_value"])
            state = self._ensure_channel_state(channel)
            state["reference_max_residual"] = current_value
            state["latest_max_residual"] = current_value
            state["cooldown_windows"] = 0
            state["last_alert_at"] = prediction.timestamp
            realerted_channels.append(channel)
            triggered_channels.append(channel)
            triggered_reasons_by_channel[channel] = "relative_worsening"
            emitted_event_scope = str(chosen_event.get("event_scope", "channel"))

        alert = bool(triggered_channels) or (
            chosen_event is not None and str(chosen_event.get("event_type", "")) == "generic_open"
        )
        if group_opened:
            event = "group_opened"
        elif group_realerted:
            event = "group_realerted"
        elif group_reset:
            event = "group_reset"
        elif opened_channels and realerted_channels:
            event = "opened_and_realerted"
        elif opened_channels:
            event = "opened"
        elif realerted_channels:
            event = "realerted"
        elif reset_channels:
            event = "reset"
        else:
            event = "ongoing" if prediction.anomaly_status else "normal"

        summary_parts: list[str] = []
        if group_opened:
            summary_parts.append(f"group_opened={','.join(active_channels)}")
        if group_realerted:
            summary_parts.append(
                "group_realerted="
                + ",".join(list(self.group_state["active_channels"]))
            )
        if group_reset:
            summary_parts.append("group_reset")
        if opened_channels:
            summary_parts.append(f"opened={','.join(opened_channels)}")
        if realerted_channels:
            summary_parts.append(f"realerted={','.join(realerted_channels)}")
        if reset_channels:
            summary_parts.append(f"reset={','.join(reset_channels)}")
        if chosen_event is not None and str(chosen_event.get("event_type", "")) == "generic_open":
            summary_parts.append("opened=generic")
        self.last_realert_reason = "; ".join(summary_parts) if summary_parts else event
        self.last_debug = self._build_debug(
            event=event,
            prediction=prediction,
            triggered_channels=sorted(triggered_channels),
            opened_channels=sorted(opened_channels),
            realerted_channels=sorted(realerted_channels),
            reset_channels=sorted(reset_channels),
            triggered_reasons_by_channel=triggered_reasons_by_channel,
            pending_channels=sorted(pending_channels),
            suppressed_by_priority=suppressed_by_priority,
            suppression_targets=suppression_targets,
            promotion_candidate_kind=promotion_candidate_kind,
            promotion_holdback_remaining=promotion_holdback_remaining,
            promotion_resolution_state=promotion_resolution_state,
            pending_event_release_reason=self.last_release_reason,
            suppression_window_expires_at=suppression_window_expires_at,
            emitted_event_scope=emitted_event_scope,
            decision_timestamp=decision_timestamp,
            alert_timestamp=alert_timestamp,
        )

        if alert:
            if opened_channels or realerted_channels:
                self.global_alert_cooldown = self.params.inter_alert_cooldown_windows
            return AlertDecision(
                alert=True,
                timestamp=alert_timestamp,
                message=self.last_realert_reason,
                decision_timestamp=decision_timestamp,
                anchored_timestamp=alert_timestamp,
                owner_level=self._group_priority() if self._group_priority() >= 0 else 0,
                owner_kind=str(self.group_state["mode_kind"]) if bool(self.group_state["is_open"]) else "channel",
                group_family=str(self.group_state.get("group_family", "")),
            )
        if reset_channels:
            return AlertDecision(
                alert=False,
                timestamp=prediction.timestamp,
                message=self.last_realert_reason,
                decision_timestamp=decision_timestamp,
                anchored_timestamp=prediction.timestamp,
                owner_level=self._group_priority(),
                owner_kind=str(self.group_state["mode_kind"]) if bool(self.group_state["is_open"]) else "",
                group_family=str(self.group_state.get("group_family", "")),
            )
        return AlertDecision(
            alert=False,
            timestamp=prediction.timestamp,
            message="Normal." if not prediction.anomaly_status else "Abnormal vibration ongoing.",
            decision_timestamp=decision_timestamp,
            anchored_timestamp=prediction.timestamp,
            owner_level=self._group_priority() if self._group_priority() >= 0 else (0 if any(bool(state["is_open"]) for state in self.channel_state.values()) else -1),
            owner_kind=str(self.group_state["mode_kind"]) if bool(self.group_state["is_open"]) else ("channel" if any(bool(state["is_open"]) for state in self.channel_state.values()) else ""),
            group_family=str(self.group_state.get("group_family", "")),
        )

    def reset(self) -> None:
        self.__init__(self.params)
