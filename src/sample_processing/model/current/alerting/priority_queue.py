"""Pending-priority-events queue: helpers for deferred alerts held back by higher-priority owners."""

from __future__ import annotations


def pending_events_debug(
    pending_priority_events: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(pending_priority_events):
        event = pending_priority_events[key]
        rows.append(
            {
                "key": key,
                "event_type": str(event["event_type"]),
                "priority": int(event["priority"]),
                "channels": list(event["channels"]),
                "first_eligible_at": int(event["first_eligible_at"]),
                "hold_until_batch": int(event["hold_until_batch"]),
                "suppressed_by_candidate": str(event["suppressed_by_candidate"]),
                "suppression_target": str(event["suppression_target"]),
                "suppression_window_expires_at": int(
                    event.get("suppression_window_expires_at", event["hold_until_batch"])
                ),
            }
        )
    return rows


def queue_pending_event(
    pending_priority_events: dict[str, dict[str, object]],
    event: dict[str, object],
    pending_channels: list[str],
    suppressed_by_priority: list[str],
    suppression_targets: list[str],
    *,
    batch_index: int,
) -> None:
    key = str(event["key"])
    if key not in pending_priority_events:
        event["first_eligible_at"] = batch_index
        pending_priority_events[key] = event
    else:
        pending_priority_events[key].update(event)
    for channel in event["channels"]:
        if channel not in pending_channels:
            pending_channels.append(channel)
    if str(event["suppressed_by_candidate"]):
        suppressed_by_priority.append(str(event["suppressed_by_candidate"]))
    if str(event["suppression_target"]):
        suppression_targets.append(str(event["suppression_target"]))


def pending_event_resolution_state(
    *,
    suppressor: str,
    current_group_priority: int,
    group3_forming: bool,
    group6_forming: bool,
    anomaly_status: bool,
) -> tuple[str, str]:
    if suppressor == "group-6":
        if current_group_priority >= 2:
            return "superseded", "group-6-confirmed"
        if group6_forming:
            return "blocked", "group-6-still-forming"
        return "releasable", "de-escalated" if not anomaly_status else "group-6-window-expired"
    if suppressor == "group-3":
        if current_group_priority >= 1:
            return "superseded", "group-3-confirmed"
        if group3_forming:
            return "blocked", "group-3-still-forming"
        return "releasable", "de-escalated" if not anomaly_status else "group-3-window-expired"
    return "releasable", "not-suppressed"


def drop_pending_events_covered_by_group(
    pending_priority_events: dict[str, dict[str, object]],
    *,
    current_priority: int,
    group_channels: list[str],
) -> dict[str, dict[str, object]]:
    if current_priority < 0:
        return pending_priority_events
    kept: dict[str, dict[str, object]] = {}
    for key, event in pending_priority_events.items():
        event_priority = int(event["priority"])
        if current_priority == 2 and event_priority <= 1:
            continue
        if current_priority == 1 and event_priority == 0:
            continue
        kept[key] = event
    return kept
