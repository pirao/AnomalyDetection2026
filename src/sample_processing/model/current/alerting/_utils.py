"""Pure numeric + buffer helpers used by the alert engine."""

from __future__ import annotations


def strict_majority_count(n_channels: int) -> int:
    """Return the smallest count that forms a strict majority."""
    return (int(n_channels) // 2) + 1 if n_channels > 0 else 0


def mean(values: list[float]) -> float:
    """Return the arithmetic mean, using 0.0 for an empty list."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def l2_norm(values: list[float]) -> float:
    """Return the Euclidean norm, using 0.0 for an empty list."""
    if not values:
        return 0.0
    return float(sum(v * v for v in values) ** 0.5)


def buffer_active(entry: object) -> bool:
    """Read an active flag from a buffer entry or coerce the value to bool."""
    if isinstance(entry, dict):
        return bool(entry.get("active", False))
    return bool(entry)


def center_anchor_entry(buffer: list[dict[str, object]]) -> dict[str, object] | None:
    """Return the center entry used as the timestamp anchor for confirmation."""
    if not buffer:
        return None
    anchor_idx = max((len(buffer) - 1) // 2, 0)
    return buffer[anchor_idx]


def confirmation_anchor_entry(
    buffer: list[dict[str, object]],
    *,
    window: int,
) -> dict[str, object] | None:
    """Return the center anchor from the active confirmation window."""
    if not buffer:
        return None
    trimmed = buffer[-window:] if len(buffer) > window else buffer
    return center_anchor_entry(trimmed)


def check_confirmation(buffer: list[object], count: int, window: int) -> bool:
    """Return whether the recent window has enough active entries."""
    trimmed = buffer[-window:] if len(buffer) > window else buffer
    if len(trimmed) < window:
        return False
    return sum(1 for item in trimmed if buffer_active(item)) >= count
