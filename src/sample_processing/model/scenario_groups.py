"""Scenario-to-group mapping shared by runtime and analysis code."""

from __future__ import annotations

from typing import Any

import pandas as pd

GROUP_DEFINITIONS: dict[str, dict[str, object]] = {
    "group_1": {
        "label": "Group 1 - large single spike",
        "description": "Large single spike",
        "scenario_ids": (1, 2, 4, 7, 9, 11, 17),
    },
    "group_2": {
        "label": "Group 2 - sudden trend increase",
        "description": "Sudden trend increase",
        "scenario_ids": (3, 6, 10, 15, 16, 18, 20, 21, 22, 24, 26, 27),
    },
    "group_3": {
        "label": "Group 3 - segments with sudden increases and back to normal",
        "description": "Segments with sudden increases and back to normal",
        "scenario_ids": (5, 8, 19, 25, 28, 29),
    },
    "group_4": {
        "label": "Group 4 - many ups and downs with localized degradation",
        "description": "Machines with many ups and downs with a specific degradation location",
        "scenario_ids": (12, 13, 14, 23),
    },
}

SCENARIO_TO_GROUP: dict[int, str] = {
    int(scenario_id): group_key
    for group_key, group_info in GROUP_DEFINITIONS.items()
    for scenario_id in group_info["scenario_ids"]
}

DEFAULT_GROUP_KEY = "default"
DEFAULT_GROUP_LABEL = "Unassigned group"


def normalize_scenario_id(value: Any) -> int | None:
    """Convert any scenario ID representation to int, or None if not convertible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_scenario_group_key(
    scenario_id: Any,
    *,
    fallback: str = DEFAULT_GROUP_KEY,
) -> str:
    """Return the group key for a scenario ID."""
    normalized = normalize_scenario_id(scenario_id)
    if normalized is None:
        return fallback
    return SCENARIO_TO_GROUP.get(normalized, fallback)


def get_scenario_group_label(
    scenario_id: Any,
    *,
    fallback: str = DEFAULT_GROUP_LABEL,
) -> str:
    """Return the human-readable group label for a scenario ID."""
    group_key = get_scenario_group_key(scenario_id)
    if group_key == DEFAULT_GROUP_KEY:
        return fallback
    return str(GROUP_DEFINITIONS[group_key]["label"])


def add_scenario_group_labels(
    df: pd.DataFrame,
    *,
    scenario_col: str = "scenario_id",
    group_col: str = "scenario_group",
    group_label_col: str = "scenario_group_label",
) -> pd.DataFrame:
    """Append scenario group key and label columns to a DataFrame."""
    out = df.copy()
    scenario_ids = out[scenario_col]
    out[group_col] = scenario_ids.map(get_scenario_group_key)
    out[group_label_col] = scenario_ids.map(get_scenario_group_label)
    return out
