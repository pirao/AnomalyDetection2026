"""Notebook-only plotting helpers (EDA and scoring widgets).

Nothing here is imported by production code. Re-exports exist only so
the EDA and model-debugging notebooks can write short imports.
"""

from .eda import create_rms_scenario_widget, create_scenario_inspector
from .scoring import (
    clear_sigmoid_scoring_caches,
    create_api_replay_widget,
    create_group_distribution_widget,
    create_sigmoid_global_residual_widget,
    create_sigmoid_scoring_widget,
    plot_alert_hierarchy_diagram,
    plot_alert_hierarchy_trace_scenario3,
)
from .style import set_plot_style

__all__ = [
    "clear_sigmoid_scoring_caches",
    "create_api_replay_widget",
    "create_group_distribution_widget",
    "set_plot_style",
    "create_rms_scenario_widget",
    "create_scenario_inspector",
    "create_sigmoid_global_residual_widget",
    "create_sigmoid_scoring_widget",
    "plot_alert_hierarchy_diagram",
    "plot_alert_hierarchy_trace_scenario3",
]
