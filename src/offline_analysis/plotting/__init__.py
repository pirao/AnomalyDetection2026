"""Notebook-only plotting helpers (EDA and scoring widgets).

Nothing here is imported by production code. Re-exports exist only so
the EDA and model-debugging notebooks can write short imports.
"""

from .eda import create_rms_scenario_widget, create_scenario_inspector
from .reporting import md_table, plot_confusion
from .scoring import (
    clear_sigmoid_scoring_caches,
    create_group_distribution_widget,
    create_offline_replay_widget,
    create_sigmoid_global_residual_widget,
    create_sigmoid_scoring_widget,
)
from .style import set_plot_style

__all__ = [
    "clear_sigmoid_scoring_caches",
    "create_offline_replay_widget",
    "create_group_distribution_widget",
    "set_plot_style",
    "create_rms_scenario_widget",
    "create_scenario_inspector",
    "create_sigmoid_global_residual_widget",
    "create_sigmoid_scoring_widget",
    "md_table",
    "plot_confusion",
]
