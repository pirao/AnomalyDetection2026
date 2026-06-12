"""Offline evaluation — simulate the production batching and scoring lifecycle.

Re-exports the public surface of the submodules below so callers can do
``from analysis.evaluation import ...`` without reaching into them directly.

Module layout
-------------
- ``batching``     — payload conversion + time/row batch iterators
- ``incidents``    — incident span extraction, labels I/O, alert-window matching
- ``simulation``   — single-scenario API replay (the per-batch model loop)
- ``evaluation``   — multi-scenario inference-test orchestration + metric reports
- ``reporting``    — notebook display helpers: ``md_table``, ``plot_confusion``

Consumers: ``plotting.scoring.api_replay_widget``, ``plotting.scoring.widgets``,
``analysis.model_cache``, and notebook 02.

``run_inference_test_evaluation`` is the canonical scenario-level evaluation
entry point. It mirrors ``src/tests/test_evaluation.py``.
"""

from .batching import df_to_timeseries, iter_row_batches, iter_time_batches
from .evaluation import (
    build_incident_window_metric_cards_df,
    build_inference_test_blocking_scenarios_df,
    build_inference_test_confusion_matrix_df,
    build_inference_test_metric_cards_df,
    build_inference_test_notebook_summary,
    build_inference_test_per_test_results_df,
    build_inference_test_scenario_coverage_df,
    diagnose_group_reassignment,
    diagnose_replay_against_incidents,
    run_inference_test_evaluation,
    summarize_inference_test_metrics,
)
from .reporting import md_table, plot_confusion
from .incidents import (
    DEFAULT_LABELS_PATH,
    assign_incident_label,
    get_incident_spans,
    load_incidents_by_scenario,
)
from .simulation import DEFAULT_DATA_DIR, simulate_api_replay_one_scenario

__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_LABELS_PATH",
    "assign_incident_label",
    "build_incident_window_metric_cards_df",
    "build_inference_test_blocking_scenarios_df",
    "build_inference_test_confusion_matrix_df",
    "build_inference_test_metric_cards_df",
    "build_inference_test_notebook_summary",
    "build_inference_test_per_test_results_df",
    "build_inference_test_scenario_coverage_df",
    "df_to_timeseries",
    "diagnose_group_reassignment",
    "diagnose_replay_against_incidents",
    "get_incident_spans",
    "iter_row_batches",
    "iter_time_batches",
    "load_incidents_by_scenario",
    "md_table",
    "plot_confusion",
    "run_inference_test_evaluation",
    "simulate_api_replay_one_scenario",
    "summarize_inference_test_metrics",
]
