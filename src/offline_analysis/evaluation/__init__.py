"""Offline evaluation - simulate the production batching and scoring lifecycle.

Re-exports the public surface of the submodules below so callers can do
``from offline_analysis.evaluation import ...`` without reaching into them directly.

Module layout
-------------
- ``batching``      - payload conversion + time/row batch iterators
- ``incidents``     - incident span extraction, labels I/O, alert-window matching
- ``simulation``    - single-scenario offline replay (the per-batch model loop)
- ``metrics``       - multi-scenario inference-test orchestration + metric computation
- ``report_tables`` - notebook display tables built from a metrics report
- ``diagnostics``   - group-reassignment + per-alert replay explanation

Notebook display helpers (``md_table``, ``plot_confusion``) live in
``offline_analysis.plotting`` - they depend on seaborn/IPython and must NOT be pulled
into this ``.venv``-safe import path.

Consumers: ``plotting.scoring.offline_replay_widget``, ``plotting.scoring.scoring_widget``,
``pipelines.model_cache``, and the model-debugging notebook (1.01).

``run_inference_test_evaluation`` is the canonical scenario-level evaluation
entry point. It mirrors ``src/tests/test_evaluation.py``.
"""

from .batching import df_to_timeseries, iter_row_batches, iter_time_batches
from .diagnostics import (
    diagnose_group_reassignment,
    diagnose_replay_against_incidents,
)
from .incidents import (
    DEFAULT_LABELS_PATH,
    assign_incident_label,
    get_incident_spans,
    load_incidents_by_scenario,
)
from .metrics import (
    prepare_scenario_frames,
    run_inference_test_evaluation,
    scenario_ids_from_data_dir,
    summarize_inference_test_metrics,
)
from .report_tables import (
    build_incident_window_metric_cards_df,
    build_inference_test_blocking_scenarios_df,
    build_inference_test_confusion_matrix_df,
    build_inference_test_metric_cards_df,
    build_inference_test_notebook_summary,
    build_inference_test_per_test_results_df,
    build_inference_test_scenario_coverage_df,
)
from .simulation import DEFAULT_DATA_DIR, simulate_offline_replay_one_scenario

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
    "prepare_scenario_frames",
    "run_inference_test_evaluation",
    "scenario_ids_from_data_dir",
    "simulate_offline_replay_one_scenario",
    "summarize_inference_test_metrics",
]
