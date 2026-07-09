"""Notebook display tables built from an inference-test ``report`` dict."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

_NOTEBOOK_TEST_GROUPS = [
    ("test_no_alert_when_no_incident", lambda df: df["n_incidents"] == 0),
    ("test_alert_fires_in_incident_window_single", lambda df: df["n_incidents"] == 1),
    ("test_at_least_one_alert_in_any_incident_window_multi", lambda df: df["n_incidents"] >= 2),
    ("test_every_incident_window_gets_an_alert", lambda df: df["n_incidents"] >= 2),
]


def build_inference_test_metric_cards_df(report: dict[str, Any]) -> pd.DataFrame:
    """Return notebook-friendly aggregate metric cards for test-aligned evaluation."""
    summary = dict(report.get("summary", {}))
    return pd.DataFrame(
        [
            {
                "metric": "precision",
                "value": float(summary.get("precision", 0.0)),
                "formula": "TP / (TP + FP)",
            },
            {
                "metric": "recall",
                "value": float(summary.get("recall", 0.0)),
                "formula": "TP / (TP + FN)",
            },
            {
                "metric": "f1",
                "value": float(summary.get("f1", 0.0)),
                "formula": "2PR / (P + R)",
            },
            {
                "metric": "alert_efficiency",
                "value": float(summary.get("alert_efficiency", 1.0)),
                "formula": "covered / n_alerts",
            },
        ]
    )


def build_inference_test_confusion_matrix_df(
    report: dict[str, Any],
    *,
    normalize: Literal["count", "row"] = "count",
) -> pd.DataFrame:
    """Return the test-aligned scenario confusion matrix for notebook display."""
    summary = dict(report.get("summary", {}))
    matrix = pd.DataFrame(
        [
            [int(summary.get("tp", 0)), int(summary.get("fn", 0))],
            [int(summary.get("fp", 0)), int(summary.get("tn", 0))],
        ],
        index=["actual positive", "actual negative"],
        columns=["predicted positive", "predicted negative"],
    )
    if normalize == "count":
        return matrix
    if normalize == "row":
        row_sums = matrix.sum(axis=1).replace(0, pd.NA)
        return matrix.div(row_sums, axis=0).fillna(0.0)
    raise ValueError("normalize must be 'count' or 'row'")


def build_inference_test_scenario_coverage_df(report: dict[str, Any]) -> pd.DataFrame:
    """Return the notebook-friendly scenario coverage table."""
    scenarios_df = report.get("scenarios_df", pd.DataFrame())
    if not isinstance(scenarios_df, pd.DataFrame) or scenarios_df.empty:
        return pd.DataFrame()
    cols = [
        "scenario_id",
        "scenario_group",
        "scenario_group_label",
        "status",
        "n_incidents",
        "covered_incident_count",
        "missed_incident_count",
        "n_alerts",
        "alert_efficiency",
        "has_alert_in_window",
        "all_incident_windows_hit",
    ]
    available = [col for col in cols if col in scenarios_df.columns]
    return scenarios_df.loc[:, available].copy()


def build_inference_test_blocking_scenarios_df(report: dict[str, Any]) -> pd.DataFrame:
    """Return only FN/PARTIAL scenarios that still need coverage work."""
    coverage_df = build_inference_test_scenario_coverage_df(report)
    if coverage_df.empty or "status" not in coverage_df.columns:
        return pd.DataFrame()
    blocking_df = coverage_df.loc[coverage_df["status"].isin(["FN", "PARTIAL"])].copy()
    if blocking_df.empty:
        return blocking_df
    return blocking_df.sort_values(
        ["status", "missed_incident_count", "scenario_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _split_pass_fail_scenarios_for_test(
    coverage_df: pd.DataFrame,
    *,
    test_name: str,
    scenario_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Return scenario ids that pass or fail a specific test_evaluation.py case."""
    passed: list[int] = []
    failed: list[int] = []
    for sid in scenario_ids:
        row = coverage_df.loc[sid]
        has_alert_in_window = bool(row["has_alert_in_window"])
        all_hit = bool(row["all_incident_windows_hit"])
        n_alerts = int(row["n_alerts"])

        if test_name == "test_no_alert_when_no_incident":
            ok = n_alerts == 0
        elif test_name in {
            "test_alert_fires_in_incident_window_single",
            "test_at_least_one_alert_in_any_incident_window_multi",
        }:
            ok = has_alert_in_window
        elif test_name == "test_every_incident_window_gets_an_alert":
            ok = has_alert_in_window and all_hit
        else:
            ok = False

        (passed if ok else failed).append(int(sid))
    return passed, failed


def build_inference_test_per_test_results_df(report: dict[str, Any]) -> pd.DataFrame:
    """Return notebook-friendly per-test results aligned to test_evaluation.py."""
    coverage_df = report.get("scenario_coverage_df", pd.DataFrame())
    if not isinstance(coverage_df, pd.DataFrame) or coverage_df.empty:
        coverage_df = build_inference_test_scenario_coverage_df(report)
    if not isinstance(coverage_df, pd.DataFrame) or coverage_df.empty:
        return pd.DataFrame(
            columns=["test", "parametrization", "passed", "failed", "failing_scenarios"]
        )

    indexed = coverage_df.set_index("scenario_id", drop=False)
    rows: list[dict[str, Any]] = []
    for test_name, selector in _NOTEBOOK_TEST_GROUPS:
        scenario_ids = sorted(indexed.loc[selector(indexed)].index.tolist())
        passed, failed = _split_pass_fail_scenarios_for_test(
            indexed,
            test_name=test_name,
            scenario_ids=scenario_ids,
        )
        rows.append(
            {
                "test": test_name,
                "parametrization": f"{len(scenario_ids)} scenarios",
                "passed": len(passed),
                "failed": len(failed),
                "failing_scenarios": ", ".join(str(sid) for sid in failed) if failed else "-",
            }
        )
    return pd.DataFrame(rows)


def build_incident_window_metric_cards_df(report: dict[str, Any]) -> pd.DataFrame:
    """Per-event (fault-window-level) precision/recall/F1.

    Uses the same notebook-friendly column shape as
    ``build_inference_test_metric_cards_df``.
    """
    wm = build_incident_window_confusion_matrix_df(report, normalize="count")
    if wm.empty:
        return pd.DataFrame(columns=["metric", "value", "formula"])
    tp = int(wm.loc["actual positive", "predicted positive"])
    fn = int(wm.loc["actual positive", "predicted negative"])
    fp = int(wm.loc["actual negative", "predicted positive"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return pd.DataFrame(
        [
            {"metric": "precision", "value": round(precision, 3), "formula": "TP / (TP + FP)"},
            {"metric": "recall", "value": round(recall, 3), "formula": "TP / (TP + FN)"},
            {"metric": "f1", "value": round(f1, 3), "formula": "2PR / (P + R)"},
        ]
    )


def build_incident_window_confusion_matrix_df(
    report: dict[str, Any],
    *,
    normalize: Literal["count", "row"] = "count",
) -> pd.DataFrame:
    """Window-level confusion matrix in the same format as the scenario-level one.

    Each incident window is one instance (actual positive). Each no-incident
    scenario is one instance (actual negative). A PARTIAL scenario with 2
    incident windows contributes 1 TP + 1 FN instead of a single scenario-level TP.

    TP = incident windows covered by at least one alert.
    FN = incident windows with no alert.
    FP = no-incident scenarios that fired at least one alert.
    TN = no-incident scenarios with no alerts.
    """
    scenarios_df = report.get("scenarios_df", pd.DataFrame())
    if not isinstance(scenarios_df, pd.DataFrame) or scenarios_df.empty:
        return pd.DataFrame()

    tp = int(scenarios_df["covered_incident_count"].sum())
    fn = int(scenarios_df["missed_incident_count"].sum())
    no_incident = scenarios_df.loc[scenarios_df["n_incidents"] == 0]
    fp = int((no_incident["n_alerts"] > 0).sum())
    tn = int((no_incident["n_alerts"] == 0).sum())

    matrix = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["actual positive", "actual negative"],
        columns=["predicted positive", "predicted negative"],
    )
    if normalize == "count":
        return matrix
    if normalize == "row":
        row_sums = matrix.sum(axis=1).replace(0, pd.NA)
        return matrix.div(row_sums, axis=0).fillna(0.0)
    raise ValueError("normalize must be 'count' or 'row'")


def build_inference_test_notebook_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Assemble notebook-oriented summary artifacts from a test-aligned report."""
    metric_cards_df = build_inference_test_metric_cards_df(report)
    confusion_matrix_df = build_inference_test_confusion_matrix_df(report, normalize="count")
    confusion_matrix_row_pct_df = build_inference_test_confusion_matrix_df(report, normalize="row")
    scenario_coverage_df = build_inference_test_scenario_coverage_df(report)
    per_test_df = build_inference_test_per_test_results_df(
        {"scenario_coverage_df": scenario_coverage_df}
    )
    blocking_scenarios_df = build_inference_test_blocking_scenarios_df(report)
    window_confusion_matrix_df = build_incident_window_confusion_matrix_df(report, normalize="count")
    window_confusion_matrix_row_pct_df = build_incident_window_confusion_matrix_df(report, normalize="row")
    window_metric_cards_df = build_incident_window_metric_cards_df(report)
    interpretation_note = (
        "Scenario matrix: each scenario is one instance regardless of how many incident "
        "windows it has - a PARTIAL (1/2 covered) counts the same as a full TP. "
        "Window matrix: each incident window is one instance, so PARTIAL scenarios "
        "contribute both a TP and a FN."
    )
    return {
        "metric_cards_df": metric_cards_df,
        "confusion_matrix_df": confusion_matrix_df,
        "confusion_matrix_row_pct_df": confusion_matrix_row_pct_df,
        "per_test_df": per_test_df,
        "window_confusion_matrix_df": window_confusion_matrix_df,
        "window_confusion_matrix_row_pct_df": window_confusion_matrix_row_pct_df,
        "window_metric_cards_df": window_metric_cards_df,
        "scenario_coverage_df": scenario_coverage_df,
        "blocking_scenarios_df": blocking_scenarios_df,
        "interpretation_note": interpretation_note,
    }
