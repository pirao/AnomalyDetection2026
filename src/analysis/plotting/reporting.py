"""Notebook display helpers for evaluation results.

Lives in ``plotting`` (notebook-only) so it is never pulled into the production
import chain: importing ``analysis.evaluation`` / ``analysis.mlflow`` must stay
``.venv``-safe, and these helpers depend on seaborn + IPython, which are only
installed in the notebook env (the ``notebooks`` extra).
"""

from __future__ import annotations

import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display


def md_table(df: pd.DataFrame) -> None:
    """Render a small DataFrame as a GitHub-flavored markdown table."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    )
    display(Markdown(f"{header}\n{sep}\n{body}"))


def plot_confusion(ax, df: pd.DataFrame, title: str) -> None:
    """Reds heatmap of a 2x2 confusion DataFrame with integer counts annotated."""
    sns.heatmap(
        df.astype(int),
        annot=True,
        fmt="d",
        cmap="Reds",
        cbar=False,
        linewidths=1,
        linecolor="white",
        square=True,
        annot_kws={"fontsize": 16, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xticklabels(["Predicted\nalert", "Predicted\nno-alert"], rotation=0)
    ax.set_yticklabels(["Actual\nincident", "Actual\nhealthy"], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
