"""Notebook display helpers for evaluation results.

Imported by notebooks running in IRV_env (Anaconda), which has seaborn installed.
Not safe to import from .venv (seaborn is not a declared dependency there).
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
    """Reds heatmap of a 2×2 confusion DataFrame with integer counts annotated."""
    sns.heatmap(
        df.astype(int),
        annot=True,
        fmt="d",
        cmap="Reds",
        cbar=False,
        linewidths=1,
        linecolor="white",
        square=True,
        annot_kws={"fontsize": 13, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticklabels(["Predicted\nalert", "Predicted\nno-alert"], rotation=0)
    ax.set_yticklabels(["Actual\nincident", "Actual\nhealthy"], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
