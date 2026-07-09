"""Shared matplotlib rcParams used across EDA and scoring figures in both notebooks."""

import matplotlib.pyplot as plt


def set_plot_style():
    """Set a consistent plotting style for matplotlib figures."""
    
    plt.style.use('default')

    plt.rcParams.update({
        'font.size': 16,
        'axes.linewidth': 2,
        'axes.titlesize': 20,
        'axes.edgecolor': 'black',
        'axes.labelsize': 20,
        'axes.grid': True,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'figure.figsize': (15, 6),
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'font.family': 'sans-serif',
        # Arial on hosts that have it (e.g. Windows); Liberation Sans is a
        # metric-identical free substitute installed in the Docker image; DejaVu
        # Sans is matplotlib's always-present final fallback. At least one always
        # resolves, so the "Generic family not found ... Arial" warning never fires.
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'legend.fontsize': 16,
        'legend.framealpha': 1,
        'legend.edgecolor': 'black',
        'legend.shadow': False,
        'legend.fancybox': True,
        'legend.frameon': True,
    })
    
