"""Reusable t-SNE computation and visualization helpers."""

from .core import compute_tsne
from .plotting import plot_tsne_groups, save_interactive_html, save_static_figure
from .tsne_types import ProjectedGroup, TSNEConfig, TSNEGroup, TSNEResult

__all__ = [
    "ProjectedGroup",
    "TSNEConfig",
    "TSNEGroup",
    "TSNEResult",
    "compute_tsne",
    "plot_tsne_groups",
    "save_interactive_html",
    "save_static_figure",
]
