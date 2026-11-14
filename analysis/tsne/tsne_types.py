"""Data structures shared by the t-SNE computation and plotting modules."""

import numpy as np

from dataclasses import dataclass, field
from typing import Sequence, Union


@dataclass
class TSNEGroup:
    """One manually defined group of feature vectors and its visual style.

    ``features`` must use its first dimension as the sample dimension. For
    example, both ``[N, D]`` and ``[N, T, D]`` arrays are accepted. The latter
    is reduced according to :class:`TSNEConfig.feature_reduction`.
    """

    name: str
    features: np.ndarray
    hover_texts: Sequence[str] | None = None
    color: str = "#4C78A8"
    marker: str = "o"
    marker_size: float = 42.0
    alpha: float = 0.82
    edge_color: str = "none"
    edge_width: float = 0.6
    max_samples: int | None = None


@dataclass
class TSNEConfig:
    """Options that affect sampling, feature preparation, and t-SNE itself."""

    perplexity: float = 30.0
    learning_rate: Union[float, str] = "auto"
    max_iter: int = 1000
    early_exaggeration: float = 12.0
    init: str = "pca"
    metric: str = "euclidean"
    tsne_seed: int = 42
    sample_seed: int = 42
    max_samples_per_group: int | None = None
    feature_reduction: str = "flatten"
    l2_normalize: bool = False
    pca_components: int | None = None


@dataclass
class ProjectedGroup:
    """The selected samples of one group after projection to two dimensions."""

    name: str
    coordinates: np.ndarray
    hover_texts: list[str]
    selected_indices: np.ndarray
    color: str
    marker: str
    marker_size: float
    alpha: float
    edge_color: str
    edge_width: float


@dataclass
class TSNEResult:
    """A reusable projection result consumed by static and interactive plots."""

    groups: list[ProjectedGroup]
    config: TSNEConfig
    feature_dimension: int
    total_input_samples: int
    total_selected_samples: int
    coordinates: np.ndarray = field(repr=False)
