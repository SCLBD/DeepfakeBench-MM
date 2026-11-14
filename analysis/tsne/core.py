"""Sampling, feature preparation, and shared t-SNE projection logic."""

from __future__ import annotations

import inspect
import numpy as np

from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Sequence

from .tsne_types import ProjectedGroup, TSNEConfig, TSNEGroup, TSNEResult


@dataclass
class _PreparedGroup:
    source: TSNEGroup
    features: np.ndarray
    hover_texts: list[str]
    selected_indices: np.ndarray


def _validate_group(group: TSNEGroup) -> np.ndarray:
    if not group.name.strip():
        raise ValueError("Each t-SNE group must have a non-empty name.")

    features = np.asarray(group.features)
    if features.ndim < 2:
        raise ValueError(
            f"Group {group.name!r} features must have at least two dimensions "
            f"[samples, features], got shape {features.shape}."
        )
    if features.shape[0] == 0:
        raise ValueError(f"Group {group.name!r} contains no samples.")
    if not np.issubdtype(features.dtype, np.number):
        raise TypeError(f"Group {group.name!r} features must be numeric, got {features.dtype}.")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"Group {group.name!r} features contain NaN or infinity.")

    if group.hover_texts is not None and len(group.hover_texts) != features.shape[0]:
        raise ValueError(
            f"Group {group.name!r} has {features.shape[0]} samples but "
            f"{len(group.hover_texts)} hover strings."
        )
    if group.marker_size <= 0:
        raise ValueError(f"Group {group.name!r} marker_size must be positive.")
    if not 0.0 <= group.alpha <= 1.0:
        raise ValueError(f"Group {group.name!r} alpha must be between 0 and 1.")
    if group.max_samples is not None and group.max_samples <= 0:
        raise ValueError(f"Group {group.name!r} max_samples must be positive or None.")
    return features


def _reduce_features(features: np.ndarray, reduction: str, group_name: str) -> np.ndarray:
    if features.ndim == 2:
        return features.astype(np.float64, copy=False)

    reduction = reduction.lower()
    if reduction == "flatten":
        reduced = features.reshape(features.shape[0], -1)
    elif reduction in {"mean", "max"}:
        # Preserve the final feature axis and reduce all intermediate axes.
        reshaped = features.reshape(features.shape[0], -1, features.shape[-1])
        reduced = reshaped.mean(axis=1) if reduction == "mean" else reshaped.max(axis=1)
    else:
        raise ValueError(
            f"Unknown feature_reduction {reduction!r} while processing group "
            f"{group_name!r}. Choose from: flatten, mean, max."
        )
    return reduced.astype(np.float64, copy=False)


def _prepare_groups(groups: Sequence[TSNEGroup], config: TSNEConfig) -> list[_PreparedGroup]:
    if not groups:
        raise ValueError("At least one t-SNE group is required.")
    if config.max_samples_per_group is not None and config.max_samples_per_group <= 0:
        raise ValueError("max_samples_per_group must be positive or None.")

    rng = np.random.default_rng(config.sample_seed)
    prepared: list[_PreparedGroup] = []
    names: set[str] = set()

    for group in groups:
        if group.name in names:
            raise ValueError(f"Duplicate t-SNE group name: {group.name!r}.")
        names.add(group.name)
        features = _validate_group(group)

        sample_limit = group.max_samples
        if sample_limit is None:
            sample_limit = config.max_samples_per_group
        sample_count = features.shape[0]
        if sample_limit is not None and sample_count > sample_limit:
            selected = np.sort(rng.choice(sample_count, size=sample_limit, replace=False))
        else:
            selected = np.arange(sample_count, dtype=np.int64)

        selected_features = _reduce_features(features[selected], config.feature_reduction, group.name)
        all_hover_texts = (
            [str(text) for text in group.hover_texts]
            if group.hover_texts is not None
            else [f"{group.name}[{index}]" for index in range(sample_count)]
        )
        selected_hover_texts = [all_hover_texts[index] for index in selected]
        prepared.append(
            _PreparedGroup(
                source=group,
                features=selected_features,
                hover_texts=selected_hover_texts,
                selected_indices=selected,
            )
        )

    dimensions = {item.features.shape[1] for item in prepared}
    if len(dimensions) != 1:
        details = ", ".join(
            f"{item.source.name}={item.features.shape[1]}" for item in prepared
        )
        raise ValueError(f"All groups must have the same feature dimension after reduction: {details}.")
    return prepared


def _build_tsne(config: TSNEConfig) -> TSNE:
    kwargs = {
        "n_components": 2,
        "perplexity": config.perplexity,
        "learning_rate": config.learning_rate,
        "early_exaggeration": config.early_exaggeration,
        "init": config.init,
        "metric": config.metric,
        "random_state": config.tsne_seed,
    }
    # scikit-learn renamed n_iter to max_iter in 1.5. Supporting both keeps
    # this utility compatible with the repository-pinned and newer versions.
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = config.max_iter
    else:
        kwargs["n_iter"] = config.max_iter
    return TSNE(**kwargs)


def compute_tsne(
    groups: Sequence[TSNEGroup],
    config: TSNEConfig | None = None,
) -> TSNEResult:
    """Project all manually defined groups together in one shared t-SNE space."""

    config = config or TSNEConfig()
    prepared = _prepare_groups(groups, config)
    all_features = np.concatenate([item.features for item in prepared], axis=0)
    total_samples = all_features.shape[0]

    if total_samples < 2:
        raise ValueError("t-SNE requires at least two selected samples.")
    if not 0 < config.perplexity < total_samples:
        raise ValueError(
            f"perplexity must be positive and smaller than the selected sample count "
            f"({total_samples}), got {config.perplexity}."
        )
    if config.max_iter < 250:
        raise ValueError("max_iter must be at least 250 for scikit-learn t-SNE.")

    if np.all(np.std(all_features, axis=0) == 0):
        raise ValueError(
            "All selected samples share identical feature vectors (zero variance in every column); "
            "t-SNE cannot produce a meaningful projection. Check the feature extraction for these samples."
        )

    if config.l2_normalize:
        norms = np.linalg.norm(all_features, axis=1, keepdims=True)
        all_features = all_features / np.maximum(norms, np.finfo(np.float64).eps)

    if config.pca_components is not None:
        maximum = min(all_features.shape)
        if not 1 <= config.pca_components <= maximum:
            raise ValueError(
                f"pca_components must be between 1 and {maximum}, "
                f"got {config.pca_components}."
            )
        all_features = PCA(
            n_components=config.pca_components,
            random_state=config.tsne_seed,
        ).fit_transform(all_features)

    coordinates = _build_tsne(config).fit_transform(all_features)

    projected_groups: list[ProjectedGroup] = []
    start = 0
    for item in prepared:
        end = start + item.features.shape[0]
        source = item.source
        projected_groups.append(
            ProjectedGroup(
                name=source.name,
                coordinates=coordinates[start:end],
                hover_texts=item.hover_texts,
                selected_indices=item.selected_indices,
                color=source.color,
                marker=source.marker,
                marker_size=source.marker_size,
                alpha=source.alpha,
                edge_color=source.edge_color,
                edge_width=source.edge_width,
            )
        )
        start = end

    return TSNEResult(
        groups=projected_groups,
        config=config,
        feature_dimension=prepared[0].features.shape[1],
        total_input_samples=sum(np.asarray(group.features).shape[0] for group in groups),
        total_selected_samples=total_samples,
        coordinates=coordinates,
    )
