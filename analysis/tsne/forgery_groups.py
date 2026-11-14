"""Mega-MMDF modality-specific forgery groups and shared CLI implementation."""

from __future__ import annotations

import argparse
import json
import numpy as np
import pickle
import re

from pathlib import Path

from analysis.tsne import TSNEConfig, TSNEGroup, plot_tsne_groups

VISUAL_GROUPS = {
    'Real Video': (0, 3),
    'FS': (1, 4),
    'EFS + FS': (6, 10),
    'FR': (2, 5, 14, 18),
    'FS + FR': (9, 13, 17, 21),
    'EFS + FR': (7, 11, 15, 19),
    'FE + FR': (8, 12, 16, 20),
    'T2AV': (22,),
}
AUDIO_GROUPS = {
    'Real Audio': (0, 1, 2, 6, 7, 8, 9),
    'TTS': (14, 15, 16, 17),
    'VC': (3, 4, 5, 10, 11, 12, 13),
    'PF': (18, 19, 20, 21),
    'T2AV': (22,),
}
COLORS = ('#111111', '#4477AA', '#EE6677', '#228833', '#CCBB44',
          '#66CCEE', '#AA3377', '#BBBBBB')
PIPELINE_RE = re.compile(r'pipeline(\d+)')


def classify_prediction(item: dict, mapping: dict) -> str:
    """Use pipeline IDs; true samples without a pipeline path are pipeline 0."""
    indices = {int(value) for value in PIPELINE_RE.findall(str(item['path']))}
    if not indices and int(item.get('label', -1)) == 0:
        indices = {0}
    if len(indices) != 1:
        raise ValueError(f'Cannot identify a unique pipeline for {item["path"]!r}')
    index = indices.pop()
    for name, members in mapping.items():
        if index in members:
            return name
    raise ValueError(f'Unknown Mega-MMDF pipeline {index}: {item["path"]}')


def build_groups(args, mapping: dict) -> list[TSNEGroup]:
    if args.max_samples_per_group <= 0:
        raise ValueError('max_samples_per_group must be positive.')
    with (args.log_dir / 'predictions' / f'{args.dataset_name}.json').open(encoding='utf-8') as handle:
        predictions = json.load(handle)
    grouped = {name: [] for name in mapping}
    for item in predictions:
        grouped[classify_prediction(item, mapping)].append(item)

    # Sample metadata before constructing feature arrays and hover strings.
    rng = np.random.default_rng(args.sample_seed)
    selected = {}
    for name, items in grouped.items():
        count = min(len(items), args.max_samples_per_group)
        indices = (np.sort(rng.choice(len(items), count, replace=False))
                   if count < len(items) else range(len(items)))
        selected[name] = [items[index] for index in indices]
        print(f'{name}: {len(items)} available -> {count} selected')
    with (args.log_dir / 'features' / args.feature_name / f'{args.dataset_name}.pkl').open('rb') as handle:
        features = pickle.load(handle)

    groups = []
    for color, (name, items) in zip(COLORS, selected.items()):
        if not items:
            print(f'Warning: no {name} samples; skipping group.')
            continue
        vectors = [features[item['path']][item['sample_frame_range']] for item in items]
        groups.append(TSNEGroup(
            name=name,
            features=np.asarray(vectors).reshape(len(items), -1),
            hover_texts=['\n'.join(f'{key}: {value}' for key, value in item.items()) for item in items],
            color=color,
            alpha=0.7,
        ))
    return groups


def main(modality: str) -> None:
    mapping = {'visual': VISUAL_GROUPS, 'audio': AUDIO_GROUPS}[modality]
    parser = argparse.ArgumentParser(description=f'Draw Mega-MMDF t-SNE grouped by {modality} forgery method.')
    parser.add_argument('--log_dir', type=Path, required=True)
    parser.add_argument('--feature_name', required=True)
    parser.add_argument('--dataset_name', choices=['Mega-MMDF'], default='Mega-MMDF')
    parser.add_argument('--title', required=True)
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    parser.add_argument('--max_samples_per_group', type=int, default=500,
                        help='Maximum samples per merged forgery category (default: 500).')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--tsne-seed', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    result = plot_tsne_groups(
        build_groups(args, mapping), args.output_dir / f'{args.title}.png',
        config=TSNEConfig(perplexity=args.perplexity, max_iter=args.max_iter,
                          sample_seed=args.sample_seed, tsne_seed=args.tsne_seed),
        title=args.title, dpi=args.dpi,
    )
    print(f't-SNE saved to: {args.output_dir / args.title} (.png and .html)')
    print(f'Projected {result.total_selected_samples} samples from {len(result.groups)} groups.')
