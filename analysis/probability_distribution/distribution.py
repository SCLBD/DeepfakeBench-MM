"""Read predictions and render score histograms for the distribution commands."""

from __future__ import annotations

import argparse
import copy
import json
import matplotlib
import numpy as np
import os

from collections.abc import Callable
from pathlib import Path

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def build_parser(description: str, title_suffix: str = 'Predicted Probability Distribution') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.set_defaults(title_suffix=title_suffix)
    parser.add_argument('--log_dir', type=Path, required=True,
                        help='Test log directory containing predictions/*.json.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name, must match predictions/<dataset_name>.json.')
    parser.add_argument('--level', type=str, default='sample', choices=['sample', 'video'],
                        help='Aggregation level. Default: sample.')
    parser.add_argument('--output_dir', type=Path,
                        default=Path(__file__).resolve().parent / 'outputs',
                        help='Output directory. Default: analysis/probability_distribution/outputs.')
    parser.add_argument('--title', type=str, default=None,
                        help=f'Figure title. Default: "<dataset> - {title_suffix}".')
    parser.add_argument('--bins', type=int, default=30, help='Histogram bin count. Default: 30.')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI. Default: 300.')
    return parser


def load_records(log_dir: Path, dataset_name: str) -> list[dict]:
    prediction_path = log_dir / 'predictions' / f'{dataset_name}.json'
    if not prediction_path.exists():
        raise SystemExit(f'Prediction file not found: {prediction_path}')
    with open(prediction_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def aggregate_video_records(records: list[dict]) -> list[dict]:
    """Aggregate segment records to video level, using the max fake probability."""
    groups = {}
    for record in records:
        video_name = os.path.dirname(record.get('path', '')) or record.get('path', '')
        groups.setdefault(video_name, []).append(record)

    aggregated = []
    for video_name, items in groups.items():
        base = copy.deepcopy(items[0])
        base['path'] = video_name
        base['label'] = int(any(int(item['label']) == 1 for item in items))
        if 'video_label' in items[0]:
            base['video_label'] = int(any(int(item.get('video_label', 0)) == 1 for item in items))
        if 'audio_label' in items[0]:
            base['audio_label'] = int(any(int(item.get('audio_label', 0)) == 1 for item in items))
        base['prob'] = float(max(item['prob'] for item in items))
        aggregated.append(base)
    return aggregated


def plot_distribution(
    groups: list[tuple[str, np.ndarray]],
    output_path: Path,
    title: str,
    bins: int,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    for group_name, values in groups:
        if len(values) == 0:
            continue
        _, _, patches = axis.hist(
            values,
            bins=bins,
            alpha=0.5,
            density=True,
            label=f'{group_name} (n={len(values)})',
        )
        color = patches[0].get_facecolor()
        axis.axvline(float(np.mean(values)), color=color, linestyle='--', linewidth=1.2)

    axis.set_xlabel('Predicted fake probability')
    axis.set_ylabel('Density')
    axis.set_xlim(-0.02, 1.02)
    axis.set_title(title)
    axis.legend(loc='best')
    axis.grid(True, linestyle='--', alpha=0.4)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


def run_distribution(args: argparse.Namespace, build_groups: Callable[[list[dict]], list[tuple[str, np.ndarray]]]) -> None:
    """Read the predictions and render the requested groups at the selected level."""
    records = load_records(args.log_dir, args.dataset_name)
    if args.level == 'video':
        records = aggregate_video_records(records)
    groups = build_groups(records)
    title = args.title or f'{args.dataset_name} - {args.title_suffix}'
    output_path = args.output_dir / f'{title}.png'
    plot_distribution(groups, output_path, title, args.bins, args.dpi)

    record_type = 'Videos' if args.level == 'video' else 'Samples'
    print(f'{record_type}: {len(records)}')
    for group_name, values in groups:
        print(f'{group_name}: n={len(values)}, mean={np.mean(values):.6f}, std={np.std(values):.6f}')
    print(f'Figure saved to: {output_path}')
