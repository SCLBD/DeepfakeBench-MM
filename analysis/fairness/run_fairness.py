"""Single-detector fairness analysis from one prediction JSON."""

from __future__ import annotations

import argparse
import copy
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os

from pathlib import Path
from sklearn import metrics as skmetrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Show per-demographic-subgroup performance and delta from the overall metric.'
    )
    parser.add_argument('--log_dir', type=Path, required=True,
                        help='Test log directory containing predictions/*.json.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name, must match predictions/<dataset_name>.json.')
    parser.add_argument('--level', type=str, default='sample', choices=['sample', 'video'],
                        help='Aggregation level. Default: sample.')
    parser.add_argument('--metric', type=str, default='ACC', choices=['ACC', 'AUC', 'AP', 'EER'],
                        help='Metric used for each subgroup. Default: ACC.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for ACC. Default: 0.5.')
    parser.add_argument('--output_dir', type=Path,
                        default=PROJECT_ROOT / 'analysis' / 'fairness' / 'outputs',
                        help='Output directory. Default: analysis/fairness/outputs.')
    parser.add_argument('--title', type=str, default=None,
                        help='Figure title. Default: "<dataset> - Fairness".')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI. Default: 300.')
    return parser.parse_args()


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
        demographic_groups = {
            item.get('demographic_group') for item in items
            if 'demographic_group' in item
        }
        if len(demographic_groups) > 1:
            raise ValueError(
                f'Video {video_name} has inconsistent demographic groups: {demographic_groups}'
            )
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


def compute_metric(records: list[dict], metric: str, threshold: float) -> float:
    labels = np.asarray([record['label'] for record in records], dtype=np.int64)
    probabilities = np.asarray([record['prob'] for record in records], dtype=np.float64)
    if metric == 'ACC':
        predictions = (probabilities > threshold).astype(np.int64)
        return float(skmetrics.accuracy_score(labels, predictions))
    if len(np.unique(labels)) < 2:
        return float('nan')
    if metric == 'AUC':
        return float(skmetrics.roc_auc_score(labels, probabilities))
    if metric == 'AP':
        return float(skmetrics.average_precision_score(labels, probabilities))
    if metric == 'EER':
        fpr, tpr, _ = skmetrics.roc_curve(labels, probabilities)
        fnr = 1.0 - tpr
        index = int(np.nanargmin(np.abs(fnr - fpr)))
        return float(fpr[index])
    raise ValueError(f'Unsupported metric: {metric}')


def main() -> None:
    args = parse_args()
    records = load_records(args.log_dir, args.dataset_name)
    if args.level == 'video':
        records = aggregate_video_records(records)
    if not all('demographic_group' in record for record in records):
        raise SystemExit('demographic_group is missing from prediction records.')

    groups = {}
    for record in records:
        group = record['demographic_group']
        groups.setdefault(group, []).append(record)

    overall = compute_metric(records, args.metric, args.threshold)
    rows = []
    for group_name, group_records in groups.items():
        subgroup_metric = compute_metric(group_records, args.metric, args.threshold)
        rows.append({
            'subgroup': group_name,
            'n': len(group_records),
            'metric': subgroup_metric,
            'delta': subgroup_metric - overall if np.isfinite(subgroup_metric) else float('nan'),
        })

    title = args.title or f'{args.dataset_name} - Fairness ({args.metric})'
    output_path = args.output_dir / f'{title}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subgroup_names = [row['subgroup'] for row in rows]
    subgroup_values = np.asarray([row['metric'] for row in rows], dtype=np.float64)
    subgroup_deltas = np.asarray([row['delta'] for row in rows], dtype=np.float64)
    subgroup_counts = np.asarray([row['n'] for row in rows], dtype=np.int64)

    figure, (axis_top, axis_bottom) = plt.subplots(
        2, 1, figsize=(max(6.5, 0.8 * len(subgroup_names)), 8.0), sharex=True
    )
    x_positions = np.arange(len(subgroup_names))
    axis_top.axhline(overall, color='gray', linestyle='--', linewidth=1.2,
                     label=f'Overall {args.metric} = {overall:.4f}')
    axis_top.scatter(x_positions, subgroup_values, s=80, color='#4c78a8', zorder=3)
    for x_position, value, count in zip(x_positions, subgroup_values, subgroup_counts):
        if np.isfinite(value):
            axis_top.annotate(f'n={count}\n{value:.4f}', (x_position, value),
                              textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    axis_top.set_ylabel(args.metric)
    axis_top.set_title(title)
    axis_top.legend(loc='best')
    axis_top.grid(True, linestyle='--', alpha=0.4)

    axis_bottom.axhline(0.0, color='black', linewidth=0.9)
    axis_bottom.bar(x_positions, subgroup_deltas, color=[
        '#d1495b' if np.isfinite(value) and value < 0 else '#2e7d32' for value in subgroup_deltas
    ])
    axis_bottom.set_ylabel(f'Delta from overall {args.metric}')
    axis_bottom.set_xticks(x_positions)
    axis_bottom.set_xticklabels(subgroup_names, rotation=30, ha='right')
    axis_bottom.grid(True, axis='y', linestyle='--', alpha=0.4)

    figure.tight_layout()
    figure.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(figure)

    record_type = 'Videos' if args.level == 'video' else 'Samples'
    print(f'Overall {args.metric} ({record_type}): {overall:.6f}')
    print('Subgroup | n | metric | delta')
    for row in rows:
        print(f"{row['subgroup']} | {row['n']} | {row['metric']:.6f} | {row['delta']:.6f}")
    print(f'Figure saved to: {output_path}')


if __name__ == '__main__':
    main()
