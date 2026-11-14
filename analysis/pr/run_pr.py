"""Draw Precision-Recall curve with AP annotation from one prediction JSON."""

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
from sklearn import metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Draw Precision-Recall curve from predictions/<dataset>.json.')
    parser.add_argument('--log_dir', type=Path, required=True,
                        help='Test log directory containing predictions/*.json.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name, must match predictions/<dataset_name>.json.')
    parser.add_argument('--level', type=str, default='sample', choices=['sample', 'video'],
                        help='Aggregation level. Default: sample.')
    parser.add_argument('--output_dir', type=Path,
                        default=PROJECT_ROOT / 'analysis' / 'pr' / 'outputs',
                        help='Output directory. Default: analysis/pr/outputs.')
    parser.add_argument('--title', type=str, default=None,
                        help='Figure title. Default: "<dataset> - Precision-Recall".')
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


def compute_pr(records: list[dict]):
    labels = np.asarray([record['label'] for record in records], dtype=np.int64)
    probabilities = np.asarray([record['prob'] for record in records], dtype=np.float64)
    if len(np.unique(labels)) < 2:
        raise SystemExit(f'{len(records)} records found, but both real and fake labels are required for PR.')

    precision, recall, thresholds = metrics.precision_recall_curve(labels, probabilities)
    ap = metrics.average_precision_score(labels, probabilities)
    positive_ratio = float(np.mean(labels))
    return labels, probabilities, precision, recall, thresholds, ap, positive_ratio


def plot_pr(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: float,
    positive_ratio: float,
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    axis.step(recall, precision, where='post', linewidth=2.0,
              label=f'Precision-Recall (AP = {ap:.4f})')
    axis.axhline(positive_ratio, linestyle='--', color='gray', linewidth=1.0,
                 label=f'Positive ratio = {positive_ratio:.4f}')
    axis.set_xlabel('Recall')
    axis.set_ylabel('Precision')
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_title(title)
    axis.legend(loc='upper right')
    axis.grid(True, linestyle='--', alpha=0.4)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    args = parse_args()
    records = load_records(args.log_dir, args.dataset_name)
    if args.level == 'video':
        records = aggregate_video_records(records)
    labels, probabilities, precision, recall, thresholds, ap, positive_ratio = compute_pr(records)
    title = args.title or f'{args.dataset_name} - Precision-Recall'
    output_path = args.output_dir / f'{title}.png'
    plot_pr(precision, recall, ap, positive_ratio, output_path, title, args.dpi)

    record_type = 'Videos' if args.level == 'video' else 'Samples'
    print(f'{record_type}: {len(labels)}')
    print(f'AP: {ap:.6f}')
    print(f'Positive ratio: {positive_ratio:.6f}')
    print(f'Figure saved to: {output_path}')


if __name__ == '__main__':
    main()
