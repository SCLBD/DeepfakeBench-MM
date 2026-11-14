"""Draw ROC curve with AUC and EER annotations from one prediction JSON."""

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
    parser = argparse.ArgumentParser(description='Draw ROC curve from predictions/<dataset>.json.')
    parser.add_argument('--log_dir', type=Path, required=True,
                        help='Test log directory containing predictions/*.json.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name, must match predictions/<dataset_name>.json.')
    parser.add_argument('--level', type=str, default='sample', choices=['sample', 'video'],
                        help='Aggregation level. Default: sample.')
    parser.add_argument('--output_dir', type=Path,
                        default=PROJECT_ROOT / 'analysis' / 'roc' / 'outputs',
                        help='Output directory. Default: analysis/roc/outputs.')
    parser.add_argument('--title', type=str, default=None,
                        help='Figure title. Default: "<dataset> - ROC".')
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


def compute_roc(records: list[dict]):
    labels = np.asarray([record['label'] for record in records], dtype=np.int64)
    probabilities = np.asarray([record['prob'] for record in records], dtype=np.float64)
    if len(np.unique(labels)) < 2:
        raise SystemExit(f'{len(records)} records found, but both real and fake labels are required for ROC.')

    fpr, tpr, thresholds = metrics.roc_curve(labels, probabilities)
    auc = metrics.roc_auc_score(labels, probabilities)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float(fpr[eer_index])
    eer_threshold = float(thresholds[eer_index])
    return labels, probabilities, fpr, tpr, auc, eer, eer_threshold


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    eer: float,
    eer_threshold: float,
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    axis.plot(fpr, tpr, linewidth=2.0, label=f'ROC (AUC = {auc:.4f})')
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle='--', color='gray', linewidth=1.0,
              label='Random guess')

    eer_line_x = np.linspace(0.0, 1.0, 200)
    axis.plot(eer_line_x, 1.0 - eer_line_x, linestyle=':', color='#d1495b',
              linewidth=1.2, label='EER line')
    axis.scatter([eer], [1.0 - eer], color='#d1495b', zorder=3, label='EER point')
    axis.annotate(
        f'EER = {eer:.4f}\nThreshold = {eer_threshold:.4f}',
        xy=(eer, 1.0 - eer),
        xytext=(0.62, 0.32),
        arrowprops={'arrowstyle': '->', 'color': '#d1495b'},
        fontsize=9,
    )

    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_title(title)
    axis.legend(loc='lower right')
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
    labels, probabilities, fpr, tpr, auc, eer, eer_threshold = compute_roc(records)
    title = args.title or f'{args.dataset_name} - ROC'
    output_path = args.output_dir / f'{title}.png'
    plot_roc(fpr, tpr, auc, eer, eer_threshold, output_path, title, args.dpi)

    record_type = 'Videos' if args.level == 'video' else 'Samples'
    print(f'{record_type}: {len(labels)}')
    print(f'AUC: {auc:.6f}')
    print(f'EER: {eer:.6f}')
    print(f'EER threshold: {eer_threshold:.6f}')
    print(f'Figure saved to: {output_path}')


if __name__ == '__main__':
    main()
