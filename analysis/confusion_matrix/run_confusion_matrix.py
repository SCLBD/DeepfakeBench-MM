"""Draw a real-vs-fake confusion matrix from one prediction JSON."""

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Draw real-vs-fake confusion matrix from predictions/<dataset>.json.'
    )
    parser.add_argument('--log_dir', type=Path, required=True,
                        help='Test log directory containing predictions/*.json.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name, must match predictions/<dataset_name>.json.')
    parser.add_argument('--level', type=str, default='sample', choices=['sample', 'video'],
                        help='Aggregation level. Default: sample.')
    parser.add_argument('--output_dir', type=Path,
                        default=PROJECT_ROOT / 'analysis' / 'confusion_matrix' / 'outputs',
                        help='Output directory. Default: analysis/confusion_matrix/outputs.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold for prob. Default: 0.5.')
    parser.add_argument('--normalize', type=str, default='none',
                        choices=['none', 'true', 'pred', 'all'],
                        help='Normalization mode for cell colors/labels. Default: none.')
    parser.add_argument('--title', type=str, default=None,
                        help='Figure title. Default: "<dataset> - Confusion Matrix".')
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


def main() -> None:
    args = parse_args()
    records = load_records(args.log_dir, args.dataset_name)
    if args.level == 'video':
        records = aggregate_video_records(records)
    labels = np.asarray([record['label'] for record in records], dtype=np.int64)
    probabilities = np.asarray([record['prob'] for record in records], dtype=np.float64)
    predictions = (probabilities > args.threshold).astype(np.int64)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    if args.normalize == 'true':
        normalized = matrix.astype(np.float64) / matrix.sum(axis=1, keepdims=True)
    elif args.normalize == 'pred':
        normalized = matrix.astype(np.float64) / matrix.sum(axis=0, keepdims=True)
    elif args.normalize == 'all':
        normalized = matrix.astype(np.float64) / matrix.sum()
    else:
        normalized = matrix

    class_names = ['Real', 'Fake']
    title = args.title or f'{args.dataset_name} - Confusion Matrix'
    output_path = args.output_dir / f'{title}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(normalized, cmap='Blues', vmin=0.0, vmax=1.0 if args.normalize != 'none' else None)
    axis.set_xticks([0, 1])
    axis.set_xticklabels([f'Predicted {name}' for name in class_names])
    axis.set_yticks([0, 1])
    axis.set_yticklabels([f'True {name}' for name in class_names])
    axis.set_xlabel('Predicted label')
    axis.set_ylabel('True label')
    axis.set_title(title)

    total = int(matrix.sum())
    for row in range(2):
        for column in range(2):
            count = int(matrix[row, column])
            if args.normalize == 'none':
                text = f'{count}\n({count / total:.1%})'
            else:
                text = f'{count}\n({normalized[row, column]:.1%})'
            axis.text(column, row, text, ha='center', va='center', fontsize=11,
                      color='white' if normalized[row, column] > 0.5 else 'black')
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(figure)

    record_type = 'Videos' if args.level == 'video' else 'Samples'
    print(f'{record_type}: {len(labels)}')
    print(f'Threshold: {args.threshold}')
    print('Confusion matrix [TN, FP; FN, TP]:')
    print(matrix)
    print(f'Accuracy: {accuracy_score(labels, predictions):.6f}')
    print(f'Precision: {precision_score(labels, predictions, zero_division=0):.6f}')
    print(f'Recall: {recall_score(labels, predictions, zero_division=0):.6f}')
    print(f'F1: {f1_score(labels, predictions, zero_division=0):.6f}')
    print(f'Figure saved to: {output_path}')


if __name__ == '__main__':
    main()
