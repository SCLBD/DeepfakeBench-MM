"""Append three Mega-MMDF modality-task video-level AUCs to a test log.

Usage:
    python analysis/predicted_prob_histrogram/modality_video_auc.py --log_dir /path/to/test_log_directory

Labels describe the union of saved evaluation windows, not necessarily the
original video's entire duration. Aggregate ALL windows before selecting tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import posixpath

from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score

STATES = {(0, 0): 'RARV', (0, 1): 'RAFV', (1, 0): 'FARV', (1, 1): 'FAFV'}


def aggregate_videos(records: list[dict], label_source: str = 'saved') -> dict[str, dict]:
    """Match metrics/auc.py: parent-directory ID, OR labels, max probability."""
    if label_source not in ('saved', 'original'):
        raise ValueError('label_source must be saved or original')
    prefix = 'original_' if label_source == 'original' else ''
    videos = {}
    for index, record in enumerate(records):
        try:
            path = record['path']
            audio = record[prefix + 'audio_label']
            video = record[prefix + 'video_label']
            prob = float(record['prob'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid prediction record {index}: {exc}') from exc
        if not isinstance(path, str) or not path:
            raise ValueError(f'Record {index}: path must be a nonempty string.')
        if audio not in (0, 1) or video not in (0, 1):
            raise ValueError(f'Record {index}: modality labels must be 0 or 1.')
        if not math.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError(f'Record {index}: prob must be finite and in [0, 1].')
        video_id = posixpath.dirname(path.replace('\\', '/'))
        if not video_id:
            raise ValueError(f'Record {index}: path has no video parent directory.')
        if video_id not in videos:
            videos[video_id] = {'audio': 0, 'video': 0, 'prob': 0.0}
        item = videos[video_id]
        item['audio'] = max(item['audio'], int(audio))
        item['video'] = max(item['video'], int(video))
        item['prob'] = max(item['prob'], prob)
    return videos


def format_report(videos: dict[str, dict], sample_count: int, label_source: str = 'saved') -> str:
    groups = {state: [] for state in STATES.values()}
    path_disagreements = 0
    for video_id, item in videos.items():
        state = STATES[item['audio'], item['video']]
        groups[state].append(item['prob'])
        path_states = [part for part in video_id.split('/') if part in groups]
        if path_states and path_states[0] != state:
            path_disagreements += 1

    lines = [
        f'[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Mega-MMDF modality-task video-level AUC:',
        'Source: predictions/Mega-MMDF.json',
        'Aggregation: dirname(path); audio/video labels=OR; fake probability=max.',
        f'Grouping: OR of {label_source} audio/video labels; path categories are not used; this calculator does not relabel.',
        f'Samples: {sample_count}; videos: {len(videos)}; path-category disagreements: {path_disagreements}',
        'Video counts: ' + ', '.join(f'{state}={len(probs)}' for state, probs in groups.items()),
        '+----------------+------------+------------+------------+',
        '| Task           | Real       | Fake       | AUC        |',
        '+----------------+------------+------------+------------+',
    ]
    real = groups['RARV']
    for state in ('RAFV', 'FARV', 'FAFV'):
        fake = groups[state]
        if real and fake:
            auc = roc_auc_score([0] * len(real) + [1] * len(fake), real + fake)
            value = f'{auc:.6f}'
        else:
            value = 'N/A'
        lines.append(f'| {"RARV vs " + state:<14} | {len(real):>10} | {len(fake):>10} | {value:>10} |')
    lines.append('+----------------+------------+------------+------------+')
    if any(not probs for probs in groups.values()):
        lines.append('N/A means the task has no real or no fake videos; AUC is undefined.')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log_dir', '--log-dir', type=Path, required=True,
                        help='Directory containing predictions/Mega-MMDF.json; append to log.txt here.')
    parser.add_argument('--label-source', choices=('saved', 'original'), default=None,
                        help='Grouping labels only. Auto: original fields if present; saved only for unmasked runs. AUC always uses saved actual labels.')
    args = parser.parse_args()
    prediction_path = args.log_dir / 'predictions' / 'Mega-MMDF.json'
    try:
        with prediction_path.open(encoding='utf-8') as handle:
            records = json.load(handle)
        if not isinstance(records, list) or not records:
            raise ValueError('Prediction JSON must be a nonempty list.')
        masked_run = 'mask_audio' in args.log_dir.name or 'mask_video' in args.log_dir.name
        source = args.label_source or ('original' if masked_run or any('original_audio_label' in r for r in records) else 'saved')
        if masked_run and source != 'original':
            raise ValueError('Masked runs require original grouping labels.')
        report, _ = fixed_group_report(records, source)
        log_path = args.log_dir / 'log.txt'
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write('\n' + report + '\n')
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(report)
    print(f'Appended to: {log_path}')


def fixed_group_report(records, grouping_source='original'):
    """Select by original video group, score against actual post-mask video labels."""
    original = aggregate_videos(records, grouping_source)
    actual = aggregate_videos(records, 'saved')
    for i, row in enumerate(records):
        if row['label'] not in (0, 1) or row['label'] != max(row['audio_label'], row['video_label']):
            raise ValueError(f'Inconsistent actual overall label at record {i}')
    groups = {s: [] for s in STATES.values()}
    for key, value in original.items():
        groups[STATES[value['audio'], value['video']]].append(key)
    results = {}
    lines = [
        f'[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Mega-MMDF fixed-group / actual-label video-level AUC (v2):',
        'This section supersedes earlier modality-task AUC sections for this protocol; overall dataset metrics are separate.',
        'Source: predictions/Mega-MMDF.json',
        f'Grouping: OR of {grouping_source} audio/video window labels, before task selection; never path categories.',
        'Scoring: OR of saved actual post-mask labels; probability=max across ALL saved windows of a video.',
        'Real/Fake counts below refer to actual scoring labels, not original group membership.',
        f'Windows: {len(records)}; videos: {len(original)}',
        'Original group video counts: ' + ', '.join(f'{s}={len(v)}' for s, v in groups.items()),
        '| Task | Actual real | Actual fake | AUC |',
    ]
    for state in ('RAFV', 'FARV', 'FAFV'):
        keys = groups['RARV'] + groups[state]
        labels = [max(actual[k]['audio'], actual[k]['video']) for k in keys]
        probs = [actual[k]['prob'] for k in keys]
        n_fake = sum(labels)
        n_real = len(labels) - n_fake
        # Both original groups must exist, and the scoring labels must have both classes.
        auc = float(roc_auc_score(labels, probs)) if groups['RARV'] and groups[state] and n_fake and n_real else None
        results[state] = {'n_real': n_real, 'n_fake': n_fake, 'auc': auc,
                          'original_real': len(groups['RARV']), 'original_target': len(groups[state])}
        value = 'N/A' if auc is None else f'{auc:.6f}'
        lines.append(f'| RARV vs {state} | {n_real} | {n_fake} | {value} |')
    lines.append('N/A: missing original comparison group, or actual labels contain fewer than two classes.')
    return '\n'.join(lines), results


if __name__ == '__main__':
    main()
