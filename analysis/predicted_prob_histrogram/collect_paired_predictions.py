"""Read-only export of paired video probabilities from a collection manifest.

The manifest is produced by collect_mask_manifest.py. Raw JSON files stay on
the machine running this script; only compact video-level pairs are exported.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import posixpath

from pathlib import Path


def read_condition(path, original, expected_fingerprint=None, masked_modality='audio'):
    rows = json.loads(Path(path).read_bytes())
    prefix = 'original_' if original else ''
    keys, videos = [], {}
    for r in rows:
        a, v = r[prefix + 'audio_label'], r[prefix + 'video_label']
        p = float(r['prob'])
        assert a in (0, 1) and v in (0, 1), 'Invalid original modality labels'
        assert r['audio_label'] in (0, 1) and r['video_label'] in (0, 1)
        assert r['label'] == max(r['audio_label'], r['video_label'])
        assert math.isfinite(p) and 0 <= p <= 1, 'Invalid probability'
        if original:
            expected = (0, v) if masked_modality == 'audio' else (a, 0)
            assert (r['audio_label'], r['video_label']) == expected, 'Not a repaired mask run'
        keys.append((r['path'], r['sample_frame_range'], a, v))
        vid = posixpath.dirname(r['path'].replace('\\', '/'))
        assert vid
        item = videos.setdefault(vid, {'audio': 0, 'video': 0, 'actual': 0, 'p': 0., 'n': 0})
        item['audio'] = max(item['audio'], a)
        item['video'] = max(item['video'], v)
        item['actual'] = max(item['actual'], r['label'])
        item['p'] = max(item['p'], p)
        item['n'] += 1
    digest = hashlib.sha256(json.dumps(sorted(keys), separators=(',', ':')).encode()).hexdigest()
    if expected_fingerprint:
        assert digest == expected_fingerprint, f'Source changed: {path}'
    return videos, digest, len(rows)


def collect(manifest, output, datasets=None, task_name='RAFV'):
    modality = 'audio' if task_name == 'RAFV' else 'video'
    remaining = 'video' if modality == 'audio' else 'audio'
    condition = 'mask_' + modality
    result = {'task': task_name, 'condition': condition,
              'protocol': f'RARV vs {task_name}; original-label OR before selection; dirname(path); max probability; {condition} - full deltas',
              'models': manifest['models'], 'records': [], 'skipped': []}
    for record in manifest['records']:
        model, dataset = record['model'], record['dataset']
        if datasets and dataset not in datasets:
            continue
        conditions = record['conditions']
        task = conditions['full']['tasks'][task_name]
        if not task['original_real'] or not task['original_target']:
            result['skipped'].append({'model': model, 'dataset': dataset, 'reason': f'Missing original RARV or {task_name} group'})
            continue
        full, fp0, n0 = read_condition(conditions['full']['path'], False, conditions['full']['fingerprint'])
        masked, fp1, n1 = read_condition(conditions[condition]['path'], True, conditions[condition]['fingerprint'], modality)
        assert fp0 == fp1 and n0 == n1 and full.keys() == masked.keys(), 'Unpaired windows/videos'
        pairs = []
        for vid, a in sorted(full.items()):
            b = masked[vid]
            assert (a['audio'], a['video'], a['n']) == (b['audio'], b['video'], b['n'])
            if a[modality] != 0:
                continue
            y = a[remaining]
            assert a['actual'] == b['actual'] == y, 'Scoring labels changed in selected group'
            pairs.append({'video_id': vid, 'y': y, 'full': a['p'], condition: b['p'], 'windows': a['n']})
        assert sum(p['y'] == 0 for p in pairs) == task['n_real']
        assert sum(p['y'] == 1 for p in pairs) == task['n_fake']
        result['records'].append({'model': model, 'dataset': dataset, 'task': task_name, 'condition': condition, 'fingerprint': fp0,
                                  'source_windows': n0, 'sources': {c: conditions[c]['path'] for c in ('full', condition)},
                                  'pairs': pairs})
        print(f'{model} / {dataset}: {len(pairs)} paired videos; windows verified', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, 'wt', encoding='utf-8') as f:
        json.dump(result, f, separators=(',', ':'))
    print(f'Exported {len(result["records"])} groups; skipped {len(result["skipped"])}: {output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--task', choices=['RAFV', 'FARV'], default='RAFV')
    args = p.parse_args()
    collect(json.loads(args.manifest.read_text(encoding='utf-8')), args.output, args.datasets, args.task)
