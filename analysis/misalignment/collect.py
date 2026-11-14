"""Pair real-only misalignment predictions with original predictions (read-only)."""

import argparse
import collections
import gzip
import hashlib
import json
import math

from pathlib import Path

MODELS = ['AVFF', 'AVH', 'Baseline', 'FGMDF', 'FRADE', 'FauForensics', 'MCL']


def config(path):
    text = path.read_text()
    return json.JSONDecoder().raw_decode(text.split('Config: ', 1)[1])[0]

def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original-root', type=Path, required=True)
    p.add_argument('--shifted-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = {}
    with gzip.open(args.output, 'wt', encoding='utf-8') as out:
        for model in MODELS:
            a = args.original_root / (model + '_train_on_Mega-MMDF')
            b = args.shifted_root / (model + '_train_on_Mega-MMDF_shift_audio_1')
            ca, cb = config(a/'log.txt'), config(b/'log.txt')
            diffs = {k: {'original': ca.get(k), 'shifted': cb.get(k)}
                     for k in set(ca)|set(cb) if ca.get(k) != cb.get(k)}
            shifted = json.loads((b/'predictions/Mega-MMDF.json').read_bytes())
            keyed = {}
            for r in shifted:
                key = (r['path'], r['sample_frame_range'])
                assert key not in keyed, ('duplicate shifted', key)
                assert (r['label'], r['audio_label'], r['video_label']) == (0,0,0)
                assert r['video_sample_frame_range'] == r['sample_frame_range']
                assert r['misalignment_applied'] == (r['audio_sample_frame_range'] != r['video_sample_frame_range'])
                keyed[key] = r
            original_file = a/'predictions/Mega-MMDF.json'
            original = json.loads(original_file.read_bytes())
            seen = set()
            counts = collections.Counter()
            for r in original:
                key = (r['path'], r['sample_frame_range'])
                if key not in keyed:
                    continue
                assert key not in seen, ('duplicate original matched key', key)
                seen.add(key)
                s = keyed[key]
                assert (r['label'], r['audio_label'], r['video_label']) == (0,0,0)
                assert all(math.isfinite(x) and 0 <= x <= 1 for x in (r['prob'], s['prob']))
                counts['shifted' if s['misalignment_applied'] else 'unchanged'] += 1
                record = dict(model=model, path=r['path'], sample_frame_range=r['sample_frame_range'],
                              aligned=r['prob'], misaligned=s['prob'], applied=s['misalignment_applied'],
                              audio_range=s['audio_sample_frame_range'], num_samples=s['num_samples_in_data'])
                out.write(json.dumps(record, separators=(',', ':')) + '\n')
            missing = sorted(set(keyed)-seen)
            audit[model] = dict(counts=counts, shifted_total=len(shifted), original_total=len(original),
                                unmatched_count=len(missing), unmatched_examples=missing[:10], config_differences=diffs,
                                sources=[str(original_file), str(b/'predictions/Mega-MMDF.json')],
                                sha256=[fingerprint(original_file), fingerprint(b/'predictions/Mega-MMDF.json')])
            print(model, dict(counts), 'unmatched', len(missing), 'config differences', list(diffs), flush=True)
            del original, shifted, keyed
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2))

if __name__ == '__main__':
    main()
