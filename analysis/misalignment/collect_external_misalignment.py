"""Pair supplied shift0/shift1 windows; retain only actual non-overlap shifts."""

import argparse
import collections
import gzip
import hashlib
import json
import math
import numpy as np

from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def records(path):
    """Stream JSON arrays and JSONL, validating the complete input."""
    with path.open(encoding='utf-8-sig') as f:
        if path.suffix == '.jsonl':
            for line in f:
                if line.strip():
                    yield json.loads(line)
            return
        decoder = json.JSONDecoder()
        buf = f.read(65536).lstrip()
        if not buf.startswith('['):
            raise ValueError(f'Expected JSON array: {path}')
        buf = buf[1:]
        first = True
        while True:
            buf = buf.lstrip()
            if not buf:
                more = f.read(65536)
                if not more:
                    raise ValueError(f'Truncated array: {path}')
                buf += more
                continue
            if buf.startswith(']'):
                if buf[1:].strip() or f.read().strip():
                    raise ValueError(f'Trailing array data: {path}')
                return
            if not first:
                if not buf.startswith(','):
                    raise ValueError(f'Missing array separator: {path}')
                buf = buf[1:].lstrip()
            while True:
                buf = buf.lstrip()
                try:
                    row, end = decoder.raw_decode(buf)
                    break
                except json.JSONDecodeError:
                    more = f.read(65536)
                    if not more:
                        raise
                    buf += more
            yield row
            buf = buf[end:]
            first = False

def native(row):
    if row.get('status', 'ok') != 'ok':
        return None
    path = row.get('relative_video_path', row.get('path'))
    if not path:
        raise ValueError('Missing relative/native path')
    path = path.replace('\\', '/')
    frame = row.get('sample_frame_range')
    if frame:
        start, end = map(int, frame.split('_'))
    else:
        start, end = int(row['start_frame']), int(row['end_frame'])
    label = int(row['label'])
    a, v = int(row['audio_label']), int(row['video_label'])
    score = float(row['prob'] if 'prob' in row else row['fake_probability'])
    if label not in (0, 1) or a not in (0, 1) or v not in (0, 1) or label != (a | v):
        raise ValueError('Invalid native authenticity labels')
    if start >= end or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid native window or score')
    if 'real_probability' in row and abs(score + float(row['real_probability']) - 1) > 2e-6:
        raise ValueError('Binary softmax scores do not sum to one')
    return path, start, end, label, a, v, score


def pair(aligned, shifted, model):
    original = {}
    for row in records(aligned):
        r = native(row)
        if r is None:
            continue
        path, start, end, label, a, v, score = r
        key = path, start, end
        if key in original or (label, a, v) != (0, 0, 0):
            raise ValueError('Duplicate or non-real aligned input')
        original[key] = row, score
    seen, deltas = set(), []
    counts = collections.Counter()
    for row in records(shifted):
        r = native(row)
        if r is None:
            counts['shifted_error_attempts'] += 1
            continue
        path, start, end, label, a, v, score = r
        key = path, start, end
        if key in seen or (label, a, v) != (0, 0, 0):
            raise ValueError('Duplicate or non-real shifted input')
        seen.add(key)
        if key not in original:
            counts['missing_aligned_window'] += 1
            continue
        before, p = original[key]
        if not row.get('misalignment_applied', False):
            counts['unshifted_windows_excluded'] += 1
            continue
        if int(row['num_samples_in_data']) < 2 or row['original_sample_index']==row['donor_sample_index']:
            raise ValueError('Applied shift has no distinct donor')
        vrange = tuple(map(int,row['video_sample_frame_range'].split('_')))
        arange = tuple(map(int,row['audio_sample_frame_range'].split('_')))
        if min(vrange[1],arange[1]) > max(vrange[0],arange[0]):
            raise ValueError('Shifted windows overlap in source time')
        if before.get('misalignment_applied',False):
            raise ValueError('Shift0 is not aligned')
        if before['video_sample_frame_range'] != before['audio_sample_frame_range']:
            raise ValueError('Shift0 modality ranges differ')
        if vrange != (start,end):
            raise ValueError('Video anchor changed in audio-shift experiment')
        deltas.append(score-p)
    counts['aligned_successful_windows'] = len(original)
    counts['shifted_successful_windows'] = len(seen)
    counts['aligned_without_shifted'] = len(original.keys()-seen)
    if counts['missing_aligned_window'] or counts['aligned_without_shifted']:
        raise ValueError('Shift0/shift1 successful cohorts do not match')
    x = np.asarray(deltas)
    return {'model':model,'deltas':deltas,'n':len(deltas), 'mean':float(x.mean()),
            'median':float(np.median(x)), 'p05':float(np.quantile(x,.05)),
            'p95':float(np.quantile(x,.95)), 'fraction_outside_display':float((abs(x)>.1).mean()),
            'counts':dict(counts),'aligned_source':str(aligned.resolve()),
            'shifted_source':str(shifted.resolve()),'aligned_sha256':digest(aligned),
            'shifted_sha256':digest(shifted),'level':'paired real windows, applied shifts only'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--existing-pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    values=collections.defaultdict(list)
    with gzip.open(args.existing_pairs,'rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['applied']:values[r['model']].append(r['misaligned']-r['aligned'])
    out=[]
    for m,delta in values.items():
        x=np.asarray(delta)
        out.append(dict(model=m,deltas=delta,n=len(delta),mean=float(x.mean()),median=float(np.median(x)),
                        p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95)),
                        fraction_outside_display=float((abs(x)>.1).mean()),
                        source=str(args.existing_pairs.resolve()),source_sha256=digest(args.existing_pairs),
                        level='existing audited paired real windows, applied shifts only'))
    for model,prefix,suffix in [('SpeechForensics','SpeechForensics','.json'),('Qwen2.5-Omni','Qwen','.jsonl'),('VideoLLaMA2.1','VideoLLaMA','.jsonl')]:
        out.append(pair(args.input_root/f'{prefix}_shift0{suffix}', args.input_root/f'{prefix}_shift1{suffix}',model))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.output,'wt',encoding='utf8') as f:
        json.dump({'records':out,'definition':'shifted fake score minus aligned fake score; real windows only',
                   'inference_performed':False},f,separators=(',',':'),allow_nan=False)
    summary=[{k:v for k,v in r.items() if k!='deltas'} for r in out]
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps([{k:r[k] for k in ['model','n','mean','median']} for r in summary]))


if __name__=='__main__':main()
