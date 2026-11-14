"""Read-only ID export, preserving all original video groups for new protocols."""

import argparse
import gzip
import json

from pathlib import Path

from collect_paired_predictions import read_condition

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--training', nargs='+', choices=['Vanilla', 'RMA', 'RMV'], default=['Vanilla'])
p.add_argument('--random-mask-root', type=Path, help='Required only for RMA/RMV inputs')
args = p.parse_args()
if any(t != 'Vanilla' for t in args.training) and args.random_mask_root is None:
    p.error('--random-mask-root is required for RMA/RMV; use --training Vanilla for the paper inputs')
manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
out = {'models': manifest['models'], 'records': [], 'dataset': 'Mega-MMDF'}
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    with gzip.open(args.output, 'rt', encoding='utf-8') as f:
        out = json.load(f)
    if out['models'] != manifest['models'] or out['dataset'] != 'Mega-MMDF':
        p.error('Cached output uses a different set of models or dataset. Choose a new output file.')
root = args.random_mask_root
for model in out['models']:
    ref = next((r for r in manifest['records'] if r['model'] == model and r['dataset'] == 'Mega-MMDF'), None)
    if ref is None:
        p.error(f'Missing Mega-MMDF predictions in the manifest for {model}.')
    expected = ref['conditions']['full']['fingerprint']
    for training in args.training:
        cached = next((r for r in out['records'] if r['model'] == model and r['training'] == training), None)
        if cached is not None:
            if cached['fingerprint'] != expected:
                p.error(f'Cached record for {model}/{training} uses different windows. Choose a new output file.')
            continue
        runs, sources = {}, {}
        for mode in ('full', 'mask_audio', 'mask_video'):
            if training == 'Vanilla':
                path = Path(ref['conditions'][mode]['path'])
            else:
                augmentation = 'audio' if training == 'RMA' else 'video'
                suffix = 'original' if mode == 'full' else mode
                path = root / f'{model}_train_on_Mega-MMDF_random_mask_{augmentation}_{suffix}' / 'predictions/Mega-MMDF.json'
            runs[mode], _, n = read_condition(path, mode != 'full', expected, mode.removeprefix('mask_'))
            sources[mode] = str(path)
            print(f'{model} {training} {mode}: verified {n} windows', flush=True)
        pairs = []
        assert runs['full'].keys() == runs['mask_audio'].keys() == runs['mask_video'].keys()
        for vid, a in sorted(runs['full'].items()):
            for mode in ('mask_audio', 'mask_video'):
                b = runs[mode][vid]
                assert (a['audio'], a['video'], a['n']) == (b['audio'], b['video'], b['n'])
                assert b['actual'] == a['video' if mode == 'mask_audio' else 'audio']
            group = ('FA' if a['audio'] else 'RA') + ('FV' if a['video'] else 'RV')
            pairs.append({'video_id': vid, 'group': group, 'windows': a['n'], **{mode: run[vid]['p'] for mode, run in runs.items()}})
        out['records'].append({'model': model, 'training': training, 'fingerprint': expected, 'sources': sources, 'pairs': pairs})
        temporary = args.output.with_name(args.output.name + '.partial')
        with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as f:
            json.dump(out, f, separators=(',', ':'))
        temporary.replace(args.output)
print(f'Completed {len(out["records"])} model/checkpoint records', flush=True)
