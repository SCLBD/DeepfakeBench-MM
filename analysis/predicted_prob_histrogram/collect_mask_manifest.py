"""Read predictions and collect paired original-group / actual-label AUCs."""

import argparse
import hashlib
import json

from pathlib import Path

from modality_video_auc import fixed_group_report

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--log-root', type=Path, required=True)
_parser.add_argument(
    '--masked-log-root',
    type=Path,
    help='Parent of Model_train_on_Mega-MMDF_mask_audio/video runs; '
    'default: log-root/mask_modality_test_bkp for historical compatibility',
)
_parser.add_argument('--datasets', nargs='+', help='Restrict collection to the named test datasets.')
_parser.add_argument('--output', type=Path, required=True)
_args = _parser.parse_args()
ROOT = _args.log_root
MASKED_ROOT = _args.masked_log_root or ROOT / 'mask_modality_test_bkp'
_args.output.parent.mkdir(parents=True, exist_ok=True)
MODELS = ['AVFF', 'AVH', 'Baseline', 'FGMDF', 'FRADE', 'FauForensics', 'MCL']
output = {'models': MODELS, 'records': [], 'issues': []}
for model in MODELS:
    full = ROOT / 'test' / (model + '_train_on_Mega-MMDF')
    for file in sorted((full / 'predictions').glob('*.json')):
        if _args.datasets is not None and file.stem not in _args.datasets:
            continue
        if file.stem not in {
            'DigiFakeAV',
            'FakeAVCeleb',
            'IDForge',
            'LAV-DF',
            'MMDF',
            'Mega-MMDF',
            'PolyGlotFake',
            'TalkingHeadBench',
        }:
            continue
        entry = {'model': model, 'training': 'Vanilla', 'dataset': file.stem, 'conditions': {}}
        reference = None
        for condition in ('full', 'mask_audio', 'mask_video'):
            path = (
                file
                if condition == 'full'
                else MASKED_ROOT / (model + '_train_on_Mega-MMDF_' + condition) / 'predictions' / file.name
            )
            rows = json.loads(path.read_bytes())
            source = 'saved' if condition == 'full' else 'original'
            prefix = '' if condition == 'full' else 'original_'
            # Order-independent matching of windows AND their original labels.
            keys = sorted((r['path'], r['sample_frame_range'], r[prefix + 'audio_label'], r[prefix + 'video_label']) for r in rows)
            fingerprint = hashlib.sha256(json.dumps(keys, separators=(',', ':')).encode()).hexdigest()
            del keys
            if reference is None:
                reference = fingerprint
            paired = reference == fingerprint
            _, results = fixed_group_report(rows, source)
            entry['conditions'][condition] = {
                'path': str(path),
                'windows': len(rows),
                'paired': paired,
                'fingerprint': fingerprint,
                'tasks': results,
            }
            if not paired:
                output['issues'].append(
                    {'model': model, 'dataset': file.stem, 'condition': condition, 'issue': 'Window / original-label mismatch'}
                )
            del rows
        output['records'].append(entry)
        print(model, file.stem, 'paired=' + str(all(c['paired'] for c in entry['conditions'].values())), flush=True)
        _args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
print('COMPLETE', len(output['records']), 'issues', len(output['issues']), flush=True)
