"""Analyze RARV vs FARV / mask video, without rerunning model inference."""

import argparse
import json

from pathlib import Path

from core import export_metrics, load_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'outputs' / 'FARV_mask_video')
    args = p.parse_args()
    data = load_pairs(args.pairs)
    assert data['task'] == 'FARV' and data['condition'] == 'mask_video'
    rows = export_metrics(data, args.output_dir, args.reference)
    summary = json.loads((args.output_dir / 'metrics.json').read_text())['domain_macro']
    for r in summary:
        print(r['model'], r['domain'], 'dp(real/fake)', round(r['delta_mean_real'], 4), round(r['delta_mean_fake'], 4),
              'dB(real/fake/bal)', round(r['delta_brier_real'], 4), round(r['delta_brier_fake'], 4), round(r['delta_brier_balanced'], 4),
              'dAUC_pp', round(r['delta_auc'] * 100, 3))
    counts = {}
    for domain, rs in [('ID', [r for r in rows if r['dataset'] == 'Mega-MMDF']),
                       ('OOD', [r for r in rows if r['dataset'] != 'Mega-MMDF'])]:
        counts[domain] = {'n': len(rs)}
        for key in ['delta_mean_real', 'delta_mean_fake', 'delta_brier_real', 'delta_brier_fake', 'delta_brier_balanced', 'delta_auc']:
            counts[domain][key] = {'positive': sum(r[key] > 0 for r in rs), 'negative': sum(r[key] < 0 for r in rs)}
    (args.output_dir / 'direction_counts.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
