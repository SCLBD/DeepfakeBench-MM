"""Plot paired Full/Mask Audio distributions and export Brier/gap metrics."""

import argparse

from pathlib import Path

from core import export_metrics, load_pairs, plot_probabilities


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True, help='collect_paired_predictions.py output (.json.gz)')
    p.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'outputs')
    p.add_argument('--reference', type=Path, help='Optional previous AUC manifest; require exact agreement')
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--bin-width', type=float, default=.01)
    p.add_argument('--bandwidth', type=float, default=.02)
    args = p.parse_args()
    if not 0 < args.bin_width <= .1 or args.bandwidth <= 0:
        p.error('Require 0 < bin-width <= 0.1 and bandwidth > 0')
    data = load_pairs(args.pairs)
    if data.get('task', 'RAFV') != 'RAFV':
        p.error('This plot layout is for RAFV/mask audio; use run_mask_video_analysis.py for FARV metrics.')
    if args.datasets:
        data['records'] = [r for r in data['records'] if r['dataset'] in args.datasets]
    export_metrics(data, args.output_dir, args.reference)
    plot_probabilities(data, args.output_dir, args.bin_width, args.bandwidth)


if __name__ == '__main__':
    main()
