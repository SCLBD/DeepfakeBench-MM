"""Ten-method paired-real-window response figure from audited compact deltas."""

import argparse
import csv
import gzip
import hashlib
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

MODELS = ['Baseline', 'AVFF', 'AVH', 'FauForensics', 'FGMDF', 'FRADE', 'MCL', 'SpeechForensics', 'Qwen2.5-Omni', 'VideoLLaMA2.1']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent / 'fixtures/ten_method_deltas.json.gz')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs/ten_methods')
    args = parser.parse_args()
    with gzip.open(args.input, 'rt', encoding='utf8') as f:
        records = json.load(f)['records']
    lookup = {r['model']: r for r in records}
    assert set(lookup) == set(MODELS)
    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 8,
            'pdf.fonttype': 42,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )
    fig, axes = plt.subplots(2, 5, figsize=(7.16, 2.75))
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.19, top=0.89, wspace=0.25, hspace=0.65)
    bins = np.linspace(-0.1, 0.3, 81)
    rows = []
    for i, m in enumerate(MODELS):
        ax = axes.flat[i]
        r = lookup[m]
        d = np.asarray(r['deltas'])
        assert len(d) == r['n'] and abs(d.mean() - r['mean']) < 1e-12
        h, e = np.histogram(d, bins)
        ax.stairs(100 * h / len(d), e, fill=True, color='#3978a8', alpha=0.65, lw=0.5)
        ax.axvline(0, color='#666666', ls=':', lw=0.6)
        ax.axvline(d.mean(), color='#CE6F27', ls='--', lw=0.85)
        ax.set(xlim=(-0.1, 0.3), ylim=(0, 55))
        ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3], ['−.1', '0', '.1', '.2', '.3'])
        ax.set_yticks([0, 25, 50])
        ax.tick_params(length=2, pad=1, labelsize=6.5, labelleft=i % 5 == 0)
        if i % 5 == 0:
            ax.set_ylabel('Proportion (%)', fontsize=7, labelpad=2)
        ax.set_title(f'({chr(97+i)}) {m}', fontsize=7.5, pad=15)
        mean = 0.0 if abs(d.mean()) < 0.00005 else d.mean()
        median = 0.0 if abs(np.median(d)) < 0.00005 else np.median(d)
        ax.text(0.5, 1.035, f'Mean {mean:+.4f}  Med. {median:+.4f}', transform=ax.transAxes, ha='center', va='bottom', fontsize=5.3)
        right = ax.twinx()
        v, c = np.unique(d, return_counts=True)
        right.step(v, 100 * np.cumsum(c) / len(d), where='post', color='#74518C', lw=0.85)
        right.set_ylim(0, 100)
        right.set_yticks([0, 50, 100])
        show = i % 5 == 4
        right.spines['right'].set_visible(show)
        right.tick_params(right=show, labelright=show, length=2, pad=1, labelsize=6.5, colors='#74518C')
        if show:
            right.set_ylabel('Cumulative (%)', fontsize=7, color='#74518C', labelpad=2)
        rows.append(
            dict(
                model=m,
                n=len(d),
                mean=float(d.mean()),
                median=float(np.median(d)),
                p05=float(np.quantile(d, 0.05)),
                p95=float(np.quantile(d, 0.95)),
                fraction_outside_display=float(((d < -0.1) | (d > 0.3)).mean()),
            )
        )
    fig.supxlabel('Sample-level score change (misaligned − aligned)', fontsize=8, y=0.04)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(args.output_dir / f'misalignment_sample_hist_cdf.{ext}', dpi=600, facecolor='white')
    plt.close(fig)
    with (args.output_dir / 'misalignment_summary.csv').open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'provenance.json').write_text(
        json.dumps(
            {
                'input': str(args.input.resolve()),
                'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'display_range': [-0.1, 0.3],
                'histogram_denominator': 'all included paired windows, including outside view',
                'cdf_denominator': 'all included paired windows',
                'summary': rows,
            },
            indent=2,
        ),
        encoding='utf8',
    )
    caption = (
        'Responses to within-clip audio reassignment on real data for seven Mega-MMDF-trained detectors and three training-free methods. '
        'Only paired windows with an applied non-overlapping shift are included. Blue histograms show the change in fake score '
        '(misaligned minus aligned); orange dashed lines show means, and purple curves show empirical cumulative distributions. '
        'The horizontal view is [-0.1,0.3]; both the histogram and CDF are normalized over all included windows. '
        'Input lengths and eligible window cohorts are method-specific.'
    )
    (args.output_dir / 'caption.txt').write_text(caption, encoding='utf8')
    (args.output_dir / 'figure.tex').write_text(
        '\\begin{figure*}[t]\n    \\centering\n'
        '    \\includegraphics[width=\\textwidth]{images_new/misalignment_sample_hist_cdf.png}\n'
        '    \\caption{' + caption + '}\n    \\label{fig:misalignment-ten-methods}\n\\end{figure*}\n',
        encoding='utf8',
    )
    print(args.output_dir / 'misalignment_sample_hist_cdf.png')


if __name__ == '__main__':
    main()
