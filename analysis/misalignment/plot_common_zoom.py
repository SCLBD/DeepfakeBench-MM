"""Common-range zooms with explicit excluded mass and all-data normalization."""

import gzip
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'analysis/misalignment/outputs'
OUT = ROOT / 'outputs/misalignment_paper'
MODELS = ['Baseline', 'AVFF', 'AVH', 'FGMDF', 'FRADE', 'FauForensics', 'MCL']
values = {m: [] for m in MODELS}
clips = {m: defaultdict(list) for m in MODELS}
with gzip.open(SRC / 'pairs.jsonl.gz', 'rt') as f:
    for line in f:
        r = json.loads(line)
        if r['applied']:
            d = r['misaligned'] - r['aligned']
            values[r['model']].append(d)
            clips[r['model']][r['path']].append(d)
summary = json.loads((SRC / 'summary.json').read_text())
arrays = {
    'window': {m: np.asarray(values[m]) for m in MODELS},
    'clip': {m: np.asarray([np.mean(ds) for ds in clips[m].values()]) for m in MODELS},
}
audit = {}
for unit, group in arrays.items():
    audit[unit] = {}
    for m, d in group.items():
        expected = summary[m]['mean_delta' if unit == 'window' else 'clip_macro_delta']
        assert abs(d.mean() - expected) < 1e-12
        audit[unit][m] = {
            'n': len(d),
            'mean': float(d.mean()),
            'median': float(np.median(d)),
            'q05_q95': np.quantile(d, [0.05, 0.95]).tolist(),
            'outside_percent': {str(lim): float(100 * np.mean(np.abs(d) > lim)) for lim in [0.05, 0.1, 0.15, 0.2, 0.3]},
        }
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
for unit, group in arrays.items():
    fig, axes = plt.subplots(1, 7, figsize=(7.16, 1.65), sharex=True, sharey=True)
    limit = 0.1 if unit == 'window' else 0.15
    bins = np.linspace(-limit, limit, 41 if unit == 'window' else 61)
    maxheight = max((np.histogram(d, bins)[0] / len(d)).max() * 100 for d in group.values())
    for ax, m in zip(axes, MODELS):
        d = group[m]
        counts, edges = np.histogram(d, bins)
        ax.stairs(counts / len(d) * 100, edges, fill=True, color='#3978a8', alpha=0.75, lw=0.5)
        ax.axvline(0, color='#555555', ls=':', lw=0.7)
        ax.axvline(d.mean(), color='#CF722D', ls='--', lw=1)
        ax.set_title(m + ('*' if m == 'AVFF' else ''), fontsize=7, pad=3)
        mean = 0.0 if abs(d.mean()) < 0.00005 else d.mean()
        annotation = f'Mean {mean:+.4f}'
        if unit != 'window':
            annotation += f'\nOut {100*np.mean(abs(d)>limit):.1f}%'
        ax.text(0.03, 0.96, annotation, transform=ax.transAxes, va='top', fontsize=5.8)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(0, maxheight * 1.42)
        ax.set_xticks([-0.1, 0, 0.1], ['−0.1', '0', '0.1'])
        ax.tick_params(labelsize=6, length=2, pad=1)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.29, top=0.76, wspace=0.16)
    axes[0].set_ylabel('Proportion (%)', fontsize=7, labelpad=3)
    xlabel = 'Sample-level probability change' if unit == 'window' else 'Within-clip mean probability change'
    fig.supxlabel(xlabel + ' (misaligned − aligned)', fontsize=8, y=0.075)
    fig.legend(
        handles=[
            Line2D([], [], color='#3978a8', lw=3, label='Paired change'),
            Line2D([], [], color='#CF722D', ls='--', lw=1, label='Mean'),
            Line2D([], [], color='#555555', ls=':', lw=0.7, label='Zero'),
        ],
        loc='upper center',
        bbox_to_anchor=(0.54, 1.035),
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    for ext in ['png', 'pdf']:
        name = 'misalignment_sample_zoom_01' if unit == 'window' else 'misalignment_clip_common_zoom'
        fig.savefig(OUT / f'{name}.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
(OUT / 'common_zoom_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
for unit in arrays:
    print(unit)
    for m, s in audit[unit].items():
        print(
            m,
            'mean',
            round(s['mean'], 6),
            'within +/-0.05',
            round(100 - s['outside_percent']['0.05'], 2),
            'within +/-0.1',
            round(100 - s['outside_percent']['0.1'], 2),
        )
