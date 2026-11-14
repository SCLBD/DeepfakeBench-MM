"""Zoomed sample histograms and full-sample empirical CDFs; dual axes."""

import gzip
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/misalignment_paper'
MODELS = ['Baseline', 'AVFF', 'AVH', 'FGMDF', 'FRADE', 'FauForensics', 'MCL']
values = {m: [] for m in MODELS}
with gzip.open(ROOT / 'analysis/misalignment/outputs/pairs.jsonl.gz', 'rt') as f:
    for line in f:
        r = json.loads(line)
        if r['applied']:
            values[r['model']].append(r['misaligned'] - r['aligned'])
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
fig, axes = plt.subplots(1, 7, figsize=(7.16, 1.25))
fig.subplots_adjust(left=0.058, right=0.944, bottom=0.27, top=0.82, wspace=0.18)
bins = np.linspace(-0.1, 0.1, 41)
for i, m in enumerate(MODELS):
    ax = axes.flat[i]
    d = np.array(values[m])
    h, e = np.histogram(d, bins)
    ax.stairs(h / len(d) * 100, e, fill=True, color='#3978a8', alpha=0.65, lw=0.5)
    ax.axvline(0, color='#555555', lw=0.65, ls=':')
    ax.axvline(d.mean(), color='#CE6F27', lw=1, ls='--')
    ax.set(xlim=(-0.1, 0.1), ylim=(0, 32))
    ax.set_xticks([-0.1, 0, 0.1], ['−0.1', '0', '0.1'])
    ax.set_yticks([0, 10, 20, 30])
    ax.tick_params(length=2, pad=1, labelsize=7, labelleft=i == 0)
    if i == 0:
        ax.set_ylabel('Proportion (%)', fontsize=8, labelpad=3)
    ax.set_title(m, fontsize=8, pad=3)
    mean = 0.0 if abs(d.mean()) < 0.00005 else d.mean()
    ax.text(0.03, 0.96, f'Mean {mean:+.4f}', transform=ax.transAxes, va='top', fontsize=6.5)
    right = ax.twinx()
    v, c = np.unique(d, return_counts=True)
    right.step(v, np.cumsum(c) / len(d) * 100, where='post', color='#74518C', lw=0.95)
    right.set_ylim(0, 100)
    right.set_yticks([0, 50, 100])
    show_right = i == 6
    right.spines['right'].set_visible(show_right)
    right.tick_params(right=show_right, labelright=show_right, length=2, pad=1, labelsize=7, colors='#74518C')
    if show_right:
        right.set_ylabel('Cumulative (%)', fontsize=8, color='#74518C', labelpad=3)
fig.supxlabel('Sample-level probability change (misaligned − aligned)', fontsize=8, y=0.025)
for ext in ['png', 'pdf']:
    fig.savefig(OUT / f'misalignment_sample_hist_cdf.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.04)
plt.close(fig)
print(OUT / 'misalignment_sample_hist_cdf.png')
