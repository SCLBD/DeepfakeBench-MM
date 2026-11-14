"""Compact single-column paired response histograms from audited predictions."""

import gzip
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'analysis/misalignment/outputs'
OUT = ROOT / 'outputs/misalignment_paper'
OUT.mkdir(parents=True, exist_ok=True)
models = ['Baseline', 'AVFF', 'AVH', 'FGMDF', 'FRADE', 'FauForensics', 'MCL']
values = {m: [] for m in models}
with gzip.open(SRC / 'pairs.jsonl.gz', 'rt') as f:
    for line in f:
        r = json.loads(line)
        if r['applied']:
            values[r['model']].append(r['misaligned'] - r['aligned'])
summary = json.loads((SRC / 'summary.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(4, 2, figsize=(3.5, 4.0), sharex=True, sharey=True)
for i, m in enumerate(models):
    ax = axs.flat[i]
    d = np.array(values[m])
    assert len(d) == summary[m]['n']
    assert abs(d.mean() - summary[m]['mean_delta']) < 1e-12
    h, edges = np.histogram(d, bins=np.linspace(-1, 1, 201))
    ax.stairs(100 * h / len(d), edges, fill=True, color='#3978a8', linewidth=0.45)
    ax.axvline(0, color='#555555', lw=0.6, ls=':')
    ax.axvline(d.mean(), color='#D16B26', lw=0.9, ls='--')
    ax.set_title(f'({chr(97+i)}) {m}' + ('*' if m == 'AVFF' else ''), pad=2, fontsize=7)
    ax.text(0.03, 0.95, f'n = {len(d):,}\nmean = {d.mean():+.4f}', transform=ax.transAxes, va='top', fontsize=5.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 65)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([0, 30, 60])
    ax.tick_params(length=2, pad=1, labelsize=6)
    if i in [5, 6]:
        ax.tick_params(labelbottom=True)
axs.flat[-1].axis('off')
axs.flat[-1].legend(
    handles=[
        Line2D([], [], color='#3978a8', lw=3, label='Paired change'),
        Line2D([], [], color='#D16B26', ls='--', lw=1, label='Mean change'),
        Line2D([], [], color='#555555', ls=':', lw=1, label='Zero'),
    ],
    loc='center',
    frameon=False,
    fontsize=6.5,
    handlelength=1.6,
)
fig.supxlabel('Fake-probability change (misaligned − aligned)', fontsize=7, y=0.014)
fig.supylabel('Windows per bin (%)', fontsize=7, x=0.014)
fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.09, hspace=0.55, wspace=0.22)
for ext in ['png', 'pdf']:
    fig.savefig(OUT / f'misalignment_prob_change.{ext}', dpi=600)
plt.close(fig)
print(OUT / 'misalignment_prob_change.png')
