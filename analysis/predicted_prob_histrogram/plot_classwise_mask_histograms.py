"""Six class/mask conditions, original and masked distributions overlaid."""

import gzip
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/missing_modality_paper'
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
MODELS = ['Baseline', 'AVFF', 'AVH', 'FGMDF', 'FRADE', 'FauForensics', 'MCL']


def main():
    with gzip.open(ROOT / 'analysis/predicted_prob_histrogram/outputs/missing_groups_ID.json.gz', 'rt', encoding='utf-8') as f:
        data = json.load(f)
    records = {r['model']: r for r in data['records'] if r['training'] == 'Vanilla'}
    metrics_path = OUT / 'classwise_mask_metrics.json'
    if not metrics_path.is_file():
        metrics_path = FIXTURES / metrics_path.name
    verified = json.loads(metrics_path.read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 8,
            'pdf.fonttype': 42,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 0.6,
        }
    )
    fig = plt.figure(figsize=(7.16, 4.25))
    outer = fig.add_gridspec(2, 1, left=0.105, right=0.995, bottom=0.06, top=0.91, hspace=0.16)
    axes = []
    for t in range(2):
        grid = outer[t].subgridspec(3, 7, wspace=0.14, hspace=0.15)
        axes.append([[fig.add_subplot(grid[j, i]) for i in range(7)] for j in range(3)])
    for i, model in enumerate(MODELS):
        for t, (condition, groups) in enumerate(
            [('mask_audio', ['RARV', 'RAFV', 'FAFV']), ('mask_video', ['RARV', 'FARV', 'FAFV'])]
        ):
            for j, group in enumerate(groups):
                pairs = [p for p in records[model]['pairs'] if p['group'] == group]
                ax = axes[t][j][i]
                scores = {k: np.array([p[k] for p in pairs]) for k in ['full', condition]}
                label = 0 if group == 'RARV' else 1
                computed_mse = np.mean((scores[condition] - label) ** 2) - np.mean((scores['full'] - label) ** 2)
                expected = next(r for r in verified if r['model'] == model)
                expected = next(r for r in expected['conditions'] if r['condition'] == condition and r['group'] == group)
                assert abs(computed_mse - expected['delta_mse']) < 1e-10
                errors = {k: np.mean(p >= 0.5) if label == 0 else np.mean(p < 0.5) for k, p in scores.items()}
                assert abs(errors[condition] - errors['full'] - expected['delta_error_rate']) < 1e-10
                for k, color in [('full', '#2878B5'), (condition, '#D94B4B')]:
                    counts, edges = np.histogram(scores[k], bins=np.linspace(0, 1, 21))
                    mass = counts / counts.sum()
                    ax.stairs(mass, edges, color=color, fill=True, alpha=0.12, linewidth=0)
                    ax.stairs(mass, edges, color=color, lw=0.85, alpha=0.88)
                ax.axvline(0.5, color='#999999', ls=':', lw=0.55, zorder=0)
                ax.set(xlim=(-0.02, 1.025), ylim=(0, 1.04))
                ax.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
                ax.set_yticks([0, 0.5, 1], ['0', '0.5', '1'])
                ax.tick_params(length=2, pad=1, labelsize=6.5, labelleft=i == 0, labelbottom=t == 1 and j == 2)
                if i == 0:
                    ax.set_ylabel(group, fontsize=7, labelpad=4)
                if t == 0 and j == 0:
                    ax.set_title(model, fontsize=7, pad=4)
    for t, title in enumerate(['(a) Mask audio', '(b) Mask video']):
        box = outer[t].get_position(fig)
        fig.text(0.012, (box.y0 + box.y1) / 2, title, rotation=90, ha='center', va='center', fontsize=8)
    y = (outer[0].get_position(fig).y0 + outer[1].get_position(fig).y1) / 2
    fig.add_artist(Line2D([0.045, 0.995], [y, y], transform=fig.transFigure, color='#777777', lw=0.7, ls='--'))
    fig.legend(
        handles=[Line2D([], [], color='#2878B5', label='Full input'), Line2D([], [], color='#D94B4B', label='Masked input')],
        loc='upper center',
        bbox_to_anchor=(0.55, 1.005),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'classwise_mask_histograms.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(OUT / 'classwise_mask_histograms.png')


if __name__ == '__main__':
    main()
