"""FAFV paired masking metrics and compact probability histograms."""

import gzip
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/missing_modality_paper'
MODELS = ['Baseline', 'AVFF', 'AVH', 'FGMDF', 'FRADE', 'FauForensics', 'MCL']


def metrics(y, p):
    real, fake = p[~y], p[y]
    return dict(
        auc=float(roc_auc_score(y, p)),
        brier=float((np.mean(real**2) + np.mean((1 - fake) ** 2)) / 2),
        fpr=float(np.mean(real >= 0.5)),
        fnr=float(np.mean(fake < 0.5)),
        w1=float(wasserstein_distance(real, fake)),
    )


def fmt(x):
    return '$0.000$' if round(x, 3) == 0 else f'${x:+.3f}$'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(ROOT / 'analysis/predicted_prob_histrogram/outputs/missing_groups_ID.json.gz', 'rt', encoding='utf-8') as f:
        data = json.load(f)
    records = {r['model']: r for r in data['records'] if r['training'] == 'Vanilla'}
    previous = json.loads((ROOT / 'analysis/predicted_prob_histrogram/fixtures/modality_reliance_metrics.json').read_text(encoding='utf-8'))
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
    fig, axes = plt.subplots(3, 7, figsize=(7.16, 2.05))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.09, top=0.89, wspace=0.14, hspace=0.20)
    results = []
    for i, model in enumerate(MODELS):
        pairs = [p for p in records[model]['pairs'] if p['group'] in ['RARV', 'FAFV']]
        y = np.array([p['group'] == 'FAFV' for p in pairs])
        row = dict(model=model, n_real=int((~y).sum()), n_fake=int(y.sum()))
        for j, key in enumerate(['full', 'mask_audio', 'mask_video']):
            p = np.array([v[key] for v in pairs])
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            row[key] = metrics(y, p)
            if j:
                row['delta_' + key] = {m: row[key][m] - row['full'][m] for m in row[key]}
                old = next(
                    r
                    for r in previous
                    if r['model'] == model and r['dataset'] == 'Mega-MMDF' and r['condition'] == key and r['training'] == 'Vanilla'
                )
                assert row['n_fake'] == old['n_fake'] and row['n_real'] == old['n_real']
                for m, oldkey in [('auc', 'auc_'), ('brier', 'brier_balanced_'), ('w1', 'wasserstein_')]:
                    assert abs(row[key][m] - old[oldkey + key]) < 1e-10, (model, key, m)
            ax = axes[j, i]
            for label, color in [(False, '#2878B5'), (True, '#D94B4B')]:
                counts, edges = np.histogram(p[y == label], bins=np.linspace(0, 1, 21))
                mass = counts / counts.sum()
                ax.stairs(mass, edges, color=color, fill=True, alpha=0.13, linewidth=0)
                ax.stairs(mass, edges, color=color, lw=0.95, alpha=0.8)
            ax.axvline(0.5, color='#9A9A9A', lw=0.6, ls=':', zorder=0)
            ax.set(xlim=(-0.02, 1.025), ylim=(0, 1.04))
            ax.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
            ax.set_yticks([0, 0.5, 1], ['0', '0.5', '1'])
            ax.tick_params(length=2, pad=1, labelsize=6.5, labelleft=i == 0, labelbottom=j == 2)
            if i == 0:
                ax.set_ylabel(['Full input', 'Mask audio', 'Mask video'][j], fontsize=6.5, labelpad=4)
            if j == 0:
                ax.set_title(model, fontsize=7, pad=4)
        results.append(row)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'fafv_input_distributions_all7.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    (OUT / 'fafv_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    headers = r'$\Delta$AUC $\uparrow$ & $\Delta$Brier $\downarrow$ & $\Delta$FPR@0.5 $\downarrow$ & $\Delta$FNR@0.5 $\downarrow$ & $\Delta W_1$'
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Changes in video-level detection metrics for RARV vs.\ FAFV on Mega-MMDF. Each entry reports $\Delta=\mathrm{masked}-\mathrm{full}$ on the same clips. Arrows indicate the preferable direction of change; Brier is class-balanced. $\Delta W_1$ describes the change in distribution distance without assigning a preferable direction.}',
        r'\label{tab:fafv_masking}',
        r'\setlength{\tabcolsep}{2.5pt}',
        r'\renewcommand{\arraystretch}{1.12}',
        r'\scriptsize',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc|ccccc@{}}',
        r'\toprule',
        r'& \multicolumn{5}{c|}{Mask audio: MARV vs MAFV} & \multicolumn{5}{c}{Mask video: RAMV vs FAMV} \\',
        r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}',
        'Detector & ' + headers + ' & ' + headers + r' \\',
        r'\midrule',
    ]
    for row in results:
        m = row['model']
        name = m + (r'~\cite{' + m + '}' if m != 'Baseline' else '')
        vals = [
            fmt(row['delta_' + key][metric])
            for key in ['mask_audio', 'mask_video']
            for metric in ['auc', 'brier', 'fpr', 'fnr', 'w1']
        ]
        lines.append(' & '.join([name] + vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table*}']
    (OUT / 'fafv_mask_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
