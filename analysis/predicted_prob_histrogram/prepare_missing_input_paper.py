"""Recompute paired video metrics and prepare missing-input manuscript assets."""

import gzip
import hashlib
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import wasserstein_distance

from core import auc

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/missing_modality_paper'
MODELS = ['Baseline', 'AVFF', 'AVH', 'FGMDF', 'FRADE', 'FauForensics', 'MCL']
TASKS = [('RAFV', 'mask_audio'), ('FARV', 'mask_video')]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = ROOT / 'analysis/predicted_prob_histrogram/outputs/missing_groups_ID.json.gz'
    refs = json.loads((ROOT / 'analysis/predicted_prob_histrogram/fixtures/three_settings_metrics.json').read_text())
    ref = {(r['model'], r['condition']): r for r in refs if r['training'] == 'Vanilla' and r['subset'] == 'single'}
    print('Loading paired cache', flush=True)
    with gzip.open(src, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    records = {r['model']: r for r in data['records'] if r['training'] == 'Vanilla'}
    result = []
    arrays = {}
    for fake, cond in TASKS:
        for m in MODELS:
            sel = [p for p in records[m]['pairs'] if p['group'] in ['RARV', fake]]
            assert len({p['video_id'] for p in sel}) == len(sel)
            y = np.array([p['group'] != 'RARV' for p in sel])
            full = np.array([p['full'] for p in sel])
            mask = np.array([p[cond] for p in sel])
            arrays[m, fake] = (y, full, mask)
            assert (int(sum(~y)), int(sum(y))) == (ref[m, cond]['n_real'], ref[m, cond]['n_fake'])
            r = dict(model=m, fake=fake, mask=cond, n_real=int(sum(~y)), n_fake=int(sum(y)))
            for name, p in [('full', full), ('mask', mask)]:
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                r[name] = dict(
                    auc=auc(y, p),
                    fpr=float(np.mean(p[~y] >= 0.5)),
                    fnr=float(np.mean(p[y] < 0.5)),
                    brier=float(0.5 * (np.mean(p[~y] ** 2) + np.mean((1 - p[y]) ** 2))),
                    w1=float(wasserstein_distance(p[~y], p[y])),
                )
                for a, b in [('auc', 'auc'), ('brier', 'brier_balanced'), ('w1', 'wasserstein')]:
                    assert abs(r[name][a] - ref[m, cond][b + '_' + ('full' if name == 'full' else cond)]) < 1e-12
            r['delta'] = {k: r['mask'][k] - r['full'][k] for k in r['full']}
            result.append(r)
            print(fake, m, 'FPR', r['full']['fpr'], r['mask']['fpr'], 'FNR', r['full']['fnr'], r['mask']['fnr'], flush=True)
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT / 'provenance.json').write_text(
        json.dumps(
            dict(
                source=str(src),
                sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                threshold=0.5,
                rule='fake if p >= 0.5',
                scope='Vanilla; Mega-MMDF; paired video-level; original-label grouping',
                checks='Unique video IDs; counts, AUC, balanced Brier and W1 reproduce prior metrics to 1e-12',
                bin_width=0.01,
            ),
            indent=2,
        ),
        encoding='utf-8',
    )
    table = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Single-modality input via zero-filling on Mega-MMDF. Each cell reports full $\rightarrow$ masked input. FPR and FNR use a fixed threshold of 0.5; Brier is class-balanced.}',
        r'\label{tab:missing_modality}',
        r'\setlength{\tabcolsep}{5pt}',
        r'\renewcommand{\arraystretch}{1.12}',
        r'\small',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lccccc}',
        r'\toprule',
        r'Detector & AUC $\uparrow$ & FPR $\downarrow$ & FNR $\downarrow$ & Brier $\downarrow$ & $W_1$ \\',
        r'\midrule',
    ]
    # Full-width table: keep each before/after pair on one line.
    for fake, cond in TASKS:
        table += [
            r'\multicolumn{6}{c}{RARV vs ' + fake + (' (mask audio)' if cond == 'mask_audio' else ' (mask video)') + r'} \\',
            r'\midrule',
        ]
        for r in sorted([r for r in result if r['fake'] == fake], key=lambda r: MODELS.index(r['model'])):
            cells = [
                f'{r["full"][k]:.3f}' + r' $\rightarrow$ ' + f'{r["mask"][k]:.3f}' for k in ['auc', 'fpr', 'fnr', 'brier', 'w1']
            ]
            name = r['model'] + (r'~\cite{' + r['model'] + '}' if r['model'] != 'Baseline' else '')
            table.append(name + ' & ' + ' & '.join(cells) + r' \\')
        table.append(r'\midrule' if fake == 'RAFV' else r'\bottomrule')
    table += [r'\end{tabular}%', r'}', r'\end{table*}']
    (OUT / 'missing_input_table.tex').write_text('\n'.join(table), encoding='utf-8')
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 6.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.titleweight': 'normal',
        }
    )
    selected = [('FRADE', 'RAFV'), ('FauForensics', 'FARV'), ('FGMDF', 'FARV')]
    for all_models in [False, True]:
        examples = [(m, f) for f, c in TASKS for m in MODELS] if all_models else selected
        fig, axes = plt.subplots(len(examples), 2, figsize=(7.16, 14) if all_models else (3.5, 3.5), squeeze=False)
        for i, (m, fake) in enumerate(examples):
            y, full, mask = arrays[m, fake]
            for j, p in enumerate([full, mask]):
                ax = axes[i, j]
                for label, color in [(False, '#2878B5'), (True, '#D94B4B')]:
                    h, edges = np.histogram(p[y == label], bins=np.linspace(0, 1, 101))
                    ax.stairs(h / h.sum(), edges, color=color, lw=0.8)
                ax.axvline(0.5, color='#777777', lw=0.6, ls=':')
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(0, 1.03)
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([0, 0.5, 1])
                ax.tick_params(pad=1, length=2)
                ax.set_title(
                    f'({chr(97+i*2+j)}) {m}: ' + ('Full' if j == 0 else ('Video-only' if fake == 'RAFV' else 'Audio-only')),
                    fontsize=6.5,
                    pad=2,
                )
                if j == 0:
                    ax.set_ylabel(f'{fake}\nFraction / bin', fontsize=6.3, labelpad=1)
                if i == len(examples) - 1:
                    ax.set_xlabel('Predicted fake probability', fontsize=6.5, labelpad=1)
        fig.legend(
            handles=[Line2D([], [], color=c, lw=1, label=n) for c, n in [('#2878B5', 'Real'), ('#D94B4B', 'Fake')]],
            loc='upper center',
            ncol=2,
            frameon=False,
            fontsize=7,
            bbox_to_anchor=(0.55, 1.005),
        )
        fig.subplots_adjust(
            left=0.13,
            right=0.985,
            top=0.91 if not all_models else 0.975,
            bottom=0.105 if not all_models else 0.035,
            hspace=0.58,
            wspace=0.29,
        )
        stem = 'all_missing_input_histograms' if all_models else 'missing_input_histograms'
        fig.savefig(OUT / f'{stem}.png', dpi=600 if not all_models else 250)
        fig.savefig(OUT / f'{stem}.pdf')
        plt.close(fig)
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
