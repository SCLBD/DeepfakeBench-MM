"""Plot video-level AUC as a function of available training data."""

import argparse
import json

from pathlib import Path

if __package__:
    from .extract import FRACTIONS, MODELS, MODULE_DIR, OUT, SHEET, extract
else:
    from extract import FRACTIONS, MODELS, MODULE_DIR, OUT, SHEET, extract


def plot():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator

    data_path = OUT / 'data.json'
    if not data_path.exists():
        data_path = MODULE_DIR / 'fixtures' / 'data.json'
    data = json.loads(data_path.read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    records = data['records']
    colors = ['#555555', '#0072B2', '#D55E00', '#009E73', '#8B62A3', '#C74576', '#9A6A13']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 8.5,
            'axes.labelsize': 9,
            'axes.titlesize': 9.5,
            'legend.fontsize': 8,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 0.65,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
        }
    )

    def panel(ax, metric, title, relative=False):
        for model, color, marker in zip(MODELS, colors, markers):
            rs = sorted((r for r in records if r['model'] == model), key=lambda r: r['fraction'])
            baseline = rs[-1][metric] if relative else 0
            xs = [r['fraction'] * 100 for r in rs]
            ys = [r[metric] - baseline for r in rs]
            ax.plot(xs, ys, color=color, linewidth=1.25, alpha=0.93, zorder=2)
            for x, y, r in zip(xs, ys, rs):
                ax.plot(
                    x,
                    y,
                    linestyle='none',
                    marker=marker,
                    markersize=4.6,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=0.85,
                    zorder=3,
                )
        ax.set_xscale('log')
        ax.set_xlim(0.85, 118)
        ax.xaxis.set_major_locator(FixedLocator([f * 100 for f in FRACTIONS]))
        ax.set_xticklabels(['1%', '2%', '5%', '10%', '20%', '50%', '100%'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(title, loc='left', pad=5)
        ax.grid(axis='y', color='#DADDE2', linewidth=0.55, zorder=0)
        ax.grid(axis='x', color='#EEEEEE', linewidth=0.45, zorder=0)
        ax.tick_params(length=3)
        if relative:
            ax.axhline(0, color='#777777', linewidth=0.7, linestyle='--')
            ax.set_ylim(-0.20, 0.012)
            ax.set_ylabel('AUC difference from 100%')
        else:
            ax.set_ylabel('Video-level AUC' if metric == 'within' else 'Mean video-level AUC')
            ax.set_ylim((0.82, 1.006) if metric == 'within' else (0.50, 0.88))
            if metric == 'within':
                ax.yaxis.set_major_locator(FixedLocator([0.85, 0.90, 0.95, 1.00]))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    handles = [
        Line2D([], [], color=c, marker=m, linewidth=1.25, markersize=4, label=model) for model, c, m in zip(MODELS, colors, markers)
    ]

    def save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / f'{name}.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.04)
        plt.close(fig)

    for relative, name in [(False, 'data_scale'), (True, 'data_scale_delta')]:
        # Designed at its final single-column width, rather than shrinking
        # a double-column figure (and all its typography) in the manuscript.
        fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.12))
        panel(axs[0], 'within', '(a) Within-dataset', relative)
        panel(axs[1], 'cross', '(b) Cross-dataset', relative)
        for ax in axs:
            ax.set_ylabel('')
            ax.set_xticklabels(['1', '2', '5', '10', '20', '50', '100'])
            ax.set_title(ax.get_title(loc='left'), loc='left', fontsize=8, pad=4)
            ax.tick_params(axis='both', labelsize=6.5, length=2, pad=2)
            for line in ax.lines:
                line.set_linewidth(1.0)
                line.set_markersize(3.0)
                line.set_markeredgewidth(0.5)
        fig.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.50, 1.01),
            ncol=7,
            frameon=False,
            fontsize=6.5,
            markerscale=0.8,
            columnspacing=0.45,
            handletextpad=0.25,
            handlelength=0.8,
            borderaxespad=0,
        )
        axs[0].set_ylabel('AUC difference from 100%' if relative else 'Video-level AUC', fontsize=8, labelpad=2)
        fig.supxlabel('Available data proportion (%, log scale)', y=0.005, fontsize=8)
        fig.subplots_adjust(left=0.135, right=0.985, bottom=0.18, top=0.81, wspace=0.34)
        save(fig, name)
    for metric, title in [('within', 'Within-dataset: Mega-MMDF'), ('cross', 'Cross-dataset: average over 7')]:
        fig, ax = plt.subplots(figsize=(3.5, 2.85))
        panel(ax, metric, title)
        fig.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.015),
            ncol=3,
            frameon=False,
            columnspacing=0.9,
            handlelength=1.3,
        )
        ax.set_xlabel('Available data proportion (%, log scale)')
        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.17, top=0.70)
        save(fig, f'data_scale_{metric}')
    lines = [
        '# Training data scale: figure and data audit',
        '',
        f"Source: `{data['workbook']}`, worksheet `{SHEET}`, A8:P56.",
        'The workbook was read only. Plots use recorded C and D values exactly, per user request. Recomputed means are audit information only.',
        '',
        '## Retained source status (all recorded points used as requested)',
        '',
    ]
    for r in records:
        if r['provisional']:
            lines.append(f"- {r['model']} {r['fraction']:.0%}: row {r['source_row']}; `{r['remote_log']}`.")
    for d in data['discrepancies']:
        lines.append(
            f"- {d['cell']}: recorded {d['recorded']:.4f}; mean of the seven source scores = {d['recomputed']:.4f}. Plots use the recorded value as explicitly requested; workbook unchanged."
        )
    lines += [
        '',
        '## Endpoint changes',
        '',
        '| Model | Within: 1% | Within: 100% | Gain | Cross: 1% | Cross: 100% | Gain |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for model in MODELS:
        rs = sorted((r for r in records if r['model'] == model), key=lambda r: r['fraction'])
        a, b = rs[0], rs[-1]
        lines.append(
            f"| {model} | {a['within']:.4f} | {b['within']:.4f} | {b['within']-a['within']:+.4f} | {a['cross']:.4f} | {b['cross']:.4f} | {b['cross']-a['cross']:+.4f} |"
        )
    lines += [
        '',
        '## Interpretation and scope',
        '',
        '- Use absolute AUC in the main figure; the delta figure is supplementary and uses AUC(fraction) minus AUC(100%), not a ratio.',
        '- X positions are logarithmic, with actual percentages labeled. Lines connect measured results without smoothing or a fitted scaling law.',
        '- All seven 100% endpoints exceed all their smaller-scale points on both metrics. Intermediate improvements are not uniformly monotonic.',
        '- Four points are provisional according to source status. No repeated-run uncertainty is supplied, so no error bars or statistical significance are asserted.',
        '- Pair sampling is stratified by demographic group according to the user-provided protocol. Pair-retention percentage is not a verified exact clip/sample count percentage.',
        '- Protocol clarification: the selected proportion applies to both training and validation data. Reported evaluation uses the full test sets at every scale, not proportionally reduced test subsets.',
        '- The pair membership, nestedness of subsets, and equivalence of full-data vs subset training budgets have not been audited here. Fixed epochs would increase optimization steps with dataset size.',
        '- Cross-dataset targets are fixed for every point. Four targets participate in the shared validation protocol; this is not an unseen-only mean.',
        '- 100% values come from the scaling worksheet as requested, rather than silently substituting rounded values from the manuscript.',
        '',
        '![Absolute AUC](data_scale.png)',
        '',
        '![AUC difference](data_scale_delta.png)',
    ]
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[lines.index('## Endpoint changes') : lines.index('## Interpretation and scope')]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workbook', type=Path, help='Required when extracting; plotting can use the bundled data.json snapshot.')
    ap.add_argument('--extract-only', action='store_true')
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()
    if not args.plot_only and args.workbook is None:
        ap.error('--workbook is required unless --plot-only is used')
    if not args.plot_only:
        extract(args.workbook)
    if not args.extract_only:
        plot()
