"""Shared plotting and probability metrics for paired RARV/RAFV predictions."""

from __future__ import annotations

import csv
import gzip
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata

REAL, FAKE = '#2878B5', '#D94B4B'
COLORS = (REAL, FAKE)
LABELS = ('RARV (real)', 'RAFV (fake)')
plt.rcParams.update(
    {
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titleweight': 'semibold',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
)


def load_pairs(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    for r in data['records']:
        pairs = r['pairs']
        assert len({p['video_id'] for p in pairs}) == len(pairs)
        y = np.array([p['y'] for p in pairs], dtype=int)
        full = np.array([p['full'] for p in pairs], dtype=float)
        mask = np.array([p[r.get('condition', 'mask_audio')] for p in pairs], dtype=float)
        assert set(y) == {0, 1} and np.isfinite(full).all() and np.isfinite(mask).all()
        assert ((0 <= full) & (full <= 1) & (0 <= mask) & (mask <= 1)).all()
        r['arrays'] = y, full, mask
    return data


def auc(y, p):
    n1, n0 = np.sum(y == 1), np.sum(y == 0)
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(record):
    y, full, mask = record['arrays']
    masked_condition = record.get('condition', 'mask_audio')
    out = {'model': record['model'], 'dataset': record['dataset'], 'n_real': int(sum(y == 0)), 'n_fake': int(sum(y == 1))}
    for condition, p in [('full', full), (masked_condition, mask)]:
        error = (p - y) ** 2
        out[f'auc_{condition}'] = auc(y, p)
        out[f'brier_{condition}'] = float(error.mean())
        out[f'brier_balanced_{condition}'] = float((error[y == 0].mean() + error[y == 1].mean()) / 2)
        out[f'gap_{condition}'] = float(p[y == 1].mean() - p[y == 0].mean())
        for label, name in [(0, 'real'), (1, 'fake')]:
            out[f'mean_{name}_{condition}'] = float(p[y == label].mean())
            out[f'brier_{name}_{condition}'] = float(error[y == label].mean())
    for name in ['auc', 'brier', 'brier_balanced', 'gap', 'mean_real', 'mean_fake', 'brier_real', 'brier_fake']:
        out[f'delta_{name}'] = out[f'{name}_{masked_condition}'] - out[f'{name}_full']
    dp = mask - full
    for label, name in [(0, 'real'), (1, 'fake')]:
        d = dp[y == label]
        out[f'delta_p_{name}_median'] = float(np.median(d))
        out[f'delta_p_{name}_q05'] = float(np.quantile(d, 0.05))
        out[f'delta_p_{name}_q95'] = float(np.quantile(d, 0.95))
        out[f'delta_p_{name}_mae'] = float(np.abs(d).mean())
        out[f'mask_hurts_fraction_{name}'] = float(np.mean(d > 0 if label == 0 else d < 0))
        out[f'exactly_unchanged_fraction_{name}'] = float(np.mean(d == 0))
    return out


def export_metrics(data, output, reference=None):
    rows = [metrics(r) for r in data['records']]
    if reference:
        refs = {(r['model'], r['dataset']): r for r in json.loads(Path(reference).read_text(encoding='utf-8'))['records']}
        for r in rows:
            for c in ('full', data.get('condition', 'mask_audio')):
                old = refs[r['model'], r['dataset']]['conditions'][c]['tasks'][data.get('task', 'RAFV')]
                assert abs(r[f'auc_{c}'] - old['auc']) < 1e-12, f'AUC mismatch: {r}'
                assert (r['n_real'], r['n_fake']) == (old['n_real'], old['n_fake'])

    def write_csv(name, content):
        with (output / name).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(content[0]))
            w.writeheader()
            w.writerows(content)

    output.mkdir(parents=True, exist_ok=True)
    write_csv('metrics_by_dataset.csv', rows)
    summary = []
    for model in data['models']:
        for domain in ('In-domain', 'Out-of-domain'):
            subset = [r for r in rows if r['model'] == model and (r['dataset'] == 'Mega-MMDF') == (domain == 'In-domain')]
            if not subset:
                continue
            item = {
                'model': model,
                'domain': domain,
                'datasets': '; '.join(r['dataset'] for r in subset),
                'n_datasets': len(subset),
                'n_real_total': sum(r['n_real'] for r in subset),
                'n_fake_total': sum(r['n_fake'] for r in subset),
            }
            for key in rows[0]:
                if key not in ('model', 'dataset', 'n_real', 'n_fake'):
                    item[key] = float(np.mean([r[key] for r in subset]))
            summary.append(item)
    write_csv('metrics_domain_macro.csv', summary)
    (output / 'metrics.json').write_text(
        json.dumps({'by_dataset': rows, 'domain_macro': summary, 'skipped': data['skipped']}, indent=2), encoding='utf-8'
    )
    return rows


def density(ax, values, color, bounds, bin_width, bandwidth):
    lo, hi = bounds
    edges = np.linspace(lo, hi, round((hi - lo) / bin_width) + 1)
    counts, _ = np.histogram(values, bins=edges)
    heights = counts / (len(values) * np.diff(edges))
    ax.stairs(heights, edges, fill=True, alpha=0.24, color=color, linewidth=0.8)
    ax.stairs(heights, edges, color=color, linewidth=0.65, alpha=0.7)
    # Gaussian binned KDE with reflected boundaries, fixed bandwidth shared
    # across all models/classes/conditions. No out-of-support density is drawn.
    step = 0.001
    fine_edges = np.linspace(lo, hi, round((hi - lo) / step) + 1)
    fine, _ = np.histogram(values, bins=fine_edges)
    smooth = gaussian_filter1d(fine.astype(float), bandwidth / step, mode='reflect')
    smooth /= len(values) * step
    ax.plot((fine_edges[:-1] + fine_edges[1:]) / 2, smooth, color=color, lw=1.5)
    ax.axvline(float(np.mean(values)), color=color, ls='--', lw=1.0, alpha=0.85)


def panel(ax, y, p, delta=False, bin_width=0.01, bandwidth=0.02, masked_modality='audio'):
    bounds = (-1, 1) if delta else (0, 1)
    for label, color in enumerate(COLORS):
        density(ax, p[y == label], color, bounds, bin_width, bandwidth)
    ax.set_xlim(*bounds)
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.linspace(*bounds, 5 if delta else 6))
    if delta:
        ax.axvline(0, color='#333333', lw=0.8, zorder=0)
    ax.set_ylabel('Within-class density')
    ax.set_xlabel(f'Change in fake probability (mask {masked_modality} - full)' if delta else 'Predicted fake probability')
    ax.grid(axis='y', alpha=0.15)
    ax.set_axisbelow(True)


def legend(fig, width, bandwidth, task='RAFV'):
    handles = [Line2D([0], [0], color=c, lw=3, label=l) for c, l in zip(COLORS, ('RARV (real)', f'{task} (fake)'))]
    handles.append(Line2D([0], [0], color='#555555', ls='--', label='Class mean'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.008,
        f'Video-level max probability | bin width {width:g} | reflected Gaussian KDE bandwidth {bandwidth:g} | each class normalized separately',
        ha='center',
        fontsize=9,
        color='#555555',
    )


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=180, facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)


def plot_probabilities(data, output, width=0.01, bandwidth=0.02, balanced_brier=False, auc_decimals=4):
    task = data.get('task', 'RAFV')
    masked_condition = data.get('condition', 'mask_audio')
    modality = masked_condition.removeprefix('mask_')
    brier_metric = 'brier_balanced' if balanced_brier else 'brier'
    brier_label = 'Balanced Brier' if balanced_brier else 'Brier'
    for dataset in sorted({r['dataset'] for r in data['records']}, key=lambda x: (x != 'Mega-MMDF', x)):
        records = [r for model in data['models'] for r in data['records'] if r['model'] == model and r['dataset'] == dataset]
        fig, axes = plt.subplots(len(records), 2, figsize=(12, 2.7 * len(records) + 1.2), squeeze=False)
        fig.suptitle(f'{dataset} | RARV vs {task} | Vanilla checkpoints', y=0.994, fontsize=16)
        for i, r in enumerate(records):
            y, full, mask = r['arrays']
            m = metrics(r)
            single, sa = plt.subplots(1, 2, figsize=(12, 4.4))
            single.suptitle(f'{r["model"]} | {dataset} | RARV vs {task}', y=0.995, fontsize=14)
            for j, (condition, p, title) in enumerate([('full', full, 'Full input'), (masked_condition, mask, 'Mask ' + modality)]):
                for ax in (axes[i, j], sa[j]):
                    panel(ax, y, p, bin_width=width, bandwidth=bandwidth)
                    ax.set_title(
                        f'{r["model"]} - {title}   |   AUC {m[f"auc_{condition}"]:.{auc_decimals}f}', loc='left', fontsize=10
                    )
                    ax.text(
                        0.5,
                        0.97,
                        f'Real {m["n_real"]:,} / Fake {m["n_fake"]:,}   |   {brier_label} {m[f"{brier_metric}_{condition}"]:.4f}',
                        transform=ax.transAxes,
                        ha='center',
                        va='top',
                        fontsize=8,
                    )
            for axs in (axes[i], sa):
                high = max(ax.get_ylim()[1] for ax in axs)
                for ax in axs:
                    ax.set_ylim(0, high * 1.06)
            legend(single, width, bandwidth, task)
            single.tight_layout(rect=(0, 0.04, 1, 0.86))
            save(single, output / dataset / r['model'])
        legend(fig, width, bandwidth, task)
        fig.tight_layout(rect=(0, 0.025, 1, 0.95), h_pad=2)
        save(fig, output / dataset / 'all_models')
        print(f'Probability plots: {dataset}', flush=True)


def plot_changes(data, output, width=0.01, bandwidth=0.02):
    task = data.get('task', 'RAFV')
    modality = data.get('condition', 'mask_audio').removeprefix('mask_')
    for dataset in sorted({r['dataset'] for r in data['records']}, key=lambda x: (x != 'Mega-MMDF', x)):
        records = [r for model in data['models'] for r in data['records'] if r['model'] == model and r['dataset'] == dataset]
        fig, axes = plt.subplots(4, 2, figsize=(13, 13.5))
        fig.suptitle(f'{dataset} | Paired probability change | RARV vs {task}', y=0.995, fontsize=15)
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, r in zip(axes.flat, records):
            ax.set_visible(True)
            y, full, mask = r['arrays']
            dp = mask - full
            single, sa = plt.subplots(figsize=(8, 4.8))
            single.suptitle(f'{r["model"]} | {dataset} | Paired probability change', y=0.995, fontsize=13)
            for target in (ax, sa):
                panel(target, y, dp, delta=True, bin_width=width, bandwidth=bandwidth, masked_modality=modality)
                target.set_title(f'{r["model"]}   |   Real {sum(y == 0):,} / Fake {sum(y == 1):,}', loc='left', fontsize=10)
                target.text(
                    0.5,
                    0.97,
                    f'Mean change: real {dp[y == 0].mean():+.4f} / fake {dp[y == 1].mean():+.4f}',
                    transform=target.transAxes,
                    ha='center',
                    va='top',
                    fontsize=8,
                )
                target.set_ylim(0, target.get_ylim()[1] * 1.08)
            legend(single, width, bandwidth, task)
            single.tight_layout(rect=(0, 0.04, 1, 0.86))
            save(single, output / dataset / r['model'])
        legend(fig, width, bandwidth, task)
        fig.tight_layout(rect=(0, 0.025, 1, 0.945), h_pad=2)
        save(fig, output / dataset / 'all_models')
        print(f'Probability-change plots: {dataset}', flush=True)
