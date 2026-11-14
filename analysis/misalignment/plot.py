"""Plot paired sample-level probability changes; exclude unchanged windows."""

import collections
import gzip
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from collect import MODELS

ROOT = Path(__file__).parent / 'outputs'
ROOT.mkdir(exist_ok=True)
data = {m: [] for m in MODELS}
control = {m: [] for m in MODELS}
with gzip.open(ROOT / 'pairs.jsonl.gz', 'rt') as f:
    for line in f:
        r = json.loads(line)
        (data if r['applied'] else control)[r['model']].append(r)

stats = {}
for m in MODELS:
    rows = data[m]
    a = np.array([r['aligned'] for r in rows])
    b = np.array([r['misaligned'] for r in rows])
    delta = b - a
    c = np.array([r['misaligned'] - r['aligned'] for r in control[m]])
    clips = collections.defaultdict(list)
    for r, d in zip(rows, delta):
        clips[r['path']].append(float(d))
    stats[m] = dict(
        n=len(rows),
        clips=len(clips),
        aligned_mean=float(a.mean()),
        misaligned_mean=float(b.mean()),
        mean_delta=float(delta.mean()),
        mean_abs_delta=float(abs(delta).mean()),
        median=float(np.median(delta)),
        quantiles=np.quantile(delta, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]).tolist(),
        clip_macro_delta=float(np.mean([np.mean(v) for v in clips.values()])),
        unchanged_n=len(c),
        unchanged_mean_abs=float(abs(c).mean()),
        unchanged_max_abs=float(abs(c).max()),
        unchanged_over_1e_4=int((abs(c) > 1e-4).sum()),
    )

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
for zoom in [False, True]:
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    for ax, m in zip(axes.flat, MODELS):
        delta = np.array([r['misaligned'] - r['aligned'] for r in data[m]])
        lim = max(0.001, float(np.quantile(abs(delta), 0.99))) if zoom else 1.0
        lim = min(1.0, lim)
        bins = np.linspace(-lim, lim, 101 if zoom else 201)
        ax.hist(delta, bins=bins, weights=np.full(len(delta), 100 / len(delta)), color='#3978a8', alpha=0.85)
        ax.axvline(0, color='#333333', lw=0.8)
        ax.axvline(delta.mean(), color='#d37534', lw=1.3, ls='--')
        ax.set_xlim(-lim, lim)
        ax.set_title(
            f'{m}{"*" if m == "AVFF" else ""} | n={len(delta):,} | mean={delta.mean():+.4f} | mean abs={abs(delta).mean():.4f}',
            fontsize=10,
        )
        ax.set_xlabel('Fake probability change (misaligned - aligned)')
        ax.set_ylabel('Samples per bin (%)')
        if zoom:
            ax.text(0.02, 0.9, f'Outside view: {100*np.mean(abs(delta)>lim):.2f}%', transform=ax.transAxes, fontsize=9)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(
        0,
        0.8,
        'Original RARV only; actual audio rotations only.\nPaired by path + original sample frame range.\nDashed orange: mean change.\n* AVFF: unchanged controls also differ (MAE 0.00456).\n'
        + (
            'Per-model zoom; bin widths differ.\nBar heights use ALL paired samples as denominator.'
            if zoom
            else 'Common range [-1, 1]; bin width 0.01.\nEach model retains its native window duration.'
        ),
        va='top',
        linespacing=1.8,
    )
    fig.suptitle('Response to within-clip audio-window misalignment' + (' (zoom)' if zoom else ''), fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    name = 'prob_change_zoom' if zoom else 'prob_change'
    fig.savefig(ROOT / (name + '.png'), dpi=170)
    fig.savefig(ROOT / (name + '.pdf'))
    plt.close(fig)
ROOT.joinpath('summary.json').write_text(json.dumps(stats, indent=2))
lines = [
    '# 同视频音频窗口错位：配对概率变化',
    '',
    '仅统计实际错位的原始 RARV 窗口。Δp = 错位后 − 正常输入。sample-level 汇总，不做 video max。',
    '',
    '| 模型 | 窗口数 | 片段数 | 正常均值 | 错位均值 | 平均 Δp | 平均绝对 Δp | 中位数 Δp | 未错位对照平均绝对差 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for m, s in stats.items():
    lines.append(
        f"| {m} | {s['n']:,} | {s['clips']:,} | {s['aligned_mean']:.4f} | {s['misaligned_mean']:.4f} | {s['mean_delta']:+.4f} | {s['mean_abs_delta']:.4f} | {s['median']:+.4f} | {s['unchanged_mean_abs']:.6g} |"
    )
lines += [
    '',
    '## 直方图',
    '',
    '共同横轴范围；bin 宽度 0.01。',
    '',
    '![完整范围](prob_change.png)',
    '',
    '各模型局部放大：范围覆盖至少约 99% 的绝对变化；图上标明范围外比例，柱高仍以全部配对窗口为分母。各图横轴、bin 宽度不同。',
    '',
    '![局部放大](prob_change_zoom.png)',
    '',
    '## 解释边界',
    '',
    '窗口长度、有效片段集合因模型而异；不能据此把跨模型响应差异全部归因于架构。错位响应并非同步性决策贡献百分比。未错位对照用于核查历史正常预测的一致性；明显不一致时结果只能作为待复核观察。',
    '',
    '配对审计与原文件 SHA256：pairs.jsonl.audit.json；逐窗口配对数据：pairs.jsonl.gz；完整统计（含片段宏平均和分位数）：summary.json。',
]
lines += [
    '',
    '## 本轮观察与审计',
    '',
    '七个模型均无未匹配窗口、无重复配对键。日志配置中模型参数、权重路径和基础 data_config 一致，后者仅新增 misalignment；JSON 根目录有变更，但本轮全部目标窗口均匹配。历史日志未记录权重文件哈希，不能据路径相同证明历史权重字节一致。',
    '',
    'AVH 平均 Δp 为 +0.0243，片段宏平均为 +0.0219；其余模型平均变化绝对值均低于 0.0005。平均绝对变化并非为零，说明个体窗口发生正负响应。',
    '',
    'AVFF 的未错位对照平均绝对差为 0.004564、最大差为 0.102609；原因尚未确定，应补相同运行条件下 shift=0 对照后再解释其错位响应。其他模型的未错位对照完全一致或仅有很小数值差（FGMDF 最大约 0.0000176）。',
    '',
    '平均绝对变化不能独立证明模型利用同步性：循环置换改变了每个 anchor 的音频内容，即便模型只依赖音频也可能产生个体分数变化，同时整体均值不变。这里报告的是配对输出响应；AVH 的系统性上移支持本次配对破坏影响最终判断，但不是细粒度同步机制的直接证明。',
]
ROOT.joinpath('report.md').write_text('\n'.join(lines), encoding='utf-8')
ROOT.joinpath('index.html').write_text(
    '<!doctype html><meta charset="utf-8"><title>Misalignment probability changes</title><style>body{max-width:1200px;margin:30px auto;font-family:Arial}img{width:100%}</style><h1>同视频音频窗口错位：配对概率变化</h1><p>仅实际错位的 RARV 窗口；Δp = misaligned − aligned。完整范围与局部放大。</p><img src="prob_change.png"><img src="prob_change_zoom.png">',
    encoding='utf-8',
)
print(json.dumps(stats, indent=2))
