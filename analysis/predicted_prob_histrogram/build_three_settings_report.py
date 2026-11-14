"""Collect existing validated in-domain metrics into three explicit settings."""

import csv
import hashlib
import json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/missing_modality_ID_assets/metrics.json'
OUT = ROOT / 'outputs/three_settings_assets'
MODELS = ['AVFF', 'AVH', 'Baseline', 'FGMDF', 'FRADE', 'FauForensics', 'MCL']
SETTINGS = [('RAFV', 'single', ['mask_audio']), ('FARV', 'single', ['mask_video']), ('FAFV', 'both', ['mask_audio', 'mask_video'])]


def table(headers, rows):
    escape = lambda value: str(value).replace('|', '\\|')
    return (
        '\n| '
        + ' | '.join(map(escape, headers))
        + ' |\n| '
        + ' | '.join(['---'] * len(headers))
        + ' |\n'
        + '\n'.join('| ' + ' | '.join(map(escape, r)) + ' |' for r in rows)
        + '\n'
    )


def delta(x):
    return '0.000' if abs(x) < 0.0005 else f'{x:+.3f}'


def change(a, b):
    return f'{a:.3f} → {b:.3f} ({delta(b-a)})'


def main():
    rows = json.loads(SOURCE.read_text(encoding='utf-8-sig'))
    lookup = {(r['model'], r['training'], r['subset'], r['condition']): r for r in rows}
    assert len(lookup) == len(rows) == 126
    chosen = []
    text = [
        '# 三个设定：单模态输入适用性与双伪造遮蔽响应',
        '日期：2026-09-23。仅 Mega-MMDF 域内；七个模型；所有指标为 video-level。重新组织已有完整结果，未重新运行推理。前两块不合并 FAFV，与此前 pooled missing-modality 报告区分。',
        '## 实验逻辑',
        table(
            ['设定', '完整输入与何种输入比较', '问题'],
            [
                [
                    'RARV vs RAFV',
                    'mask audio（video-only，zero-filling）',
                    '只有视频伪造时，现有多模态模型对零填充单模态输入的适用性',
                ],
                [
                    'RARV vs FARV',
                    'mask video（audio-only，zero-filling）',
                    '只有音频伪造时，现有多模态模型对零填充单模态输入的适用性',
                ],
                ['RARV vs FAFV', '分别 mask audio / mask video', '双伪造条件下，对两路遮蔽的输出与性能响应是否不对称'],
            ],
        ),
        '这些比较均保留 fake 模态，因此 real/fake 标签不随指定遮蔽改变。此处 zero-filling 是现有实现的遮蔽协议，不等同于所有真实缺失媒体或独立单模态数据集。',
        '原始 video 标签由全部保存窗口的 original audio/video label 分别 OR 聚合，再选择类别；video 分数取窗口最大值。FAFV 不保证每个窗口都同时双伪造。每模型内不同输入和训练方式已配对；不同模型样本数可能不同。',
        'AUC ↑ 衡量排序；类平衡 Brier ↓ 为 real/fake 类内均方误差各占一半，并非纯校准指标；W₁ 是两类分数分布的距离，不单独代表检测质量。平均 Δp 为 mask−full，平均 |Δp| 避免正负抵消，不代表错误率。全部显示三位小数，零可能是舍入结果。',
        'RMA = random mask audio training；RMV = random mask video training。增强表分别比较相同输入下 Vanilla→增强训练，不能与同 checkpoint 的 full→mask 混淆。',
    ]
    for idx, (fake, subset, conditions) in enumerate(SETTINGS, 1):
        text.append(f'## {idx}. RARV vs {fake}')
        counts = []
        for model in MODELS:
            ref = lookup[model, 'Vanilla', subset, conditions[0]]
            counts.append([model, ref['n_real'], ref['n_fake']])
            for training in ['Vanilla', 'RMA', 'RMV']:
                for condition in conditions:
                    r = lookup[model, training, subset, condition]
                    assert (r['n_real'], r['n_fake']) == (ref['n_real'], ref['n_fake'])
                    assert r['fake_groups'] == fake
                    chosen.append(dict(r, setting='RARV vs ' + fake))
        text.append(table(['模型', 'real 视频数', 'fake 视频数'], counts))
        for condition in conditions:
            text.append(f'### {condition}：正常训练，Full → Mask（Δ）')
            normal = [lookup[m, 'Vanilla', subset, condition] for m in MODELS]
            text.append(
                table(
                    ['模型', 'AUC ↑', '类平衡 Brier ↓', 'W₁'],
                    [
                        [r['model']]
                        + [change(r[k + '_full'], r[k + '_' + condition]) for k in ['auc', 'brier_balanced', 'wasserstein']]
                        for r in normal
                    ],
                )
            )
            text.append(
                table(
                    ['模型', 'real 平均 Δp', 'fake 平均 Δp', 'real 平均 |Δp|', 'fake 平均 |Δp|'],
                    [
                        [
                            r['model'],
                            delta(r['delta_mean_real']),
                            delta(r['delta_mean_fake']),
                            f"{r['delta_p_real_mae']:.3f}",
                            f"{r['delta_p_fake_mae']:.3f}",
                        ]
                        for r in normal
                    ],
                )
            )
            for training in ['RMA', 'RMV']:
                text.append(f'### {condition}：Vanilla → {training}（Δ）')
                for mode in ['full', condition]:
                    text.append(f'输入：{mode}。这里的 Δ 是增强训练−正常训练。')
                    entries = []
                    for model in MODELS:
                        a, b = [lookup[model, t, subset, condition] for t in ['Vanilla', training]]
                        entries.append(
                            [model]
                            + [change(a[k + '_' + mode], b[k + '_' + mode]) for k in ['auc', 'brier_balanced', 'wasserstein']]
                        )
                    text.append(table(['模型', 'AUC ↑', '类平衡 Brier ↓', 'W₁'], entries))
    text += [
        '## 阅读边界与下一步',
        '前两块先描述 zero-filling 对单模态输入检测的影响，再检验简单训练增强是否改善遮蔽输入、是否损害完整输入；不预设所有模型都下降或增强必然有效。第三块将 FAFV 的输出变化与 RARV vs FAFV 的 AUC 变化分开解读，后者也受 real 侧响应影响。',
        '第三块的 Vanilla 数据中，七个模型 FAFV 平均 |Δp| 均为 mask video 更大；但 AVFF、MCL 的 AUC 为 mask audio 损害更大，FGMDF 两种 AUC 损害接近。不能由此直接归因为某一路 artifacts 的贡献，也不称为 shortcut。',
        '当前不以 CKA 解释决策依赖。未预设不敏感阈值，未做统计显著性或等效性检验；接近 1 的分数饱和与窗口 max 聚合可能隐藏变化。',
        '## 来源与复现',
        '- 原始汇总：`outputs/missing_modality_ID_assets/metrics.json`；来源清单：同目录 `provenance.json`。该清单记录 63 个预测文件及既有配对验证。',
        '- 正常输入远端：`/mnt/aoss/DeepfakeBench-MM/logs/test/{model}_train_on_Mega-MMDF/predictions/Mega-MMDF.json`。',
        '- 正常训练遮蔽输入：`/mnt/aoss/DeepfakeBench-MM/logs/mask_modality_test_bkp/{model}_train_on_Mega-MMDF_mask_{audio|video}/predictions/Mega-MMDF.json`。',
        '- 增强训练：`logs/random_mask_modality_test/{model}_train_on_Mega-MMDF_random_mask_{audio|video}_{original|mask_audio|mask_video}/predictions/Mega-MMDF.json`。',
        '- 本次精确结果：`outputs/three_settings_assets/metrics.json`、`metrics.csv`（84 条比较记录）；未舍入值及概率变化分位数保留。',
        '- 生成脚本：`analysis/predicted_prob_histrogram/build_three_settings_report.py`。',
    ]
    assert len(chosen) == 84
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'metrics.json').write_text(json.dumps(chosen, indent=2), encoding='utf-8')
    fields = list(dict.fromkeys(k for r in chosen for k in r))
    with (OUT / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(chosen)
    provenance = json.loads((SOURCE.parent / 'provenance.json').read_text(encoding='utf-8-sig'))
    provenance['collection_source_sha256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    provenance['collection_scope'] = '84 in-domain records; single RAFV, single FARV, both FAFV; Vanilla/RMA/RMV'
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    report = ROOT / 'outputs/three_settings_mask_modality_report.md'
    report.write_text('\n\n'.join(text) + '\n', encoding='utf-8')
    print(f'Validated {len(chosen)} records. Wrote {report}')


if __name__ == '__main__':
    main()
