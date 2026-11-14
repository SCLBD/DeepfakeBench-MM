"""Recompute pooled and stratified in-domain missing-modality analysis."""

import argparse
import csv
import gzip
import hashlib
import json
import numpy as np

from pathlib import Path
from scipy.stats import wasserstein_distance
from sklearn.metrics import brier_score_loss, roc_auc_score

from core import metrics, plot_changes, plot_probabilities

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/missing_modality_ID_assets'
TASKS = [('mask_audio', 'RAFV', 'video-only'), ('mask_video', 'FARV', 'audio-only')]


def table(headers, rows):
    return (
        '\n| '
        + ' | '.join(headers)
        + ' |\n| '
        + ' | '.join(['---'] * len(headers))
        + ' |\n'
        + '\n'.join('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)
        + '\n'
    )


def triplet(r, metric, condition):
    a, b = r[metric + '_full'], r[metric + '_' + condition]
    d = b - a
    delta = '0.000' if abs(d) < 0.0005 else f'{d:+.3f}'
    return f'{a:.3f} → {b:.3f} ({delta})'


def link(p):
    return '<' + p.as_posix() + '>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--mask-manifest', type=Path, help='Optional reference from collect_mask_manifest.py.')
    parser.add_argument('--rafv-metrics', type=Path, help='Optional metrics.json from the RAFV analysis.')
    parser.add_argument('--farv-metrics', type=Path, help='Optional metrics.json from the FARV analysis.')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.input, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    models = data['models']
    rows, plots, counts = [], {}, []
    model_trainings = {(record['model'], record['training']) for record in data['records']}
    if len(model_trainings) != len(data['records']):
        parser.error('The cache contains duplicate model/training records.')
    missing_models = [model for model in models if (model, 'Vanilla') not in model_trainings]
    if missing_models or not models:
        parser.error(f'The cache must contain Vanilla scores for every listed model. Missing: {missing_models}')

    pair_refs = {
        record['model']: [(pair['video_id'], pair['group'], pair['windows']) for pair in record['pairs']]
        for record in data['records']
        if record['training'] == 'Vanilla'
    }
    for record in data['records']:
        model, training = record['model'], record['training']
        pairs = record['pairs']
        identity = [(p['video_id'], p['group'], p['windows']) for p in pairs]
        if model not in pair_refs or identity != pair_refs[model]:
            raise ValueError(f'Cross-checkpoint video mismatch for {model}/{training}.')
        if training == 'Vanilla':
            counts.append([model] + [sum(p['group'] == g for p in pairs) for g in ['RARV', 'RAFV', 'FARV', 'FAFV']])
        for condition, single, available in TASKS:
            for subset, groups in [('pooled', [single, 'FAFV']), ('single', [single]), ('both', ['FAFV'])]:
                selected = [p for p in record['pairs'] if p['group'] == 'RARV' or p['group'] in groups]
                y = np.array([int(p['group'] != 'RARV') for p in selected])
                if set(y) != {0, 1}:
                    raise ValueError(f'{model}/{training} requires real and fake videos for {condition}/{subset}.')
                full = np.array([p['full'] for p in selected])
                mask = np.array([p[condition] for p in selected])
                r = {'model': model, 'dataset': 'Mega-MMDF', 'condition': condition, 'arrays': (y, full, mask)}
                m = metrics(r)
                for mode, pred in [('full', full), (condition, mask)]:
                    m['wasserstein_' + mode] = float(wasserstein_distance(pred[y == 0], pred[y == 1]))
                    assert abs(m['gap_' + mode]) <= m['wasserstein_' + mode] + 1e-12 <= 1 + 1e-10
                m['delta_wasserstein'] = m['wasserstein_' + condition] - m['wasserstein_full']
                m.update(training=training, condition=condition, subset=subset, fake_groups='+'.join(groups))
                for mode, pred in [('full', full), (condition, mask)]:
                    assert abs(m['auc_' + mode] - roc_auc_score(y, pred)) < 1e-12
                    assert abs(m['brier_' + mode] - brier_score_loss(y, pred)) < 1e-12
                rows.append(m)
                if training == 'Vanilla' and subset == 'pooled':
                    plots.setdefault(
                        condition, {'models': models, 'task': single + ' + FAFV', 'condition': condition, 'records': []}
                    )['records'].append(r)
    lookup = {(r['model'], r['training'], r['condition'], r['subset']): r for r in rows}
    # A common real set makes pooled AUC exactly a fake-count-weighted mean.
    for model, training in sorted(model_trainings):
        for condition, _, _ in TASKS:
            a, b, c = [lookup[model, training, condition, s] for s in ['single', 'both', 'pooled']]
            w = a['n_fake'] / (a['n_fake'] + b['n_fake'])
            assert c['n_fake'] == a['n_fake'] + b['n_fake']
            for key in ['auc_full', 'auc_' + condition, 'delta_mean_fake', 'delta_brier_balanced']:
                assert abs(c[key] - (w * a[key] + (1 - w) * b[key])) < 1e-12

    reference_paths = {'RAFV': args.rafv_metrics, 'FARV': args.farv_metrics, 'manifest': args.mask_manifest}
    # Optionally reproduce metrics from independent single-fake and FAFV exports.
    for condition, single, _ in TASKS:
        if reference_paths[single] is None:
            continue
        old = json.loads(reference_paths[single].read_text(encoding='utf-8'))['by_dataset']
        for r in old:
            if r['dataset'] == 'Mega-MMDF' and r['model'] in models:
                new = lookup[r['model'], 'Vanilla', condition, 'single']
                for key, value in r.items():
                    if isinstance(value, (float, int)):
                        assert abs(new[key] - value) < 1e-12, (r['model'], key)
    if args.mask_manifest is not None:
        manifest = json.loads(args.mask_manifest.read_text(encoding='utf-8'))
        for ref in manifest['records']:
            if ref['dataset'] != 'Mega-MMDF' or ref['model'] not in models:
                continue
            for condition, _, _ in TASKS:
                new = lookup[ref['model'], 'Vanilla', condition, 'both']
                for mode in ['full', condition]:
                    old = ref['conditions'][mode]['tasks']['FAFV']
                    assert (new['n_real'], new['n_fake']) == (old['n_real'], old['n_fake'])
                    assert abs(new['auc_' + mode] - old['auc']) < 1e-12
    (OUT / 'metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    with (OUT / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'provenance.json').write_text(
        json.dumps(
            {
                'input': str(args.input.resolve()),
                'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'sources': [{k: v for k, v in r.items() if k != 'pairs'} for r in data['records']],
                'validation': 'Matching video identities, labels and window counts across checkpoints; '
                'sklearn AUC/Brier; weighted-mixture identity',
                'references': {kind: str(path.resolve()) for kind, path in reference_paths.items() if path is not None},
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    if args.plots:
        for condition, d in plots.items():
            plot_probabilities(d, OUT / condition / 'probability', balanced_brier=True, auc_decimals=4)
            plot_changes(d, OUT / condition / 'change')
    text = [
        r'''# Missing Modality：域内合并与分组评估

评估范围为域内 Mega-MMDF。正常训练表展示 AUC、类平衡 Brier 和 W₁，统一使用原尺度、三位小数，格式为 full → mask (Δ)。有增强训练结果时，增强训练表采用增强前 → 增强后（Δ）。原始未舍入数值另存。

## 1. 为什么先研究域内

我们研究：**在训练数据集对应的测试分布下，当用户仅提供音频或视频，且可观察模态中仍含伪造时，已训练多模态检测器的排序能力、概率预测质量与稳定性如何变化？**

选择域内测试是为了减少跨数据集的伪造方法、生成／采集 pipeline、压缩等分布变化对解释的干扰，先建立输入模态可用性变化的受控基线。完整与遮蔽输入使用同一批视频及 checkpoint，唯一输入处理差异是将指定模态置零。**这不等于证明完全排除了方法或 pipeline 泛化因素**：除非另行核验训练／测试覆盖，否则应表述为“避免额外引入跨数据集迁移”。跨域泛化是后续独立扩展，本版不据域外现象概括结论。

更直接的控制来自视频内配对：同一视频的来源与原始伪造 pipeline 固定，前后差异并非换了一批新方法的数据造成。但该差异仍包含证据移除、零填充输入分布变化及融合交互改变，不能进一步单独归因于其中一个机制。

实验可作为多模态检测器用于单模态深伪检测的初步适应性检验，包括上传文件时移除某一路内容的场景。零填充仍保留原模型结构，不等同于已测过原生单模态数据集；自然静音或静止画面也不等于该操作。

### 零填充的应用动机与实现层级

不能写成“没有音轨或视频轨的文件，解码通常得到全零”。缺失轨道意味着没有该流的样本；播放器呈现无声或空白，不证明解码器返回了全零数组。FFmpeg 对不存在的流可以忽略可选映射或报错，而静音／纯色画面可以通过专用源显式生成，参见 [流映射文档](https://www.ffmpeg.org/ffmpeg.html#Advanced-options) 与 [静音及纯色源文档](https://ffmpeg.org/ffmpeg-filters.html)。

**可采用的逻辑是：** 对需要同时接收两路输入的检测器，当媒体只提供一路时，预处理端可以用无内容的占位信号补齐另一输入。本 benchmark 采用固定长度的零音频波形／黑色视频帧作为一种明确、可复现的缺失模态处理基线，评价该协议下的适应性。这里是设计选择，不是声称所有播放器或解码器都这样实现。

当前本地 `datasets/utils/data_helpers/` 在 mask video 时生成零像素帧，在 mask audio 时生成零波形；`datasets/audio_video/audio_video_dataset.py` 随后仍执行视频归一化，以及按配置执行音频频谱转换和归一化。所以这是**原始信号层面的置零**，不能未经核验就称为“模型最终输入张量全部为零”。例如归一化 (x−μ)/σ 中，黑色 x=0 会变成 −μ/σ；零波形经对数频谱、MFCC 或标准化后也未必为零。这里的实现定位来自当前本地代码；历史运行的精确实现仍以对应代码版本和配置为准。

因此静音/黑屏可作为占位协议的直观描述，但自然录制的静音可能含背景噪声，黑屏也不等于任意编码格式的所有字节为零。本实验不覆盖全部自然缺失、编码或前处理方式。适合论文的说法是：**“We simulate an unavailable modality by replacing its raw signal with a fixed-shape silent waveform or black frames, followed by the detector's standard preprocessing.”** 使用此句描述历史运行前，应确认运行版本采用相同的置零阶段。

## 2. 新任务口径：合并评估

| 输入变化 | 真实组 | 伪造组（主结果） | 仍可观察的伪造 |
|---|---|---|---|
| Full → mask audio / video-only | RARV | RAFV + FAFV | fake video |
| Full → mask video / audio-only | RARV | FARV + FAFV | fake audio |

FAFV 中被移除的模态也含伪造，但剩余模态仍为 fake，故 Full 与 mask 的视频计分标签保持一致。原来的 RAFV/FARV 单组实验没有算错，只是专门研究“真实辅助模态被移除”；**新合并实验不能再统称为移除 real modality**。FAFV 子组会同时失去一路伪造线索，部分下降可能来自信息减少，不宜直接认定为模型缺陷。

每个视频先聚合全部采样窗口的 original audio/video label（OR），再按聚合标签划组；视频 fake probability 为所有窗口的最大值。不会根据路径猜类别，也不会先筛窗口。配对核验窗口指纹、视频 ID、标签、窗口数，正常训练与增强训练保持一致。不同模型采样集合可能不同。

为何不将 FARV 在 mask audio 后归入本实验 real，或 RAFV 在 mask video 后归入 real？它们在 Full 时为 fake，在 mask 后为 real，纳入同一前后对照会改变任务标签。本研究保留标签不变的子集。若未来评价“全部单模态输入的实际真假”，可以把这些转为 real 的样本纳入另一个仅评估缺失输入的任务，不能混用这里的配对解释。

**正文仅讨论合并任务；子组统计保留在底层数据，不再展开讨论。** 同一 RARV 真实组下，合并 AUC 等于两个子组 AUC 按 fake 视频数加权；并非两组 AUC 简单平均。合并只计一次 RARV，不能把两个二分类数据表直接拼接导致重复 real。

$$\mathrm{AUC}_{pool}=w\mathrm{AUC}_{single}+(1-w)\mathrm{AUC}_{FAFV},\qquad w=\frac{N_{single}}{N_{single}+N_{FAFV}}.$$

fake 概率均值／Brier 同样按 fake 数量加权。因此主表体现当前测试集的组别比例；附录同时给两子组等权结果，用于检查对该比例的敏感性。该等权结果是组别标准化汇总，不是新的实际样本集合。

## 3. 指标

统一 Δ=mask−full。AUC 是排序判别能力；Brier 是概率预测误差，越低越好，不等同于 confidence、纯校准或两类均值 gap。

| 指标 | 建议表述 | 前后变化的含义 |
|---|---|---|
| AUC ↑ | 排序判别能力 | 下降表示 fake 相对 real 的排序能力减弱；不代表所有性能维度 |
| 类平衡 Brier ↓ | 概率预测质量（均方概率误差） | 上升表示概率离正确标签的平均平方距离增大；两类等权 |
| Wasserstein-1（W₁） | 类间预测概率分布距离 | 下降表示两类分布在概率轴上更接近；本身没有正确分类方向 |

“预测的置信度是否能够合理运用”不是 Brier 的准确解释：它评价输出概率与实际标签的匹配质量，错误而极端的概率会受到较大惩罚，但不能分离置信度、校准和判别能力。因此三个视角宜表述为 **排序判别能力、概率预测质量、类间分布距离**。本版取消正常训练表中的 real/fake 概率均值及均值变化列；分布形态继续由直方图辅助解释，原始统计仍保留在数据附件。

$$B_R=\frac{1}{N_R}\sum_{y_i=0}p_i^2,\quad B_F=\frac{1}{N_F}\sum_{y_i=1}(1-p_i)^2,\quad B_{bal}=\frac{B_R+B_F}{2}.$$

类别平衡只让 real/fake 各占一半，**不会自动让 RAFV/FARV 与 FAFV 两个 fake 子组各占一半**。视频 max 分数虽在 [0,1] 内，不假定是已校准事件概率。real 的正 mean(Δp) 或 fake 的负 mean(Δp) 表示向错误方向移动，但均值变化不能确定任意阈值下的错误率。额外保留 mean(|Δp|) 以防正负抵消；稳定不等于准确。

小数位仅是展示精度，小数值是否可忽略需要实际效应量或区间依据，不能用舍入决定改善／无变化。当前没有多种子或显著性检验。

### 新增：real 与 fake 概率分布的 Wasserstein-1 距离

采用成熟的一维 **Wasserstein-1 distance（W₁，也称 Earth Mover's Distance）**。分别对同一输入条件下 real/fake 的视频分数构造经验分布，不是计算同一视频 mask 前后的距离。它可理解为把一类概率分布搬成另一类所需的最小平均搬运距离：

$$W_1(P_R,P_F)=\int_0^1|F_R(t)-F_F(t)|\,dt.$$

这里 F 是累积分布函数，两类分布各自归一化为质量 1，类内视频等权；允许 real/fake 数量不同。W₁ 位于 [0,1]，与概率横轴同尺度，不依赖直方图 bin 或 KDE。合并 fake 仍按实际视频数构成，W₁ 直接从合并原始分数计算，不能简单加权两个子组的 W₁。

主表展示 **W₁ full → mask (Δ)**；减小表示两个类别的概率分布在该距离意义下更接近。它是分布距离，不是有方向的分类性能指标：标签预测完全反向也可能距离很大，且不单独衡量校准或阈值错误率，必须结合 AUC、类别均值与 Brier。

例如 real/fake 都集中在 0.1/0.9 时 W₁=0.8，压缩到 0.4/0.6 时 W₁=0.2，但两个例子的 AUC 都是 1。相反，两类一起平移且不截断时 W₁ 不变，Brier 仍可能变化。W₁ 因而补充概率轴上的分布距离，不能替代现有指标。

有符号均值间距 G=mean(p|fake)−mean(p|real) 保留在完整数据的 gap_* 字段中，满足 W₁≥|G|；当 fake 分数一阶随机占优于 real 时二者相等。报告不将 W₁ 当成独立于均值间距的全新机制证据，也不进行 Wasserstein 显著性检验。

定义与经验分布实现见 [SciPy 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)。相关论文为 Ramdas, Garcia & Cuturi (2015), [On Wasserstein Two Sample Testing and Related Families of Nonparametric Tests](https://arxiv.org/abs/1509.02237)，讨论 Wasserstein 与非参数两样本检验的联系；这里采用距离本身，不采用其检验流程。

正常训练表统一保留三位小数。Δ=mask−full，先用未舍入值相减再舍入；舍入到零统一显示 0.000，不表示严格不变。AUC 变化 +0.001 表示约 +0.1 个百分点，并不自动等于没有变化。方向计数仍按未舍入值计算。

## 4. 正常训练：合并主结果
'''
    ]

    def get(model, condition, subset='pooled', training='Vanilla'):
        return lookup[model, training, condition, subset]

    for condition, single, available in TASKS:
        text.append(f'### RARV vs ({single} + FAFV)：{available}\n')
        rr = [get(model, condition) for model in models]
        text.append(
            table(
                ['模型', 'AUC ↑', '类平衡 Brier ↓', 'W₁（分布距离）'],
                [[r['model']] + [triplet(r, k, condition) for k in ['auc', 'brier_balanced', 'wasserstein']] for r in rr],
            )
        )
        text.append(
            f"AUC 下降 {sum(r['delta_auc']<0 for r in rr)}/{len(rr)}；类平衡 Brier 变差 {sum(r['delta_brier_balanced']>0 for r in rr)}/{len(rr)}。\n"
        )
        text.append(
            f"W₁ 在 {sum(r['delta_wasserstein']<0 for r in rr)}/{len(rr)} 模型上减小，说明这两类概率分布在概率轴上的距离收缩；仍须结合 AUC 判断排序方向。\n"
        )
        if args.plots:
            text.append('图中蓝色为 RARV，红色为合并 fake。直方图 bin=0.01，反射 Gaussian binned KDE 带宽 0.02。')
            for kind in ['probability', 'change']:
                stem = OUT / condition / kind / 'Mega-MMDF/all_models'
                text.append(f'![{single}+FAFV {kind}]({stem.as_posix()}.png)\n\n[矢量 PDF]({link(stem.with_suffix(".pdf"))})\n')

    if any(training != 'Vanilla' for _, training in model_trainings):
        text.append(
            '## 5. Random mask modality：增强训练对照\n\n'
            'RMA 为 random mask audio 训练，RMV 为 random mask video 训练。'
            '每格为 **增强前 → 增强后（Δ）**。测试视频和输入模式固定，仅比较不同训练 checkpoint。'
            'Brier 均为类平衡版本。'
        )

    def training_cell(a, b, key):
        d = b[key] - a[key]
        delta = '0.000' if abs(d) < 0.0005 else f'{d:+.3f}'
        return f'{a[key]:.3f} → {b[key]:.3f} ({delta})'

    for condition, single, available in TASKS:
        train = 'RMA' if condition == 'mask_audio' else 'RMV'
        selected_models = [model for model in models if (model, train) in model_trainings]
        if not selected_models:
            continue
        text.append(f'### RARV vs ({single}+FAFV)：{train}\n')
        pairs = [(get(model, condition), get(model, condition, training=train)) for model in selected_models]
        keys = [k + '_' + mode for mode in ['full', condition] for k in ['auc', 'brier_balanced', 'wasserstein']]
        text.append(
            table(
                ['模型', 'Original AUC ↑', 'Original Brier ↓', 'Original W₁', 'Mask AUC ↑', 'Mask Brier ↓', 'Mask W₁'],
                [[a['model']] + [training_cell(a, b, k) for k in keys] for a, b in pairs],
            )
        )
        n_auc = sum(b['auc_' + condition] > a['auc_' + condition] for a, b in pairs)
        n_b = sum(b['brier_balanced_' + condition] < a['brier_balanced_' + condition] for a, b in pairs)
        text.append(
            f'**缺失输入：** AUC 在 {n_auc}/{len(pairs)} 模型上提高，Brier 在 {n_b}/{len(pairs)} 模型上降低。W₁ 只辅助描述类间分布距离，不作为单独的性能提升判据。\n'
        )
        for key, label, sign in [
            ('auc_full', 'Original AUC 下降', -1),
            ('brier_balanced_full', 'Original Brier 上升', 1),
            ('brier_balanced_' + condition, 'Mask Brier 上升', 1),
        ]:
            names = [a['model'] for a, b in pairs if (b[key] - a[key]) * sign > 0]
            text.append(
                label
                + '的模型：'
                + ('、'.join(names) if names else '无')
                + '。计数按未舍入结果，不表示小变化具有统计或实际显著性。\n'
            )
    text.append('## 附录 A：组别数量与等权敏感性\n' + table(['模型', 'RARV', 'RAFV', 'FARV', 'FAFV'], counts))
    for condition, _, _ in TASKS:
        entries = []
        for model in models:
            a, b, c = [get(model, condition, s) for s in ['single', 'both', 'pooled']]
            entries.append(
                [
                    model,
                    f"{a['n_fake']/c['n_fake']:.3f}",
                    f"{c['delta_auc']:+.3f}",
                    f"{(a['delta_auc']+b['delta_auc'])/2:+.3f}",
                    f"{c['delta_brier_balanced']:+.3f}",
                    f"{(a['delta_brier_balanced']+b['delta_brier_balanced'])/2:+.3f}",
                ]
            )
        text.append(
            '### '
            + condition
            + '\n'
            + table(['模型', 'single fake 占比', '合并 ΔAUC', '两组等权 ΔAUC', '合并 ΔB bal', '两组等权 ΔB bal'], entries)
        )
    text.append(
        f'## 附录 B：完整指标与来源\n\n共读取 {len(data["records"])} 个模型和训练设置组合，'
        f'生成 {len(rows)} 行指标（合并／single／FAFV）。'
        '核对了 sklearn AUC/Brier 和合并加权恒等式。\n'
    )
    for title, path in [
        ('全部指标 CSV', OUT / 'metrics.csv'),
        ('全部指标 JSON（未舍入）', OUT / 'metrics.json'),
        ('逐文件来源与配对指纹', OUT / 'provenance.json'),
        ('域内预测缓存', args.input.resolve()),
    ]:
        text.append(f'- [{title}]({link(path)})\n')
    (ROOT / 'outputs/missing_modality_report.md').write_text('\n\n'.join(text), encoding='utf-8')
    print(f'Validated {len(rows)} rows; report written', flush=True)


if __name__ == '__main__':
    main()
