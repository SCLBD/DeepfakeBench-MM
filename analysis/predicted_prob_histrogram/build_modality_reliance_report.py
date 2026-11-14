"""Summarize existing paired FAFV predictions; no inference or CKA computation."""

import hashlib
import json

from pathlib import Path

from analyze_missing_groups import table, triplet

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/missing_modality_ID_assets/metrics.json'
OUT = ROOT / 'outputs/modality_reliance_assets'
OUT.mkdir(parents=True, exist_ok=True)
rows = [r for r in json.loads(SOURCE.read_text(encoding='utf-8')) if r['training'] == 'Vanilla' and r['subset'] == 'both']
assert len(rows) == 14 and all(r['fake_groups'] == 'FAFV' for r in rows)
models = list(dict.fromkeys(r['model'] for r in rows))
lookup = {(r['model'], r['condition']): r for r in rows}
for model in models:
    a, v = [lookup[model, c] for c in ['mask_audio', 'mask_video']]
    for key in ['n_real', 'n_fake', 'auc_full', 'brier_balanced_full', 'wasserstein_full']:
        assert a[key] == v[key], (model, key)
(OUT / 'metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
(OUT / 'provenance.json').write_text(
    json.dumps(
        {
            'source': str(SOURCE),
            'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'selection': "training=Vanilla, subset=both (RARV vs FAFV)",
            'source_manifest': str(ROOT / 'outputs/missing_modality_ID_assets/provenance.json'),
        },
        indent=2,
    ),
    encoding='utf-8',
)
parts = [
    r'''# RARV vs FAFV：模态依赖不对称分析起点

日期：2026-09-23。当前仅整理 Mega-MMDF 域内、七个正常训练 checkpoint 的既有预测。没有新运行推理，没有计算 CKA；特征缓存覆盖与提取位置尚未核验。与 missing modality 报告独立保存。

## 1. 问题与术语

研究问题：两模态均含伪造时，检测器对两种模态遮蔽的响应是否不对称？完整输入下，fusion 表征与 audio/video 表征的样本关系结构是否也有不对称？两种观察是否一致？

优先称为 **asymmetric modality reliance / 模态依赖不对称**，并具体报告“遮蔽敏感性”和“表征相似性”。两路依赖不相等，不代表不合理；伪造证据强度与任务难度可能不同。Over-reliance 需要额外说明过度依赖的损害，shortcut 需要识别具体不可靠线索及其干预证据。当前不预设存在 shortcut。

RARV 与 FAFV 在遮蔽任一路后仍分别为 real 与 fake，避免标签改变。原始视频组按全部保存窗口的原始 audio/video 标签分别 OR 后确定；预测概率沿用窗口 max。不同输入在每模型内严格配对，不同模型的样本集合可能不同。FAFV 的音频／视频伪造区间不一定在每个窗口都同时存在，视频级“双 fake”不等于每个窗口均双 fake。

## 2. 行为结果：同一任务，分别遮蔽两路

每格为 **Full → Mask（Δ）**，Δ=mask−full，三位小数；方向计数按未舍入值。Brier 为 real/fake 各占一半的类平衡 Brier。W₁ 仅衡量类别分布距离，不单独解释为性能。
'''
]
for condition in ['mask_audio', 'mask_video']:
    parts.append(f'### {condition}\n')
    subset = [lookup[m, condition] for m in models]
    parts.append(
        table(
            ['模型', 'real / fake 视频数', 'AUC ↑', 'Brier ↓', 'W₁'],
            [
                [r['model'], f"{r['n_real']} / {r['n_fake']}"]
                + [triplet(r, k, condition) for k in ['auc', 'brier_balanced', 'wasserstein']]
                for r in subset
            ],
        )
    )
parts.append(
    '### 两种遮蔽响应的对照\n\n这里只比较敏感性，不把结果直接命名为 audio/video shortcut。AUC 损失=Full−Mask；Brier 增量=Mask−Full。数值更大表示对应度量下损害更大，未作显著性检验。'
)
contrast = []
for model in models:
    a, v = [lookup[model, c] for c in ['mask_audio', 'mask_video']]
    auc_mode = 'mask audio' if a['delta_auc'] < v['delta_auc'] else 'mask video'
    b_mode = 'mask audio' if a['delta_brier_balanced'] > v['delta_brier_balanced'] else 'mask video'
    contrast.append(
        [
            model,
            f"{-a['delta_auc']:.3f} / {-v['delta_auc']:.3f}",
            f"{a['delta_brier_balanced']:.3f} / {v['delta_brier_balanced']:.3f}",
            auc_mode,
            b_mode,
        ]
    )
parts.append(table(['模型', 'AUC 损失：mask A / V', 'Brier 增量：mask A / V', 'AUC 损害更大', 'Brier 损害更大'], contrast))
parts.append(
    r'''## 3. 能支持什么，不能支持什么

遮蔽实验直接测量固定模型对该具体输入干预的总响应；它混合了证据缺失、输入分布变化及交互改变，不能将损失全部归为原始输入上的模态贡献。两边损失都小也可能来自两路证据冗余，而不是都没使用。AUC/Brier 对不对称的判断不一致时应保留分歧，不强行给每个模型贴一个依赖方向。

当前代码在原始信号阶段生成零波形／黑帧，再进行标准预处理，最终张量未必全零；历史运行的实现以其代码与配置为准。不能将零遮蔽视为唯一或天然的缺失处理方式。

CKA+t-SNE+mask 可以构成互补观察，不能仅因为三者一致就宣布机制被证明。后续更强验证可考虑保标签的匹配替换或渐进扰动，但它们也可能改变内容与跨模态对应关系，不存在自动消除混淆的简单替换。

## 4. CKA：衡量同一批样本的内部几何关系

设 X、Y 为顺序一一对应的 n 个样本的两种表征，维度 dₓ、dᵧ 可以不同；对各特征列按样本维度中心化得到 Xc、Yc。线性 CKA 为：

$$\operatorname{CKA}(X,Y)=\frac{\|X_c^\top Y_c\|_F^2}{\|X_c^\top X_c\|_F\;\|Y_c^\top Y_c\|_F}.$$

这与比较中心化样本 Gram 矩阵的归一化 Frobenius 内积等价。它不是不配对分布距离，也不是专门测量真假可分性。来源：[Kornblith et al., ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html)。

计划比较 CKA(audio,fusion) 与 CKA(video,fusion)，先用完整输入，再分别报告整体、RARV 内部和 FAFV 内部。类内结果帮助检查相似性是否主要来自共同真假分组，不能替代整体结果。t-SNE 仅作可视化，不在二维坐标上计算 CKA，也不以两张独立 t-SNE 图形相似证明依赖。

高 CKA 不等于分类贡献大：fusion 的主要变化可能来自视频身份等信息，但分类头实际读取较小的音频真假方向。交互后的 audio/video token 还可能已包含另一模态的信息，需核查提取层位置，而非只看张量名。

## 5. CKA 的内存与实现计划

**不用默认构造 n×n 矩阵。** 例如 n=100,000 时，一个 float32 Gram 矩阵就约 37.3 GiB，两个至少 74.5 GiB，尚未计中间量。这里的视频集合已超过十万量级，直接 Gram 法不适合。

线性 CKA 可以使用上面的特征空间等价式，仅保留 dₓ×dₓ、dᵧ×dᵧ、dₓ×dᵧ 三个统计矩阵。若 dₓ=1024、dᵧ=2048，三个 float64 矩阵合计 **56 MiB**，另加读取批次、矩阵计算临时空间与均值向量。无需同时将全部特征放进 RAM 或 GPU。

推荐两遍分批读取：第一遍计算全局均值；第二遍用全局均值中心化每批，累加 XcᵀXc、YcᵀYc、XcᵀYc，再统一归一化。这在数学上与全量线性 CKA 一致（浮点误差除外）。**不能把每批分别中心化所得 CKA 简单取平均，冒充全量 CKA。** 零方差表征的分母为零，应报告 undefined。

内存下降不等于计算时间很少：密集计算仍约 O(n(dₓ²+dᵧ²+dₓdᵧ))。如果展平时空 token 后维度数万，特征空间矩阵本身也会很大。先核对 pooling 和维度，再选择等价计算；必要时固定分层抽样并检查多次抽样稳定性。核 CKA（如 RBF）不直接适用该线性等价式。样本数/维度比例也会影响有限样本估计解释，不能任意比较不同样本构成的数值。

## 6. 接下来需要的数据核验

1. 七模型正常输入的 audio/video/fusion 特征是否已保存完整；当前尚未确认。
2. 每个 feature 位于交互前还是交互后，维度与时间/token pooling 是什么。
3. 特征的窗口 ID、采样区间与预测是否逐项匹配，三种表征使用同一行顺序。
4. 视频级 label 用全部窗口聚合，但 CKA 分析单位另行明确。不能把窗口 max 概率聚合直接套到 feature；也不默认取“最高分窗口”，以免选择结果影响解释。
5. 若做窗口级 CKA，明确视频如何等权、窗口真假标签如何定义；若做视频级 CKA，明确固定 pooling，不把不同视频窗口数混成隐含权重。

历史方案还包含 linear probe、Fisher ratio、silhouette。probe 目前暂不做；不同辅助 CE 使解释更复杂，但不意味着 probe 本身无效。跨样本错配和时间偏移属于后续对应关系／同步性议题，不作为已经完成的依赖验证。

## 7. 数据与历史依据

- [本次十四组完整指标](./modality_reliance_assets/metrics.json)
- [本次来源与哈希](./modality_reliance_assets/provenance.json)
- [原始预测文件清单与窗口指纹](./missing_modality_ID_assets/provenance.json)

数据来自此前读取的 Vanilla 全输入与两种 mask 输入的 Mega-MMDF 预测，使用缓存中的 FAFV 子组；本次无需 SSH、重新推理或读取大体积 feature。尚未得到任何 CKA 数值。
'''
)
(ROOT / 'outputs/modality_reliance_report.md').write_text('\n\n'.join(parts), encoding='utf-8')
print('Wrote modality_reliance_report.md; 14 paired conditions, CKA not yet computed.')
for r in contrast:
    print(r)
