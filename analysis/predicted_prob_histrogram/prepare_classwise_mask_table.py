"""Six class-specific masking conditions; signed error-rate and MSE deltas."""

import json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/missing_modality_paper'
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
MODELS = ['Baseline', 'AVFF', 'AVH', 'FGMDF', 'FRADE', 'FauForensics', 'MCL']
single_path = OUT / 'metrics.json'
both_path = OUT / 'fafv_metrics.json'
single = json.loads((single_path if single_path.is_file() else FIXTURES / single_path.name).read_text(encoding='utf-8'))
both = json.loads((both_path if both_path.is_file() else FIXTURES / both_path.name).read_text(encoding='utf-8'))
old_single = json.loads((FIXTURES / 'three_settings_metrics.json').read_text(encoding='utf-8'))
old_both = json.loads((FIXTURES / 'modality_reliance_metrics.json').read_text(encoding='utf-8'))


def fmt(x):
    return '$0.000$' if round(x, 3) == 0 else f'${x:+.3f}$'


lines = [
    r'\begin{table*}[t]',
    r'\centering',
    r'\caption{Class-specific changes under modality masking on Mega-MMDF. Each entry reports $\Delta=\mathrm{masked}-\mathrm{full}$ on the same clips. FPR and FNR use a fixed threshold of 0.5. MSE is computed against the class label (real: 0; fake: 1). Lower changes are preferable for all columns.}',
    r'\label{tab:classwise_masking}',
    r'\setlength{\tabcolsep}{2pt}',
    r'\renewcommand{\arraystretch}{1.12}',
    r'\scriptsize',
    r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc|cccccc@{}}',
    r'\toprule',
    r'& \multicolumn{6}{c|}{Mask audio: video retained} & \multicolumn{6}{c}{Mask video: audio retained} \\',
    r'\cmidrule(lr){2-7}\cmidrule(lr){8-13}',
    r'& \multicolumn{2}{c}{RARV} & \multicolumn{2}{c}{RAFV} & \multicolumn{2}{c|}{FAFV} & \multicolumn{2}{c}{RARV} & \multicolumn{2}{c}{FARV} & \multicolumn{2}{c}{FAFV} \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}',
    'Detector & '
    + ' & '.join(
        [
            r'$\Delta$FPR $\downarrow$ & $\Delta$MSE $\downarrow$',
            r'$\Delta$FNR $\downarrow$ & $\Delta$MSE $\downarrow$',
            r'$\Delta$FNR $\downarrow$ & $\Delta$MSE $\downarrow$',
        ]
        * 2
    )
    + r' \\',
    r'\midrule',
]
results = []
for model in MODELS:
    values = []
    row = {'model': model, 'conditions': []}
    b = next(r for r in both if r['model'] == model)
    for condition, group in [('mask_audio', 'RAFV'), ('mask_video', 'FARV')]:
        s = next(r for r in single if r['model'] == model and r['fake'] == group)
        old = next(
            r
            for r in old_single
            if r['model'] == model
            and r['dataset'] == 'Mega-MMDF'
            and r['training'] == 'Vanilla'
            and r['condition'] == condition
            and r['fake_groups'] == group
        )
        ob = next(
            r
            for r in old_both
            if r['model'] == model and r['dataset'] == 'Mega-MMDF' and r['training'] == 'Vanilla' and r['condition'] == condition
        )
        assert abs(s['delta']['fpr'] - b['delta_' + condition]['fpr']) < 1e-12
        assert abs(old['delta_brier_real'] - ob['delta_brier_real']) < 1e-12
        for cl, rate, mse in [
            ('RARV', s['delta']['fpr'], old['delta_brier_real']),
            (group, s['delta']['fnr'], old['delta_brier_fake']),
            ('FAFV', b['delta_' + condition]['fnr'], ob['delta_brier_fake']),
        ]:
            values.extend([fmt(rate), fmt(mse)])
            row['conditions'].append(dict(condition=condition, group=cl, delta_error_rate=rate, delta_mse=mse))
    results.append(row)
    name = model + (r'~\cite{' + model + '}' if model != 'Baseline' else '')
    lines.append(' & '.join([name] + values) + r' \\')
lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table*}']
table = '\n'.join(lines) + '\n'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'classwise_mask_table.tex').write_text(table, encoding='utf-8')
(OUT / 'classwise_mask_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
test = OUT / 'table_layout_check'
test.mkdir(exist_ok=True)
refs = '\n'.join(r'\bibitem{' + m + '} ' + m + '.' for m in MODELS if m != 'Baseline')
(test / 'main.tex').write_text(
    r'''\documentclass[letterpaper,journal]{IEEEtran}
\usepackage{amsmath,amsfonts,array,booktabs,cite}
\begin{document}
\title{Modality Masking: Table Layout Check}
\author{Layout preview}
\maketitle
'''
    + table
    + r'''
\section{Layout check}
This preview uses the manuscript's IEEEtran journal class and a full-width table without resizing. The manuscript body text provides a font-size reference.
\begin{thebibliography}{99}
'''
    + refs
    + r'''
\end{thebibliography}
\end{document}
''',
    encoding='utf-8',
)
print(table)
