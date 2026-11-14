"""Rebuild four appendix pages from the audited 770 local pipeline AUCs.

No inference, source edits, or remote access. Run from any working directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))

import fitz
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.patches import Patch

MODELS = ['Baseline', 'AVFF', 'AVH', 'FauForensics', 'FGMDF', 'FRADE', 'MCL']
TRAIN = ['FakeAVCeleb', 'LAV-DF', 'PolyGlotFake', 'IDForge', 'Mega-MMDF']
ORDER = [3, 1, 4, 6, 10, 2, 5, 14, 18, 9, 13, 17, 21, 7, 11, 15, 19, 8, 12, 16, 20, 22]
VISUAL = [('Real', [3], '#8C8C8C'), ('FS', [1, 4, 6, 10], '#4C78A8'),
          ('FR', [2, 5, 14, 18], '#F58518'), ('FS+FR', [9, 13, 17, 21], '#E45756'),
          ('EFS+FR', [7, 11, 15, 19], '#72B7B2'), ('FE+FR', [8, 12, 16, 20], '#54A24B'),
          ('T2AV', [22], '#B279A2')]
AUDIO = [('Real', [1, 2, 6, 7, 8, 9], ''), ('VC', [3, 4, 5, 10, 11, 12, 13], '////'),
         ('TTS', [14, 15, 16, 17], '...'), ('PF', [18, 19, 20, 21], '---'), ('T2AV', [22], 'xx')]
PAGES = [['Baseline', 'AVFF'], ['AVH', 'FauForensics'], ['FGMDF', 'FRADE'], ['MCL']]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identify(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def visual(pid):
    return next((name, color) for name, ids, color in VISUAL if pid in ids)


def audio(pid):
    return next((name, hatch) for name, ids, hatch in AUDIO if pid in ids)


def load_and_verify(source, provenance_path):
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    actual = [(r['detector'], r['train_dataset'], int(r['pipeline_id'])) for r in rows]
    expected = set(itertools.product(MODELS, TRAIN, range(1, 23)))
    if len(rows) != 770 or len(set(actual)) != 770 or set(actual) != expected:
        raise ValueError('Expected every one of 7 x 5 x 22 unique pipeline results')
    values = dict(zip(actual, (float(r['video_auc']) for r in rows)))
    if not all(.2 <= value <= 1 for value in values.values()):
        raise ValueError('A value is outside the current displayed range [0.2,1]')
    for row in rows:
        pid = int(row['pipeline_id'])
        if row['visual_group'] != visual(pid)[0] or row['audio_group'] != audio(pid)[0]:
            raise ValueError('Pipeline grouping differs from the defined plot order')
        if int(row['visual_order']) != ORDER.index(pid) + 1:
            raise ValueError('Unexpected visual plot order')
    provenance = json.loads(provenance_path.read_text(encoding='utf8'))
    cells = provenance['values']
    keys = [(r['model'], r['training_dataset'], r['pipeline_id']) for r in cells]
    if len(cells) != 770 or len(set(keys)) != 770 or set(keys) != expected:
        raise ValueError('Incomplete or duplicate packaged provenance')
    sources = {}
    for key, cell in zip(keys, cells):
        if abs(values[key] - cell['video_auc']) >= 1e-12:
            raise ValueError(f'CSV differs from audited per-cell provenance: {key}')
        source_key = key[:2]
        sources[source_key] = dict(model=key[0], training_dataset=key[1],
                                  path=cell['source_log'], log_sha256=cell['source_log_sha256'],
                                  table_heading_line=cell['source_section_line'], time=cell['source_time'])
    return rows, values, list(sources.values()), cells


def legend(fig, single):
    vh = [Patch(facecolor='none', edgecolor='none', label='Visual:')]
    vh += [Patch(facecolor=color, edgecolor='none', label=name) for name, _, color in VISUAL]
    ah = [Patch(facecolor='none', edgecolor='none', label='Audio:')]
    ah += [Patch(facecolor='white', edgecolor='#343434', hatch=hatch, label=name, linewidth=.4)
           for name, _, hatch in AUDIO]
    if single:
        # Matplotlib fills legend columns first; reorder handles for left-to-right rows.
        vh = [vh[i] for i in [0, 4, 1, 5, 2, 6, 3, 7]]
        ah = [ah[i] for i in [0, 3, 1, 4, 2, 5]]
        sets = [(vh, .063, 4), (ah, .008, 3)]
    else:
        sets = [(vh, .042, 8), (ah, .008, 6)]
    for handles, bottom, ncol in sets:
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.51, bottom), ncol=ncol,
                   frameon=False, fontsize=7.0, handlelength=1.10, handleheight=.90,
                   handletextpad=.36, columnspacing=.70, borderaxespad=0, labelspacing=.4)


def draw_page(models, values, number, output, dpi):
    single = len(models) == 1
    size = (3.50, 6.00) if single else (7.16, 6.00)
    fig = plt.figure(figsize=size)
    left, right = (.125, .985) if single else (.065, .987)
    gap = 0 if single else .080
    width = (right - left - gap * (len(models) - 1)) / len(models)
    bottom, top = (.195, .928) if single else (.157, .928)
    row_gap = .044
    height = (top - bottom - 4 * row_gap) / 5
    for col, model in enumerate(models):
        xleft = left + col * (width + gap)
        fig.text(xleft + width / 2, .975, model, ha='center', va='center', fontsize=10.0, fontweight='bold')
        for row, training in enumerate(TRAIN):
            ybottom = top - (row + 1) * height - row * row_gap
            ax = fig.add_axes([xleft, ybottom, width, height])
            group_start = 0
            for name, ids, color in VISUAL:
                ax.axvspan(group_start - .5, group_start + len(ids) - .5, color=color, alpha=.10, lw=0, zorder=0)
                if group_start:
                    ax.axvline(group_start - .5, color='#AAAAAA', lw=.35, zorder=1)
                group_start += len(ids)
            for xpos, pid in enumerate(ORDER):
                ax.bar(xpos, values[model, training, pid] - .2, bottom=.2, width=.82,
                       color=visual(pid)[1], hatch=audio(pid)[1], edgecolor='#343434', linewidth=.30, zorder=3)
            ax.axhline(.5, color='#555555', ls=(0, (1.5, 1.8)), lw=.65, zorder=4)
            ax.set(xlim=(-.6, 21.6), ylim=(.2, 1.025), yticks=[.2, .6, 1.0])
            ax.set_xticks(range(22), [str(pid) for pid in ORDER])
            ax.tick_params(length=1.8, width=.45, pad=1.7, labelsize=6.8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylabel('AUC', fontsize=7.2, labelpad=2.4)
            if col == 0:
                fig.text((left + right) / 2, ybottom + height + .0105,
                         f'Training source: {training}', ha='center', va='center', fontsize=8.0)
    fig.text((left + right) / 2, bottom - .051, 'Mega-MMDF test pipeline ID', ha='center', fontsize=8.0)
    legend(fig, single)
    stem = f'pipeline_appendix_{number:02d}_' + '_'.join(models).lower()
    png, pdf = output / f'{stem}.png', output / f'{stem}.pdf'
    fig.savefig(png, dpi=dpi, facecolor='white')
    fig.savefig(pdf, dpi=dpi, facecolor='white')
    plt.close(fig)
    return dict(page=number, models=models, rows=5, columns=len(models),
                training_order=TRAIN, value_count=5 * len(models) * 22,
                width_inches=size[0], height_inches=size[1], png=png.name, pdf=pdf.name)


def write_captions(output, pages):
    common = ('Video-level AUC on all 22 Mega-MMDF test pipelines after training on each of the five development datasets. '
              'Rows indicate training sources and columns indicate detectors. '
              'Colors encode visual-operation groups and hatches encode audio-operation groups, following the main-text figure; '
              'the FS color also includes EFS+FS pipelines 6 and 10. '
              'Pipeline IDs are shown in the same grouped order in every panel. '
              'The dotted line marks AUC=0.5; the vertical axis starts at 0.2. '
              'These are individual pipeline AUCs, without pooling predictions across pipelines.')
    lines = ['% Generated figure fragments. Paths are relative to this directory.',
             '% Requires graphicx. No training-free results are represented.']
    for page in pages:
        single = len(page['models']) == 1
        environment = 'figure' if single else 'figure*'
        width = r'\columnwidth' if single else r'\textwidth'
        models = ' and '.join(page['models'])
        lines += [f'\\begin{{{environment}}}[!t]', r'  \centering',
                  f"  \\includegraphics[width={width}]{{{page['pdf']}}}",
                  f'  \\caption{{Pipeline evaluation for {models}. {common}}}',
                  f"  \\label{{fig:appendix-pipelines-{page['page']:02d}}}",
                  f'\\end{{{environment}}}', '']
    (output / 'captions.tex').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, default=HERE / 'fixtures/pipeline_auc_770.csv')
    parser.add_argument('--provenance-json', type=Path, default=HERE / 'fixtures/pipeline_auc_770.json')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, values, sources, cells = load_and_verify(args.input_csv, args.provenance_json)
    snapshot = output / 'pipeline_auc_770.csv'
    if snapshot.exists():
        assert snapshot.read_bytes() == args.input_csv.read_bytes(), 'Source changed; use a new output directory'
    else:
        shutil.copyfile(args.input_csv, snapshot)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 8, 'axes.linewidth': .5, 'hatch.linewidth': .28,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    pages = [draw_page(models, values, i + 1, output, args.dpi) for i, models in enumerate(PAGES)]
    assert sum(p['value_count'] for p in pages) == 770
    write_captions(output, pages)
    combined = fitz.open()
    renders = output / 'render_check'
    renders.mkdir(exist_ok=True)
    for p in pages:
        with fitz.open(output / p['pdf']) as doc:
            assert len(doc) == 1
            combined.insert_pdf(doc)
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(180 / 72, 180 / 72), alpha=False)
            pix.save(renders / f"page_{p['page']:02d}.png")
            txt = doc[0].get_text()
            assert all(model in txt for model in p['models'])
            assert all(training in txt for training in TRAIN)
            assert 'Mega-MMDF test pipeline ID' in txt
    combined.save(output / 'pipeline_appendix_all_4pages.pdf')
    combined.close()
    data = dict(metric='video-level AUC', models=MODELS, training_sources=TRAIN, pipeline_order=ORDER,
                visual_groups=[dict(name=n, pipeline_ids=p, color=c) for n, p, c in VISUAL],
                audio_groups=[dict(name=n, pipeline_ids=p, hatch=h) for n, p, h in AUDIO],
                values=cells)
    (output / 'pipeline_auc_770.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    report = dict(status='verified against packaged audited provenance; rendered for visual QA', source_csv=identify(args.input_csv),
                  provenance=identify(args.provenance_json), remote_log_sources=sources,
                  expected_values=770, unique_values=len(values), raw_csv_rows=len(rows), source_cell_mismatches=0,
                  dpi=args.dpi, pages=pages, code=identify(__file__),
                  notes=['No inference or training; all bars preserve source CSV values.',
                         'Sources are historical audited log summaries, not new remote reads.',
                         'Y-axis begins at 0.2, preserving the body figure scale; dotted reference at 0.5.',
                         'Visual FS groups pipelines 1,4,6,10; P6/P10 are EFS+FS in pipeline definitions.',
                         'No AUC mean or pooled score was substituted for a pipeline value.'],
                  outputs=[identify(output / p[e]) for p in pages for e in ['png', 'pdf']])
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(pages=len(pages), checked_values=len(values), source_cell_mismatches=0,
                          output=str(output)), indent=2))


if __name__ == '__main__':
    main()
