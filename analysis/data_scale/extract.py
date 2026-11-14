"""Read training-data scale results from a workbook into outputs/data.json."""

import argparse
import hashlib
import json
import math
import statistics

from pathlib import Path

MODELS = ['Baseline', 'AVFF', 'AVH', 'FauForensics', 'FGMDF', 'FRADE', 'MCL']
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
DATASETS = ['FakeAVCeleb', 'LAV-DF', 'PolyGlotFake', 'IDForge', 'TalkingHeadBench', 'DigiFakeAV', 'MMDF']
SHEET = 'Small-scale scaling law'
MODULE_DIR = Path(__file__).resolve().parent
OUT = MODULE_DIR / 'outputs'


def extract(workbook):
    import openpyxl

    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    ws = wb[SHEET]
    records, discrepancies = [], []
    for row in ws.iter_rows(min_row=8, max_row=56):
        v = [c.value for c in row]
        model, fraction, within = v[:3]
        assert model in MODELS and fraction in FRACTIONS
        scores = v[5:12]
        assert len(scores) == 7 and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in scores)
        assert isinstance(within, (int, float)) and 0 <= within <= 1
        cross = v[3]
        assert isinstance(cross, (int, float)) and 0 <= cross <= 1
        recomputed = statistics.mean(scores)
        if not math.isclose(recomputed, cross, abs_tol=1e-10):
            discrepancies.append(dict(cell=f'D{row[0].row}', recorded=cross, recomputed=recomputed))
        records.append(
            dict(
                model=model,
                fraction=fraction,
                within=within,
                cross=cross,
                per_dataset=dict(zip(DATASETS, scores)),
                source_row=row[0].row,
                status=v[12],
                provisional='待确认' in str(v[12]),
                remote_log=v[14],
                remote_log_line=v[15],
            )
        )
    for model in MODELS:
        assert sorted(r['fraction'] for r in records if r['model'] == model) == FRACTIONS
    audit = dict(
        workbook=str(workbook),
        sha256=hashlib.sha256(workbook.read_bytes()).hexdigest(),
        sheet=SHEET,
        range='A8:P56',
        aggregation='Use recorded C and D values exactly, per user request; F:L recomputation is audit only.',
        discrepancies=discrepancies,
        records=records,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'data.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    wb.close()
    print(
        json.dumps(
            dict(
                rows=len(records),
                discrepancies=discrepancies,
                provisional=[(r['model'], r['fraction']) for r in records if r['provisional']],
            ),
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, required=True, help='Source results workbook.')
    extract(parser.parse_args().workbook)
