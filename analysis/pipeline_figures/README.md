# Pipeline comparison figures

`run_pipeline_figures.py` builds four appendix pages from 770 saved AUCs: seven detectors × five training sources × 22 Mega-MMDF test pipelines.

Install PyMuPDF in the project environment to render the PDF pages for review.

```bash
pip install PyMuPDF
python analysis/pipeline_figures/run_pipeline_figures.py
```

Paths resolve from the module location. Defaults are `fixtures/pipeline_auc_770.csv`, `fixtures/pipeline_auc_770.json`, and `outputs/`. Override them with:

```bash
python analysis/pipeline_figures/run_pipeline_figures.py \
  --input-csv /path/to/pipeline_auc_770.csv \
  --provenance-json /path/to/pipeline_auc_770.json \
  --output-dir /path/to/figures \
  --dpi 600
```

The CSV and provenance JSON must agree on all 770 unique detector/training-source/pipeline keys. Source paths, hashes, section locations, and timestamps are retained as provenance; original logs are unnecessary for plotting. `fixtures/historical_sources.json` records the source CSV/audit/version hashes.

The first three pages pair Baseline/AVFF, AVH/FauForensics, and FGMDF/FRADE in 5 × 2 layouts. MCL uses 5 × 1. Rows encode training datasets, colors visual operations, and hatches audio operations. FS includes EFS+FS pipelines 6 and 10. The displayed AUC range starts at 0.2 with a 0.5 reference line; input checks enforce the range and pipeline groups.

Outputs are 600 dpi PNGs, individual and combined PDFs, `captions.tex`, numeric CSV/JSON, `verification.json`, and page renders in `render_check/`. LaTeX references PDFs relative to the output directory. Training-free methods have no training-source axis in this experiment.
