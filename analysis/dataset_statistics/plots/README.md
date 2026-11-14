# Mega-MMDF statistics plots

These scripts read saved raw-clip metadata. See [collection](../README.md) for JSONL generation and parquet conversion. Inputs and outputs resolve next to the plotting scripts; the full parquet and resolution JSONL are external inputs. Install `pyarrow` as shown in the [collection instructions](../README.md) before running scripts that read Parquet.

| Script | Input and output |
|---|---|
| `duration_distribution.py` | Parquet; duration histogram/CDF and quantiles. |
| `file_size_distribution.py` | Parquet; audio+video size histogram/CDF and summary. |
| `pipeline_counts.py` | Parquet; pipeline count bars/pies and counts. |
| `demographic_pie.py` | Parquet annotations; demographic distributions by split. |
| `leading_silence_hist.py` | Parquet; real-audio RARV+RAFV versus fake-audio FARV+FAFV silence density. |
| `resolution_pie.py` | `resolution.jsonl`; overall resolution shares/counts. |
| `resolution_analysis.py` | `resolution.jsonl`; resolution by modality/pipeline and video authenticity. |
| `extract_resolution.py` | Original media and parquet path list; resumable ffprobe JSONL with dimensions or errors. |

For example:

```bash
python analysis/dataset_statistics/plots/duration_distribution.py
python analysis/dataset_statistics/plots/pipeline_counts.py
python analysis/dataset_statistics/plots/demographic_pie.py
python analysis/dataset_statistics/plots/leading_silence_hist.py
python analysis/dataset_statistics/plots/resolution_pie.py
python analysis/dataset_statistics/plots/resolution_analysis.py
```

The plotting entry points have no CLI options. The resolution collector requires `--root` and ffprobe:

```bash
python analysis/dataset_statistics/plots/extract_resolution.py \
  --root /path/to/original/Mega-MMDF \
  --parquet analysis/dataset_statistics/plots/dataset_statistics.parquet \
  --output analysis/dataset_statistics/plots/resolution.jsonl --workers 8
```

Parquet/output default to those filenames next to the script, and `--workers` defaults to 64. Existing JSONL paths are skipped on resume.

`plot_common.py` supplies `load_df()`, `save()`, and `write_stats()`, Agg rendering, 150 dpi, and the Okabe-Ito palette. Duration/size statistics retain tails outside displayed ranges. Resolution plots merge small displayed shares into Other while saving full counts. Leading-silence plots use 5 ms bars, a 1 ms smoothing grid, and 2.5 ms Gaussian bandwidth. Use the same verified raw-clip metadata for all plots.
