# Mega-MMDF raw-clip statistics

`compute_dataset_statistics.py` collects resumable metadata from original clips and split annotations, then finalizes the JSONL as parquet. Counts refer to raw clips, separately from preprocessed segments or model windows.

Install `pyarrow` to write Parquet with `--finalize` or read it when plotting.

```bash
pip install pyarrow
```

```bash
python analysis/dataset_statistics/compute_dataset_statistics.py \
  --root /path/to/original/Mega-MMDF \
  --output analysis/dataset_statistics/plots/dataset_statistics.jsonl \
  --workers 8
python analysis/dataset_statistics/compute_dataset_statistics.py \
  --output analysis/dataset_statistics/plots/dataset_statistics.jsonl --finalize
```

`--root` defaults to the current directory, `--output` to `<root>/dataset_statistics.jsonl`, `--workers` to 96, and `--threshold` to 5e-4. `--limit` caps remaining tasks; existing JSONL paths are skipped on resume. `--finalize` converts the existing JSONL to same-stem parquet and prints a summary.

The collector enumerates original split JSONs and reads original audio/video files. Records contain paths, splits, modality/pipeline, demographic annotations, duration, FPS, sample/frame counts, file size, and leading-silence duration. Demographic fields use supplied annotations; silence uses the configured waveform threshold.

[Plot scripts](plots/README.md) use `plots/dataset_statistics.parquet` and, for resolution plots, `plots/resolution.jsonl`. Supply the verified 1.3M-raw-clip metadata or regenerate it from the matching complete raw-data/annotation version. These large inputs are external. Plots and numerical summaries are saved alongside the helpers. New statistics should retain the same raw-clip unit, metadata version, and split conventions.
