# Temporal misalignment analysis

These scripts compare saved aligned predictions with the real-only within-clip audio-window shift protocol in the [misalignment configs](../../configs/detectors/audio_video/misaligned_test/README.md).

## Collect native windows

```bash
python analysis/misalignment/collect.py \
  --original-root logs/test --shifted-root logs/misaligned_test \
  --output analysis/misalignment/outputs/pairs.jsonl.gz
```

Each root must contain seven model directories. Original runs are `{Model}_train_on_Mega-MMDF`; shifted runs are `{Model}_train_on_Mega-MMDF_shift_audio_1`. Models are AVFF, AVH, Baseline, FGMDF, FRADE, FauForensics, and MCL. Each run needs `log.txt` with a serialized `Config:` block and `predictions/Mega-MMDF.json`.

Shifted records must retain native `path`, `sample_frame_range`, actual audio/video ranges, `misalignment_applied`, `num_samples_in_data`, and zero source-authenticity labels. Pairing uses `(path, sample_frame_range)`. The collector checks duplicates, labels/ranges, finite scores, and configuration differences, then writes paired JSONL and `pairs.jsonl.audit.json` with hashes and unmatched/configuration diagnostics. Native prediction inputs and the full paired cache are external.

## Plot saved pairs

```bash
python analysis/misalignment/plot.py
python analysis/misalignment/plot_hist_cdf.py
```

`plot.py` reads module-local `outputs/pairs.jsonl.gz` and writes `summary.json`, full-range/per-model zoom PNG/PDFs, and a report there. `plot_hist_cdf.py` uses the same cache and writes under project `outputs/misalignment_paper/`; create that directory first. `plot_paper.py` and `plot_common_zoom.py` provide alternative layouts; the latter also needs `summary.json`.

Statistics include actually shifted windows (`applied == true`); unshifted windows remain controls. Response is `misaligned - aligned` at the native-window level. The selected histogram displays `[-0.1,0.1]`, while the CDF and statistics use all included windows. The historical AVFF unchanged-control mismatch remains in the source provenance. Changes to shift/modality require matching config/metadata changes and separate run directories.

## Ten-method paper figure

```bash
python analysis/misalignment/plot_ten_methods.py
```

The default input is bundled `fixtures/ten_method_deltas.json.gz`; `--input` and `--output-dir` select other paths. The 2 × 5 histogram/CDF layout displays `[-0.1,0.3]` to include SpeechForensics' larger positive response. Both use all included windows as denominator; saved summaries include full values and out-of-view fractions.

To reconstruct the compact fixture from actual shift0/shift1 records and audited seven-model pairs:

```bash
python analysis/misalignment/collect_external_misalignment.py \
  --input-root '/path/to/test result/misalign test' \
  --existing-pairs /path/to/audited_seven_model_pairs.jsonl.gz \
  --output /path/to/deltas.json.gz
```

This collector verifies duplicates, source-real labels, shared anchors, applied shifts, distinct donors, and non-overlapping source audio/video intervals. Single-window/unapplied shifts are excluded; method-specific eligible cohorts reflect different input lengths.
