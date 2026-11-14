# Paired modality-mask score analysis

This module collects and analyzes saved complete-input and zero-masked scores for seven Vanilla detectors. The directory spelling is retained. Supply paired complete and masked predictions as inputs.

## Collect paired inputs

The manifest expects `test/{Model}_train_on_Mega-MMDF/predictions/<dataset>.json` below `--log-root`. Masked predictions live at `{Model}_train_on_Mega-MMDF_mask_{audio|video}/predictions/<dataset>.json` below `--masked-log-root`; its default is `<log-root>/mask_modality_test_bkp`. Models are AVFF, AVH, Baseline, FGMDF, FRADE, FauForensics, and MCL. Use `--datasets Mega-MMDF` to collect only the in-domain comparisons.

```bash
python analysis/predicted_prob_histrogram/collect_mask_manifest.py \
  --log-root /path/to/logs --masked-log-root /path/to/masked_runs --output /path/to/mask_manifest.json
python analysis/predicted_prob_histrogram/collect_paired_predictions.py \
  --manifest /path/to/mask_manifest.json --output /path/to/rafv_pairs.json.gz --task RAFV
python analysis/predicted_prob_histrogram/collect_paired_predictions.py \
  --manifest /path/to/mask_manifest.json --output /path/to/farv_pairs.json.gz --task FARV
```

Collectors pair native windows by path/frame range and verify counts and sorted original-label fingerprints. Test-time mask outputs preserve `original_audio_label`, `original_video_label`, and `original_label`; older files need equivalent original-label provenance. Video aggregation uses all evaluated windows by `dirname(path)`, maximum score, and ORed original modality labels before group selection. Different native window lengths retain detector-specific eligible cohorts.

RAFV selects RARV+RAFV and compares full versus mask-audio scores. FARV selects RARV+FARV and compares full versus mask-video. Missing groups are skipped explicitly. Exports retain model/dataset, provenance/fingerprints, and paired `video_id`, `y`, scores, and window counts. These task subsets are separate from complete-domain ROC/PR inputs.

## Metrics and plots

```bash
python analysis/predicted_prob_histrogram/run_predicted_prob_histrogram.py \
  --pairs /path/to/rafv_pairs.json.gz --datasets Mega-MMDF
python analysis/predicted_prob_histrogram/run_mask_video_analysis.py \
  --pairs /path/to/farv_pairs.json.gz --reference /path/to/mask_manifest.json
```

The RAFV entry point exports metrics and histograms; options include `--reference`, `--output-dir`, `--bin-width` (0.01), and `--bandwidth` (0.02). Default plots are `outputs/<dataset>/<model>.{png,pdf}` and `all_models.{png,pdf}`. The FARV entry point requires a reference manifest and exports metrics under `outputs/FARV_mask_video/` by default.

`core.py` implements AUC, class means, Brier scores, score gaps, and paired changes. Standard Brier is `mean((p-y)**2)`; balanced Brier gives real and fake equal weight. Score gap is `mean(p|fake)-mean(p|real)` and deltas are masked minus full. JSON/CSV preserve unrounded values. Cross-domain summaries average dataset-level metrics with equal weight. Historical classwise error rates use fake iff `p >= 0.5`.

Histograms normalize each class separately. Gaussian binned KDE uses boundary reflection and fixed bandwidth; dashed lines show class means. The [paired-change tool](../predicted_prob_change_histogram/README.md) plots differences for the same video IDs.

## Four-group analysis

Generate `outputs/mask_manifest.json` with the
[mask-modality commands](../../commands/mask_modality/README.md). Run the following
from the repository root to collect the video-level differences and generate
pooled and stratified metrics and plots for Mega-MMDF.

```bash
python analysis/predicted_prob_histrogram/collect_missing_groups.py \
  --manifest outputs/mask_manifest.json \
  --output analysis/predicted_prob_histrogram/outputs/missing_groups_ID.json.gz
python analysis/predicted_prob_histrogram/analyze_missing_groups.py \
  --input analysis/predicted_prob_histrogram/outputs/missing_groups_ID.json.gz \
  --mask-manifest outputs/mask_manifest.json --plots
```

The collector defaults to Vanilla and retains all four original-label groups
(RARV, RAFV, FARV, and FAFV) and the full, mask-audio, and mask-video scores for
each video. Use a fresh cache filename for a new collection when prediction
files change. The analyzer writes CSV, JSON, provenance, and optional plots
under `outputs/missing_modality_ID_assets/`, plus
`outputs/missing_modality_report.md`. It can run with the paired cache alone.
Supply `--mask-manifest` to check FAFV AUC and counts, or `--rafv-metrics` and
`--farv-metrics` with the independently generated `metrics.json` files from
the single-fake analyses above.

For legacy random-mask training comparisons, append
`--training RMA RMV --random-mask-root /path/to/random_mask_runs` to the collector
command using the same cache. The augmented checkpoints and predictions are
additional inputs. The analyzer compares available RMA and RMV scores with the
Vanilla scores for the same models and videos.

## Reproduce published assets

The standard command above analyzes new runs. For exact reconstruction from the
historical seven-model predictions, write the paired cache to
`analysis/predicted_prob_histrogram/outputs/missing_groups_ID.json.gz` as shown
above. Then run `prepare_missing_input_paper.py` and `prepare_fafv_paper.py` to
verify and export metrics against the bundled historical values. Follow with
`prepare_classwise_mask_table.py` and `plot_classwise_mask_histograms.py` to
build the six-condition table and histograms. The table can also be rebuilt
directly from `fixtures/`. The table and histogram scripts use regenerated
metrics in `outputs/missing_modality_paper/` when present, with the bundled
values as a fallback.

`build_modality_reliance_report.py` uses the historical Vanilla FAFV results.
`build_three_settings_report.py` requires the historical results for all seven
models under Vanilla, RMA, and RMV.
