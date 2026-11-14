# t-SNE command batches

The shell batches call the t-SNE entry points using saved predictions and named feature pickles. Each changes to the project root and accepts `LOG_ROOT` (default `logs`).

```bash
LOG_ROOT=/path/to/logs bash analysis/tsne/commands/run_tsne_every_pipeline_on_Mega-MMDF_exp.sh
```

`run_tsne_real_vs_fake_on_<dataset>.sh` and `run_tsne_RARV_vs_RAFV_vs_FARV_vs_FAFV_on_<dataset>.sh` cover the eight benchmark datasets. Each contains 105 commands: seven detector families × five training sources × three feature types. The three Mega-MMDF grouping batches also have 105 commands. Their `_exp.sh` variants contain 198 commands: FGMDF, FRADE, and FauForensics × 22 single-pipeline training sources × three features.

| Grouping | Test-run script | Single-pipeline script | Samples per group |
|---|---|---|---:|
| Pipeline | `run_tsne_every_pipeline_on_Mega-MMDF.sh` | `run_tsne_every_pipeline_on_Mega-MMDF_exp.sh` | 200 |
| Visual operation | `run_tsne_visual_forgery_on_Mega-MMDF.sh` | `run_tsne_visual_forgery_on_Mega-MMDF_exp.sh` | 500 |
| Audio operation | `run_tsne_audio_forgery_on_Mega-MMDF.sh` | `run_tsne_audio_forgery_on_Mega-MMDF_exp.sh` | 500 |

MCL uses `a_feat`, `v_feat`, and `av_feat`; the other detectors use `audio_feature`, `video_feature`, and `fusion_feature`. Each command requires matching `<log_dir>/predictions/<dataset>.json` and `<log_dir>/features/<feature_name>/<dataset>.pkl` inputs with native path/frame-range indexing. Feature caches are external inputs.

Batches run sequentially, continue after individual failures, and return nonzero if any command fails. Existing PNG/HTML filenames are redrawn. Defaults write to `analysis/tsne/outputs/`, with model/training/group/feature represented in titles. Use the [individual entry points](../README.md) for smaller comparisons or custom output paths and sampling settings.
