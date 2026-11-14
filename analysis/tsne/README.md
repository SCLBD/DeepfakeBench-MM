# t-SNE feature analysis

The entry points load saved predictions and feature pickles, sample groups, and write PNG and interactive HTML projections. Inputs are `<log_dir>/predictions/<dataset>.json` and `<log_dir>/features/<feature_name>/<dataset>.pkl`, indexed consistently by `(path, sample_frame_range)`.

| Script | Grouping |
|---|---|
| `run_tsne_real_vs_fake_one_dataset.py` | Real versus fake. |
| `run_tsne_RARV_vs_RAFV_vs_FARV_vs_FAFV_one_dataset.py` | Four audio/video authenticity combinations. |
| `run_tsne_every_pipeline_on_Mega-MMDF.py` | Real plus manipulation pipelines 1–22. |
| `run_tsne_visual_forgery_on_Mega-MMDF.py` | Eight merged visual-operation categories. |
| `run_tsne_audio_forgery_on_Mega-MMDF.py` | Five audio categories: Real Audio, TTS, VC, PF, and T2AV. |

```bash
python analysis/tsne/run_tsne_real_vs_fake_one_dataset.py \
  --log_dir logs/test/AVFF_train_on_Mega-MMDF --feature_name fusion_feature \
  --dataset_name Mega-MMDF --title 'AVFF - Real vs. Fake - Fusion Feature' --max_samples_per_group 10000
python analysis/tsne/run_tsne_visual_forgery_on_Mega-MMDF.py \
  --log_dir logs/test/AVFF_train_on_Mega-MMDF --feature_name video_feature \
  --title 'AVFF - Visual Forgery - Video Feature'
python analysis/tsne/run_tsne_audio_forgery_on_Mega-MMDF.py \
  --log_dir logs/test/AVFF_train_on_Mega-MMDF --feature_name audio_feature \
  --title 'AVFF - Audio Forgery - Audio Feature'
```

Outputs default to `analysis/tsne/outputs/`; `--title` supplies the filename stem. Shared CLI settings include `--perplexity 30`, `--max-iter 1000`, `--sample-seed 0`, `--tsne-seed 0`, and `--dpi 600`.

The every-pipeline entry point defaults to at most 200 samples per group (at most 4,600 points). Pipeline0 belongs to Real; `--pipelines` selects or reorders manipulation pipelines. The modality entry points default to 500 samples per merged category, at most 4,000 visual or 2,500 audio points. Sampling is uniform within merged categories; `--max_samples_per_group` overrides the limit. The grouping modality is independent of `--feature_name`, and separate calls fit independent coordinates. Real paths without a pipeline ID use pipeline0; ambiguous/unknown pipeline paths raise an error.

`forgery_groups.py` defines merged categories. `tsne_types.py` defines groups/configuration/results, `core.py` handles feature preparation and projection, and `plotting.py` renders static/interactive outputs. [Command batches](commands/README.md) cover the benchmark datasets and training sources.

## Minimal example

Run the [demo](run_tsne_demo.py) to project 20 hand-written feature vectors with the same API. It writes a PNG and an interactive HTML file under `outputs/`.

```bash
python analysis/tsne/run_tsne_demo.py
```
