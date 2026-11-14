# Cross-pipeline experiments

## Prepare the pipeline manifests

After preprocessing Mega-MMDF, run from the repository root:

```bash
python commands/cross_pipeline/prepare_pipeline_manifests.py
```

The script uses `json_dir/audio-video/Mega-MMDF.json` from [configs/path.yaml](../../configs/path.yaml).
For each fake pipeline `Pi` (1–22), it creates `Mega-MMDF-pipeline<i>.json` in the same configured
directory. Each train, validation and test split keeps every real `P0` record and every fake `Pi`
record from its original split. Records keep their paths, labels and other metadata. The manifests
refer to the same preprocessed media as the full dataset.

Use `--pipelines 1 10` to create selected manifests, `--source-json` and `--output-dir` to customize
paths, or `--overwrite` to regenerate existing manifests. Training and testing read the manifests
from the `json_dir/audio-video/` directory configured for the experiment.

## Train and test

The [configurations](../../configs/detectors/audio_video/cross_pipeline/README.md) train FGMDF,
FRADE and FauForensics separately on each pipeline and select checkpoints on the same pipeline's
validation split. To run all 66 training configurations:

```bash
bash commands/cross_pipeline/single_pipeline_train_commands.sh
```

The commands start fresh in `logs/exp_train/`. Use the [training guide](../train/README.md) to resume
an individual run. For testing, put each selected model-only checkpoint in `WEIGHTS_DIR` with a
name such as `FGMDF_train_on_pipeline1.pth`:

```bash
WEIGHTS_DIR=/path/to/selected/checkpoints bash commands/cross_pipeline/single_pipeline_test_commands.sh
```

The test batch evaluates each trained model on all 22 pipeline manifests and saves video-level
AUC and predictions under `logs/exp/`. For pipeline-wise evaluation of models trained on complete
datasets, pass the same test manifest names to [standard testing](../test/README.md).
The [pipeline figures](../../analysis/pipeline_figures/README.md) and [t-SNE guide](../../analysis/tsne/README.md)
describe the corresponding analysis tools.
