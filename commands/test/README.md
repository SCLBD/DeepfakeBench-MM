# Testing

Use the configuration corresponding to the checkpoint, and prepare the assembled
manifest for each test dataset. Replace the weight path with a model-only file
from `checkpoints/` or a released detector checkpoint.

```bash
python test.py \
  --detector_path configs/detectors/audio_video/standard/baseline_train_on_megammdf.yaml \
  --weight_path /path/to/model_only_checkpoint.pth \
  --test_datasets Mega-MMDF FakeAVCeleb LAV-DF \
  --log_dir logs/test/baseline_megammdf \
  --metrics 'sample-level AUC' 'video-level AUC'
```

Predictions are saved under `--log_dir/predictions/<dataset>.json`. Omit
`--metrics` to run prediction-only inference. Add `--feature_keys` followed by
the feature names supported by the detector when feature export is needed.

The same output directory, checkpoint, and data setting can be reused to compute
additional metrics from complete saved predictions. Choose a new directory for
a different checkpoint, dataset version, or intervention; cached files are
recognized by existence rather than checkpoint identity.

Full `training_states/` containers are for [training resume](../train/README.md). See the
[metric guide](../../metrics/README.md) for metric names, aggregation, and annotation
requirements.
