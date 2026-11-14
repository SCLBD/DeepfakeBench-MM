# Training and resume

Prepare every dataset named in the configuration's `train_datasets` and
`val_datasets`. For example, the Baseline Mega-MMDF configuration trains on
Mega-MMDF and validates on FakeAVCeleb, LAV-DF, PolyGlotFake, IDForge, and Mega-MMDF.

```bash
python train.py \
  --detector_path configs/detectors/audio_video/standard/baseline_train_on_megammdf.yaml \
  --log_dir logs/train/baseline_megammdf \
  --seed 1024
```

This starts fresh and clears the selected run directory. Omit `--log_dir` to
create `log_dir/train/<model_name>_<timestamp>/` under the configured logging root.

To preserve files and restore the latest full state in an existing run:

```bash
python train.py \
  --detector_path configs/detectors/audio_video/standard/baseline_train_on_megammdf.yaml \
  --log_dir logs/train/baseline_megammdf \
  --resume
```

Keep the original model, data, optimizer, scheduler, and training-length settings
when resuming. If no full state exists, the trainer reports this and starts from
scratch without removing files.

Model-only test weights are written under `checkpoints/`; resumable states are
under `training_states/`. See the [trainer guide](../../trainers/README.md) and
[scheduler guide](../../schedulers/README.md) for stage-specific optimization.
