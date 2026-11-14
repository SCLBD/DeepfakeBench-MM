# Real-only audiovisual window misalignment

The supplied [configurations](../../configs/detectors/audio_video/misaligned_test/README.md) keep only
RARV data and rotate one modality's complete sampled windows within each clip.
Their default is an audio shift of one window. They use the ordinary trained
checkpoints without changing model parameters.

```bash
python test.py \
  --detector_path configs/detectors/audio_video/misaligned_test/baseline_misaligned_real.yaml \
  --weight_path weights/Baseline_train_on_Mega-MMDF.pth \
  --test_datasets Mega-MMDF \
  --log_dir logs/misaligned_test/Baseline_train_on_Mega-MMDF_shift_audio_1
```

For an aligned control, copy that configuration, set
`data_config.misalignment.shift: 0`, and run it with a separate output directory.
To shift video and landmarks instead, set `modality: video` in a copied
configuration and use another output directory.

The dataset includes clips with one complete window, where reassignment cannot
change the pairing. For paired score-change analysis, filter
`misalignment_applied == true` (at least two windows, `N >= 2`), then join aligned
and shifted records by `(path, sample_frame_range)`. Actual audio/video window
ranges are included in the saved metadata.

The full seven-model [command list](misaligned_test_commands.sh) uses each
detector's Mega-MMDF checkpoint:

```bash
bash commands/misaligned/misaligned_test_commands.sh
```

The labels describe authentic source streams and remain zero. The examples omit
metrics for prediction export; ordinary two-class AUC/EER is undefined for this
real-only cohort. See [misalignment analysis](../../analysis/misalignment/README.md)
for the paired comparison inputs and plotting tools.
