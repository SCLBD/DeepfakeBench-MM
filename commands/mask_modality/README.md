# Missing-modality tests

The test-time mask represents unavailable audio by a zero waveform, or unavailable
video by black frames, before the model's representation and normalization steps.
No offline perturbation generation is needed.

```bash
python test.py \
  --detector_path configs/detectors/audio_video/standard/baseline_train_on_megammdf.yaml \
  --weight_path /path/to/model_only_checkpoint.pth \
  --test_datasets Mega-MMDF \
  --robustness_type mask_audio \
  --log_dir logs/test/baseline_megammdf_mask_audio \
  --metrics 'video-level AUC'

python test.py \
  --detector_path configs/detectors/audio_video/standard/baseline_train_on_megammdf.yaml \
  --weight_path /path/to/model_only_checkpoint.pth \
  --test_datasets Mega-MMDF \
  --robustness_type mask_video \
  --log_dir logs/test/baseline_megammdf_mask_video \
  --metrics 'video-level AUC'
```

Do not pass `--robustness_param` for modality masks. Each condition needs its own
test output directory, including the complete-input reference.

Saved test records contain effective labels after masking and the pre-mask
`original_audio_label`, `original_video_label`, and `original_label`. Removing a
forged modality can change the effective label. Select the intended original
authenticity groups when reproducing a RARV/RAFV/FARV/FAFV comparison; the command
does not automatically construct those separate cohorts.

Prediction-only export is available by omitting `--metrics`. The
[mask-analysis tools](../../analysis/predicted_prob_histrogram/README.md) describe
the paired prediction inputs used for distributions and score changes.

The [seven-detector batch](mask_modality_test_commands.sh) runs both masking
conditions on the five datasets listed in the script. To collect the Mega-MMDF
predictions for the paper's comparison, first obtain the full-input runs as
described in [standard testing](../test/README.md). Use
`logs/test/<Model>_train_on_Mega-MMDF` for each full-input run.

```bash
bash commands/mask_modality/mask_modality_test_commands.sh
python analysis/predicted_prob_histrogram/collect_mask_manifest.py \
  --log-root logs --masked-log-root logs/mask_modality --datasets Mega-MMDF \
  --output outputs/mask_manifest.json
```
