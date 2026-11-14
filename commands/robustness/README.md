# Post-processing robustness

Generate a perturbed copy of the prepared test data before running the detector.
For example, generate H.264-compressed video with CRF 23:

```bash
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py \
  --datasets Mega-MMDF --type video_h264 --param 23 --seed 0
```

The default source is the test split, and the default destination is
`preprocessed_dir/audio-video_video_h264_23/`. The generator transforms affected
files and copies the unaffected streams and available landmarks. It keeps the
ordinary dataset manifest; existing final output files are skipped on a rerun.

Then select the same operation and actual parameter in testing:

```bash
python test.py \
  --detector_path configs/detectors/audio_video/standard/baseline_train_on_megammdf.yaml \
  --weight_path /path/to/model_only_checkpoint.pth \
  --test_datasets Mega-MMDF \
  --robustness_type video_h264 --robustness_param 23 \
  --log_dir logs/test/baseline_megammdf_h264_23 \
  --metrics 'video-level AUC'
```

`--param` and `--robustness_param` are actual transformation parameters, not
severity indices. Other operations include `audio_aac`, `video_downscale`,
`video_frame_drop`, `audio_dropout`, and `video_delay`.

The [perturbation manual](../../datasets/utils/perturbations/README.md) documents
parameter units, suggested values, seeds, output naming, and custom destinations.
Use separate generator/output directories when changing auxiliary generation
settings, and a new test `--log_dir` for a different data version.

## Full command lists

[generate_robustness_data.sh](generate_robustness_data.sh) generates six operations
at three actual parameter values. Pass every dataset that will be evaluated:

```bash
bash commands/robustness/generate_robustness_data.sh \
  Mega-MMDF FakeAVCeleb LAV-DF PolyGlotFake IDForge
```

The selected values reproduce the parameter choices in the research
configuration, rather than built-in severity presets:

| Operation | Three actual values |
|---|---|
| `video_h264` | `23`, `30`, `37` |
| `video_downscale` | `0.6666666666666666`, `0.4444444444444444`, `0.3333333333333333` |
| `video_frame_drop` | `0.05`, `0.1`, `0.2` |
| `audio_aac` | `64k`, `32k`, `16k` |
| `audio_dropout` | `0.05`, `0.1`, `0.2` |
| `video_delay` | `0.08`, `0.16`, `0.32` |

Random generation uses `--seed 0`. Audio dropout uses the generator defaults of
20-ms packets and 2-ms boundary fades. Seeds are generation settings and are not
passed to `test.py`.

The [Mega-MMDF-trained list](robustness_test_commands_megammdf.txt) evaluates
seven checkpoints on five datasets. The
[all-training-source list](robustness_test_commands_all_train_sets.txt) evaluates
five training sources on all eight test datasets. Generate each listed test
dataset before running the corresponding list from the repository root:

```bash
bash commands/robustness/robustness_test_commands_megammdf.txt
```

For the eight-dataset list, also generate DigiFakeAV, MMDF, and TalkingHeadBench.
Reuse the clean predictions from [standard testing](../test/README.md) to
compute the change in video-level AUC.

Results go under `logs/robustness_offline/`. Overlapping evaluations in the
two lists share output directories and can reuse predictions. The offline
frame-loss operator acts on source frames before model sampling, so its results
can differ from historical runs that dropped frames after sampling.
