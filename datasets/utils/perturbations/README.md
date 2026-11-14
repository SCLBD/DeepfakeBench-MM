# Offline perturbations

Generate perturbed versions of standardized, preprocessed datasets before evaluation. File operators implement the transformations; the runner selects prepared files from dataset metadata and writes a separate data directory. The current runner supports **audio-video** datasets.

## Implemented files and functions

### [video_perturbations.py](video_perturbations.py)

- `video_h264`: apply an H.264 compression pass.
- `video_downscale`: reduce each frame's spatial resolution and restore its original dimensions.
- `video_frame_drop`: conceal random frame losses by repeating the last received frame.
- `transform_video_file`: apply a custom frame transform and save lossless RGB H.264 while preserving the video timeline.

### [audio_perturbations.py](audio_perturbations.py)

- `audio_aac`: encode/decode through AAC and save the resulting compression artifacts in a WAV.
- `audio_dropout`: replace random audio packets with silence and fade their boundaries.
- `read_audio_file`: decode prepared mono audio without resampling.
- `write_audio_file`: save processed audio as a lossless floating-point WAV.

### [audio_video_perturbations.py](audio_video_perturbations.py)

- `video_delay`: shift the waveform relative to the unchanged video timeline.
- `get_perturbation_directory`: derive the shared output-directory name from an operation and its actual parameter.

`AUDIO_VIDEO_OPERATIONS` lists the six supported operations, their affected file kinds and parameter parsers. The generator and evaluation loader use this same mapping.

### [common_perturbations.py](common_perturbations.py)

- `numeric_param`: validate and normalize a finite numeric parameter.
- `crf_param`: validate an H.264 CRF.
- `ratio_param`: validate a spatial downscaling ratio.
- `probability_param`: validate a frame-loss or packet-loss probability.
- `bitrate_param`: validate an explicit audio bitrate.
- `run_ffmpeg`: run FFmpeg and report encoder failures.
- `atomic_output`: publish a final file only after its temporary output completes successfully.
- `stable_file_seed`: derive reproducible randomness from a base seed and relative file path.
- `copy_file`: copy an unaffected file through atomic output.

### [run_perturbation_on_audio_video_datasets.py](run_perturbation_on_audio_video_datasets.py)

- `parse_args`: parse the operation, actual parameter and generation options.
- `resolve_root`: resolve configured storage paths relative to the repository root.
- `collect_split_files`: collect unique video, paired audio and available landmark files for selected dataset splits.
- `generate`: transform affected files, copy the others, skip existing outputs and report file counts.
- `main`: run the command-line generator.

[image_perturbations.py](image_perturbations.py) and the [image](run_perturbation_on_image_datasets.py), [audio-only](run_perturbation_on_audio_datasets.py) and [video-only](run_perturbation_on_video_datasets.py) runners are `Todo:` placeholders. Audio/video operators can be reused when those runners are implemented. Function docstrings describe the detailed API contracts.

## Setup and generation

Install the project dependencies and make **FFmpeg and FFprobe** available in `PATH`. Generation uses NumPy, OpenCV and PyYAML; the FFmpeg build needs `libx264`, `libx264rgb` and AAC encoding. Generation does not require a detector checkpoint.

Prepare the standardized data and configure `preprocessed_dir` and `json_dir` in [configs/path.yaml](../../../configs/path.yaml). Run commands from the repository root:

```bash
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_h264 --param 23 --datasets Mega-MMDF
```

This reads `json_dir/audio-video/Mega-MMDF.json`, selects its **test split by default**, and reads clean files from `preprocessed_dir/audio-video/`. Omit `--datasets` to process every dataset JSON in `json_dir/audio-video/`. Each operation starts from the clean prepared data; source files are unchanged.

### Actual parameters

`--param` is the operation's **actual parameter**, rather than a severity level. The examples below are suggested values, not built-in benchmark presets.

| `--type` | Meaning of `--param` | Example values |
|---|---|---|
| `video_h264` | H.264 CRF, from 0 to 51; larger means stronger compression | `23`, `30`, `37` |
| `audio_aac` | AAC bitrate in bits/s or `k`/`M` notation; smaller means stronger compression | `64k`, `32k`, `16k` |
| `video_downscale` | Width/height multiplier in (0, 1]; frames are then restored to their original dimensions | `0.67`, `0.44`, `0.33` |
| `video_frame_drop` | Loss probability per source frame in [0, 1]; the first frame is always retained | `0.05`, `0.1`, `0.2` |
| `audio_dropout` | Loss probability per audio packet in [0, 1] | `0.05`, `0.1`, `0.2` |
| `video_delay` | Signed relative offset in seconds; positive `d` uses clean audio at `t+d` for output time `t`, with silent boundaries | `0.08`, `0.16`, `0.32` |

Values do not imply equal perceptual severity across operations or datasets. Numeric parameters are normalized for directory naming: `0.10` and `1e-1` both produce suffix `_0.1`. Bitrate spellings such as `64k` and `64000` remain distinct directory names.

### Other CLI options

| Option | Default | Purpose |
|---|---|---|
| `--datasets NAME [NAME ...]` | All dataset JSONs | Select dataset names without `.json` |
| `--split {train,val,test}` | `test` | Select the metadata split |
| `--path_config PATH` | Repository `configs/path.yaml` | Use another path configuration |
| `--output_dir PATH` | `preprocessed_dir/audio-video_<type>_<param>` | Override the complete destination root |
| `--seed N` | `0` | Base seed for random operations |
| `--packet_duration_ms MS` | `20` | Packet duration for `audio_dropout` |
| `--fade_duration_ms MS` | `2` | Dropout boundary fade; `0` disables fading |

Relative storage roots in `path.yaml` and relative `--output_dir` values resolve from the repository root. `--path_config` itself follows the shell's normal path interpretation. For example:

```bash
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type audio_dropout --param 0.1 --datasets Mega-MMDF FakeAVCeleb --seed 42 --packet_duration_ms 20 --fade_duration_ms 2
```

## Output directories

The default destination appends `_<type>_<param>` to the clean modality directory:

```text
preprocessed_dir/
├── audio-video/
│   └── Mega-MMDF/.../
│       ├── clip.wav
│       └── clip/
│           ├── 0_100.mp4
│           └── 0_100.npy
└── audio-video_video_h264_23/
    └── Mega-MMDF/.../
        ├── clip.wav
        └── clip/
            ├── 0_100.mp4
            └── 0_100.npy
```

Only files referenced by the selected split are collected. Repeated references and shared WAVs are processed once. The affected stream is transformed; unaffected streams and available matching NPY annotations are copied unchanged. A complete shared WAV is processed or copied even when a selected record uses only part of it. No new split JSON is written.

Random decisions are fixed per prepared file using the base seed and its relative path, before model-specific windowing, sampling or feature conversion. Video downscaling and frame loss preserve dimensions, FPS and frame count, and use lossless RGB H.264 storage. Audio operations preserve the original sample rate and length.

Annotations, authenticity labels and partial-forgery intervals are not recomputed. Account for this when defining an evaluation protocol involving shifted content. Fixed `video_delay` offsets are separate from the model-window reassignment used by [misalignment analysis](../../../analysis/misalignment/README.md).

## Resume and existing files

Rerun the same command after interruption. **Existing final output files are treated as complete and skipped.** Files are written to temporary siblings and published only on success; unfinished temporary files are ignored. File and encoder failures are reported while remaining files continue, then the command exits with an error. A rerun retries missing final files.

Directory names encode only the operation and main parameter. When changing the seed, packet/fade settings or source data, choose another `--output_dir` or remove the affected old outputs first. Existing damaged or outdated final files are also skipped; remove those files to regenerate them.

## Evaluate a generated version

Use [test.py](../../../test.py) with the same type and actual parameter, an existing detector configuration and a prepared checkpoint:

```bash
python test.py --detector_path configs/detectors/audio_video/standard/baseline_train_on_megammdf.yaml \
  --weight_path weights/Baseline.pth --test_datasets Mega-MMDF \
  --robustness_type video_h264 --robustness_param 23 \
  --metrics "sample-level AUC" "video-level AUC" --log_dir logs/test/Baseline_h264_23
```

The loader selects `preprocessed_dir/audio-video_video_h264_23/` and uses the ordinary Dataset. Perturbations are already stored in those files. Omitting `--metrics` saves predictions without computing metrics. Modality masks use `--robustness_type mask_audio` or `mask_video` without a parameter.

For a custom generator `--output_dir`, set `data_config.data_directory` in the detector configuration to the output folder name beneath `preprocessed_dir`, or its absolute path, and omit the robustness flags. Use a new `--log_dir` whenever the data version changes so existing predictions are not reused. Required files are checked before pending inference; complete saved predictions can be reused for metric-only evaluation.

## Add an operation

Implement a documented file-processing function in the appropriate operator module, then add its name, affected file kind, function and actual-parameter parser to `AUDIO_VIDEO_OPERATIONS`. The runner handles traversal, copying unaffected files and skipping existing outputs.
