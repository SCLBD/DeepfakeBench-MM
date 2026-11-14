# Audio-video datasets

Load fixed-length audio-video samples from the shared preprocessing format. [AudioVideoDataset](audio_video_dataset.py) supports training, validation and testing through the common dataset builder.

## Quick start

**1. Prepare the data.** Follow the [audio-video preprocessing guide](../../preprocess/audio_video/README.md), then configure [configs/path.yaml](../../configs/path.yaml):

| Setting | Required contents |
|---|---|
| `preprocessed_dir` | Prepared media under `audio-video/<dataset>/...` |
| `json_dir` | Split metadata at `audio-video/<dataset_name>.json` |

The JSON contains `train`, `val` and `test` lists. Each record supplies `path`, `label`, `audio_label` and `video_label`. `path` is relative to `preprocessed_dir/audio-video/`. For example, `ExampleDataset/clip/2_102.mp4` uses audio from `ExampleDataset/clip.wav` and, if requested, landmarks from `ExampleDataset/clip/2_102.npy`. The preprocessing guide includes the full output layout and JSON example.

**2. Use the detector's data configuration.** Run from the repository root with the project dependencies installed:

```python
import yaml
from datasets.builder import choose_dataset

config_path = "configs/detectors/audio_video/standard/baseline_train_on_megammdf.yaml"
with open(config_path, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

dataset = choose_dataset(
    data_type=config["data_type"],
    data_config=config["data_config"],
    mode="test",
    dataset_name="Mega-MMDF",
)
sample = dataset[0]
print(sample["video"].shape, sample["audio"].shape, sample["label"])
```

For training, use `mode="train"`, replace `dataset_name` with `dataset_list`, and pass the YAML's `aug_config`. Use `mode="val"` with a single `dataset_name` for validation. A PyTorch `DataLoader` can batch the returned samples.

## Configuration

Use the selected [detector YAML](../../configs/detectors/audio_video/README.md) to keep its input representation compatible with the model. The main `data_config` fields are:

| Field | Purpose |
|---|---|
| `num_frames`, `stride` | Number of video frames and sampling stride; required |
| `frame_size` | Output frame height/width; required |
| `frame_mean`, `frame_std` | Video normalization; defaults: 0 and 1 |
| `color_mode` | `RGB` (default) or `Gray` |
| `audio_sampling_rate` | Audio rate; default: 16000 Hz |
| `spectrogram_type`, `spectrogram_backend`, `spectrogram_params` | Audio representation; omit the type for waveform input |
| `audio_mean`, `audio_std` | Spectrogram normalization; defaults: 0 and 1 |
| `data_directory` | Optional alternate folder under `preprocessed_dir`; defaults to `audio-video` |
| `with_landmark` | Return aligned landmarks; default: false |
| `max_samples_per_data` | Maximum windows per prepared record in train/val; default: 2 |

One sample spans `num_frames * stride` source frames at 25 FPS. Training samples random windows within the selected portions of each record; validation uses fixed starts. Testing uses all complete, non-overlapping windows and drops the incomplete tail, without the train/val window cap.

`aug_config` is a separate builder argument: `"default"` enables the loader's default training augmentation; `{}` or `null` disables it; a nonempty mapping overrides the specified defaults. Training augmentation is not applied in validation or testing.

## Returned data

`dataset[index]` returns a dictionary. Shapes below describe one sample before batching; `T = num_frames`, `H = W = frame_size`, `L` is waveform length, and `Ta` and `F` are audio feature time steps and bins.

| Field | Format / meaning |
|---|---|
| `video` | Float32 tensor `(C, T, H, W)`, with C=3 for RGB or C=1 for Gray |
| `audio` | Float32 waveform `(L,)` or spectrogram/features `(Ta, F)` |
| `frame_label` | Long tensor `(T,)` of sampled video-frame labels |
| `audio_label`, `video_label` | Scalar modality labels for this window |
| `label` | Scalar overall label: fake if either modality is fake |
| `path` | Relative prepared video path |
| `sample_frame_range` | Source-frame interval string `start_end`, with exclusive end |
| `landmarks` | Optional float32 tensor `(T, 68, 2)` |
| `demographic_group` | Optional annotation carried from metadata |

Labels use real=0/fake=1 and account for the sampled window's temporal forgery annotations. Default collation adds a batch dimension to tensors and numeric labels.

Offline perturbations reuse the same loader and split JSON with another `data_directory`. Generate the files first using the [perturbation runner](../utils/perturbations/README.md). Missing paired files are reported before inference.

Supporting data-processing tools are described in the [utilities guide](../utils/README.md).
