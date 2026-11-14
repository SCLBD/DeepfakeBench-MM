# Audio-video detector configurations

| Settings | Contents |
|---|---|
| [`standard/`](standard/) | Standard training and test configurations |
| [`pretraining/`](pretraining/) | AVFF adversarial pretraining and MCL teacher pretraining |
| [`cross_pipeline/`](cross_pipeline/README.md) | FGMDF, FRADE, and FauForensics configurations by training pipeline |
| [`small_data_scale/`](small_data_scale/README.md) | Reduced Mega-MMDF training data configurations |
| [`misaligned_test/`](misaligned_test/README.md) | Real-only correspondence controls |

Use the [shared fields](../README.md) and a YAML matching your detector. The
[dataset implementation](../../../datasets/audio_video/audio_video_dataset.py)
and [dataset builder](../../../datasets/builder.py) define the fields below.

## Shared `data_config` fields

| Fields | Meaning |
|---|---|
| `frame_size`, `num_frames`, `stride` | Square frame size, number of sampled frames, and sampling stride. A window spans `num_frames * stride` source frames. |
| `color_mode`, `with_landmark` | `RGB` (default) or `Gray` frames, and whether to load landmarks |
| `frame_mean`, `frame_std` | Video normalization (defaults 0 and 1) |
| `audio_sampling_rate`, `remove_dc_offset` | Sampling rate (default 16000 Hz) and optional waveform centering |
| `spectrogram_type`, `spectrogram_backend`, `spectrogram_params` | Audio representation and backend options. Leave `spectrogram_type` unset for a waveform. See [supported combinations](../../../datasets/utils/data_helpers/audio_helpers.py). |
| `audio_mean`, `audio_std` | Spectrogram normalization (defaults 0 and 1) |
| `max_samples_per_data` | Maximum windows per training or validation record (default 2) |
| `data_directory` | Prepared media directory under `preprocessed_dir` (default `audio-video`). Split manifests remain under `json_dir/audio-video`. |

Keep the supplied video and audio representations for each method. For training,
`aug_config` accepts `video`, `audio`, and `audiovisual` overrides. See the
[augmentation fields](../../../datasets/utils/augmentations/README.md).

## Dataset variants

| `data_type` | Extra fields and behavior |
|---|---|
| `audio_video` | Standard sampling and input representation |
| `audio_video_real` | Same fields. Keeps only windows from real source records. |
| `audio_video_mask_modality` | `aug_config.audiovisual.mask_modality.type` selects `audio`, `video`, or `all`, and `prob` sets the probability (default 0.5). `all` draws the two masks independently. Validation stays clean. |
| `audio_video_misaligned_real` | Test only. Set `data_config.misalignment.modality` to `audio` or `video` and `shift` to an integer offset in complete windows within one clip. Defaults are `audio` and 1. |

With `audio_video`, the builder selects the mask loader automatically when
`aug_config.audiovisual.mask_modality` is set for training or a modality mask is
requested at test time. The test-time mask comes from the CLI. An offline
robustness condition selects a separate prepared media directory. The misaligned
loader evaluates records with real audio and real video, and accepts no additional
robustness condition. The legacy `data_config.only_real_data` switch selects
real-only data for standard training and validation. New real-only configurations
use `data_type: audio_video_real`.
