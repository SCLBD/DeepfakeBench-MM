# Data helpers

Shared helpers read prepared media, convert audio features, normalize model inputs and check paired files. Dataset classes select records and sampling windows. The functions currently implemented are listed below; their docstrings describe arguments and array conventions.

## [video_helpers.py](video_helpers.py)

- `load_video_frames_fixed_stride`: read a video window with fixed-stride sampling, or supply black frames when video is masked.
- `load_landmarks_fixed_stride`: read landmarks using the same frame window and stride.
- `video_normalization`: scale pixels and apply the configured normalization statistics.
- `video_denormalization`: restore normalized video to pixels for display or saving.
- `video_to_rgb_frames`: prepare display-ready video as RGB frames, including grayscale expansion.

## [audio_helpers.py](audio_helpers.py)

- `load_audio_segment`: read an audio window with optional temporal shift, padding, DC removal or modality masking.
- `convert_audio_spectrogram`: compute STFT, Mel, Log-Mel or MFCC features with the supported backend for that representation.
- `spectrogram_normalization`: apply the configured feature normalization statistics.
- `spectrogram_denormalization`: restore features using their original normalization statistics.

## [audio_video_helpers.py](audio_video_helpers.py)

- `validate_audio_video_files`: check that every referenced video, paired waveform and optionally landmark file exists in the selected data directory.

## [image_helpers.py](image_helpers.py)

`Todo:` placeholder; image helpers are not implemented yet.

Each implemented file includes a small `__main__` example. Adjust its local example paths before running it from the repository root. For visualization and MP4 export, see [data presenters](../data_presenters/README.md).
