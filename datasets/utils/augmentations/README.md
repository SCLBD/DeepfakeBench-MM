# Training augmentations

Build and apply configurable training augmentations. Audio and video are currently supported; Dataset classes determine when augmentation runs. Function docstrings describe the arguments and array conventions.

## [video_augmentations.py](video_augmentations.py)

- `init_video_aug`: build a replayable pipeline for flipping, rotation, Gaussian blur, color changes and image compression, with optional landmark support.
- `apply_video_aug`: apply the same sampled transformation to every frame and apply matching geometric changes to landmarks.

## [audio_augmentations.py](audio_augmentations.py)

- `init_audio_aug`: build a pipeline for Gaussian noise, pitch shifting and volume gain.
- `apply_audio_aug`: apply the initialized pipeline to an audio waveform.

## [image_augmentations.py](image_augmentations.py)

`Todo:` placeholder; image augmentation is not implemented yet.

Both implemented files include a self-contained `__main__` example. Run them from the repository root to save before/after images under `outputs/augmentation_examples/`; the audio example also exports a playable MP4. See [data presenters](../data_presenters/README.md) for the display functions.
