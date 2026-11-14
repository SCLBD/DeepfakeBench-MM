# Data presenters

Create visual inspections of prepared samples. Combined audio-video presentation is currently implemented; function docstrings describe input preparation, display options and timing requirements.

## [audio_video_presenter.py](audio_video_presenter.py)

- `present_audio_video`: compose sampled video frames and a waveform or audio-feature panel into one image, with optional saving and layout metadata.
- `render_audio_video`: alias of `present_audio_video`.
- `render_spectrogram`: render an audio-feature panel with configurable colormap and contrast.
- `render_waveform`: render a waveform panel using a per-pixel amplitude envelope.
- `save_audio_video`: export video and its matching waveform as a playable MP4 using FFmpeg.

`present_audio_video` supports frame sizing, spacing, colors, labels and a shared time axis. Use [data helpers](../data_helpers/README.md) to restore normalized inputs before presentation when needed.

## Other presenter files

[image_presenter.py](image_presenter.py), [video_presenter.py](video_presenter.py) and [audio_presenter.py](audio_presenter.py) are `Todo:` placeholders.

The self-contained example uses NumPy, Pillow and Matplotlib; MP4 export also requires FFmpeg in `PATH`. Run it from the repository root:

```bash
python datasets/utils/data_presenters/audio_video_presenter.py
```

Example images and an MP4 are saved under `outputs/presenter_examples/`.
