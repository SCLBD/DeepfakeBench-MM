# Real-only misalignment settings

These test configurations use `data_type: audio_video_misaligned_real`. Within
`data_config.misalignment`, `modality` chooses audio or video to shift, and `shift`
sets a signed circular offset in complete windows from the same clip. The default
is audio and 1. `shift: 0` provides an aligned control. Single-window clips stay
aligned. The [audio-video reference](../README.md) describes the shared fields.
