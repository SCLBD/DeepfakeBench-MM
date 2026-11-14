# Backbones

This package contains reusable backbone implementations. The benchmark makes
small adaptations to expose the inputs, features, and checkpoint-loading
interfaces used by its detectors. See [the detector guide](../detectors/audio_video/README.md)
for method-specific networks defined with their detector.

## Structure

| Directory | Backbones |
|---|---|
| [audio_backbones/](audio_backbones/) | [AudioMAE](audio_backbones/audiomae_backbone.py), [Cnn14](audio_backbones/Cnn14_backbone.py) |
| [video_backbones/](video_backbones/) | [MARLIN](video_backbones/marlin_backbone.py), [ViViT](video_backbones/ViViT_backbone.py) |
| [audio_video_backbones/](audio_video_backbones/) | [AV-HuBERT](audio_video_backbones/av_hubert/) |
| [image_backbones/](image_backbones/) | Planned |

## Imports

Use `BACKBONE['AudioMAE']` and `BACKBONE['MARLIN']` for registered construction.
The [package initializer](__init__.py) imports those classes to populate
the [backbone registry](../tools/registry.py). MCL imports Cnn14 and ViViT directly.

See `__init__`, `forward`, and the feature or reconstruction methods in each
linked implementation for its inputs and outputs. AudioMAE and MARLIN also
define `predefined_versions` on their classes. The [pretrained weights guide](../pretrained/README.md)
describes any required initialization files.

## Dependencies

[install.sh](../install.sh) installs the Python environment and dependencies.
AV-HuBERT uses the Fairseq and AV-HuBERT packages included under
`audio_video_backbones/av_hubert/`. AVH and SpeechForensics add this directory to
the import path and import the bundled packages when restoring checkpoints.
Keep both packages and their resources together.

## Extension

Place a new backbone in the directory for its modality. Import the class
directly where the detector uses it. For configuration-based construction,
register it with `@BACKBONE.register_module(module_name='YourBackbone')` and import
it in the package initializer. Document its constructor, input and output
shapes, and any initialization weights.
