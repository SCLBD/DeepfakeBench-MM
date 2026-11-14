# Detectors

This package connects detector implementations to the benchmark's training and evaluation interfaces. The [audio-video guide](audio_video/README.md) lists the currently implemented methods.

| Module | Purpose |
|---|---|
| [abstract_detector.py](abstract_detector.py) | Interface for one optimization step per batch |
| [abstract_alternative_detector.py](abstract_alternative_detector.py) | Interface for named optimization stages |
| [audio_video/](audio_video/) | Audio-video detector adapters and their method-specific networks |
| [audio/](audio/), [image/](image/), [video/](video/) | Planned unimodal detector implementations |
| [__init__.py](__init__.py) | Imports implementations to populate the detector registry |

## Construction and use

[tools.registry](../tools/registry.py) creates `DETECTOR`, `BACKBONE` and `LOSSFUNC` before importing their implementation packages. Construct a detector with the exact registered `model_name`:

```python
model = DETECTOR[config['model_name']](config['model_config'])
predictions = model(data_dict)
losses = model.get_losses(data_dict, predictions)
```

The ordinary trainer optimizes `losses['overall']` and records additional loss terms. Classification evaluation calls `model(data_dict, inference=True)` and reads the batch vector `predictions['prob']` as the fake score. Other prediction keys and the `features`/`classifier` helper APIs depend on the method. Pretraining adapters can return reconstruction outputs instead of classification scores.

`AbstractAlternativeDetector` also declares `get_stage_parameters(stage_name)`. The alternative trainer selects parameter groups by name, calls `model(data_dict, stage_name=stage_name)`, and obtains losses with `get_losses(data_dict, predictions, stage=stage_name)`. Validation omits the stage; the detector defines its validation objective. AVFF adversarial pretraining uses the generator objective in that case.

Evaluation normally restores a model state dictionary with strict `load_state_dict`. An adapter can implement `load_checkpoint(weight_path)` when the checkpoint must restore its own architecture and task configuration. SpeechForensics uses this hook for a full Fairseq checkpoint and requires `inference=True`.

## Adding a detector

1. Implement the appropriate abstract interface and document required batch fields, outputs and train/evaluation behavior.
2. Register a unique name with `@DETECTOR.register_module(module_name='YourDetector')`, then import the class from this package's initializer.
3. Supply matching model, data and loss configuration. The detector chooses which authenticity labels each loss consumes.
4. Return `overall` for optimization and `prob` for classification evaluation. Expose feature outputs explicitly when they are needed by `test.py`.
5. Document initialization checkpoints separately from the final evaluation checkpoint.

See the [backbone guide](../backbones/README.md) and [loss guide](../losses/README.md) for reusable components.
