# Detector configurations

Each detector YAML is a complete configuration. Shared filesystem paths come from
[`path.yaml`](../path.yaml). The modality guides describe their input fields,
such as the [audio-video settings](audio_video/README.md).

## Model

`model_name` selects a registered detector. The detector constructor receives
`model_config`, which defines the model parameters, initialization and losses
expected by that detector. Use the supplied YAML for the method and check its
constructor in the [detector package](../../detectors/README.md). Individual
detectors define how they use any configured loss weights.

## Data

Set `data_type` to select a dataset class through the
[dataset builder](../../datasets/builder.py). Set `data_config` according to that
class’s input and sampling interface. The supplied configurations show the
representation expected by each model. `num_workers` sets the number of workers
per loader (default 4). `aug_config` controls training augmentation and accepts
`default`, a mapping of overrides, or an empty mapping or `null` to disable it.

## Training

| Field | Meaning |
|---|---|
| `train_type` | `basic` for a single optimizer or `alternative` for alternating stages |
| `train_datasets`, `train_batch_size` | Prepared training manifest names and batch size |
| `val_datasets`, `val_batch_size` | Prepared validation manifest names and batch size. Omit or leave `val_datasets` empty when validation is unnecessary. `val_batch_size` is also the default test batch size. |
| `metric_scoring` | Metric used for validation, such as `sample-level AUC` or `loss` |
| `val_every_epoch`, `val_per_epoch` | Epoch interval and number of validation points within eligible epochs (default 1 each) |
| `num_epochs`, `record_per_epoch` | Total training epochs and loss-recording points each epoch (default 10 recording points) |
| `optimizer` | Optimizer `type` and `params` for basic training |
| `scheduler` | Scheduler `type`, `params` and `interval` for basic training, or `null` to disable it |
| `train_stages` | For alternating training, a list of `name`, `update_step` (default 1), `optimizer`, and optional `scheduler` entries |

Use `interval: step` to advance a scheduler after optimizer updates or
`interval: epoch` to advance it after completed epochs. In alternating training,
each stage uses its own optimizer and scheduler. Pick a stage name supported by
the detector. See the [training](../../trainers/README.md),
[optimizer](../../optimizers/README.md), and [scheduler](../../schedulers/README.md)
interfaces for their options. Test-only configurations need the model, data,
loader settings, and `val_batch_size` as a test fallback.
