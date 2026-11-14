# Trainers

The [training entrypoint](../train.py) reads the detector and path configuration,
prepares seed/distributed settings, selects a trainer from `train_type`, and calls
`run()`. Trainer construction creates logging, model, data loaders, optimizers,
and optional resume state.

| `train_type` | Class | Optimization |
|---|---|---|
| `basic` | [BasicTrainer](basic_trainer.py) | One optimizer for `losses['overall']` |
| `alternative` | [AlternativeTrainer](alternative_trainer.py) | Named stages with separate parameters and optimizers |

Both trainers use [choose_dataset](../datasets/builder.py), seeded data loaders,
and optional distributed sampling. Training drops incomplete final batches.
The loaders enable `persistent_workers=True`, which requires positive
`num_workers`. DDP uses NCCL and torchrun's `LOCAL_RANK`. The non-distributed
entrypoint selects CUDA when available; individual detector and loss constructors
can still require CUDA or pretrained assets.

## Optimization and validation

Basic training calls the detector and `get_losses`, then updates its optimizer
from `overall`.

Alternating training builds a cycle from `train_stages`, repeating each stage
name by its `update_step` value, defaulting to one. Each batch updates the selected
stage's optimizer. The cycle restarts each epoch. The detector
provides `get_stage_parameters`, stage-aware forward calls, and stage-aware losses.

Both trainers require an explicit `interval` for each enabled scheduler:
`step` advances after its optimizer updates, while `epoch` advances once at the
end of the training epoch. Basic training reads the top-level `scheduler`;
alternating training reads `train_stages[].scheduler`. Step-based custom schedules
use the associated optimizer's total planned update count, including incomplete
stage cycles at each epoch boundary. Epoch schedules use `num_epochs`. See the
[scheduler guide](../schedulers/README.md) for configuration and parameter units.

Each alternating step clears model gradients before calling
`model(data_dict, stage_name=stage_name)` and
`get_losses(data_dict, predictions, stage=stage_name)`. The detector controls
gradient flow between stages. For AVFF adversarial pretraining, validation omits
the stage and records the generator objective.

`val_every_epoch` selects validation epochs; `val_per_epoch` selects evenly spaced
validation points within them. Validation uses inference mode and
[MetricBuilder](../metrics/builder.py), including its unweighted dataset average.
Trainer prediction records omit demographic annotations, so fairness evaluation
uses test predictions with the required metadata. `record_per_epoch` controls
training loss-log points.

## Outputs and resume

Rank zero writes model-only state dictionaries under `checkpoints/`, including
best-metric files and `epochX_stepY.pth` files at validation points. At every epoch
boundary it writes `training_states/epochX_full_state.pth` with the model,
epoch, optimizer, and scheduler state. Alternating states use stage-specific
optimizer/scheduler entries. Epoch schedulers advance before these full states
are saved. [test.py](../test.py) loads the model-only files.

The entrypoint prepares the run directory before constructing either trainer.
`--log_dir` selects that exact directory. Without `--resume`, existing contents
are removed before a fresh run; without `--log_dir`, the entrypoint generates
`log_dir/train/{model_name}_{timestamp}` under the configured logging root.
Only rank zero prepares files in distributed training, and all ranks receive
the same directory. Source and input-data directories are rejected as run paths.

`--resume` is a boolean flag: it preserves the directory selected by `--log_dir`,
restores its highest numbered full training state, and starts at the next epoch.
Missing state files are logged and training starts from scratch without clearing
existing files. Resume does not restore an in-epoch cursor or random-number
generator state. Retain the original scheduler configuration and training-length
settings when resuming.

```bash
python train.py \
  --detector_path configs/detectors/audio_video/standard/baseline_train_on_megammdf.yaml \
  --log_dir logs/train/baseline_megammdf

python train.py \
  --detector_path configs/detectors/audio_video/standard/baseline_train_on_megammdf.yaml \
  --log_dir logs/train/baseline_megammdf --resume
```

For AVFF adversarial pretraining, use its alternating-stage configuration:

```bash
python train.py --detector_path configs/detectors/audio_video/pretraining/avff_pretrain_adv.yaml
```

Run commands from the repository root with the configured data and dependencies.
Both trainers use the prepared `path_config.log_dir` without appending another
directory, and read `config.resume` as a boolean. A new optimization loop needs
its own trainer and a selection branch in `train.py`. New detectors using the existing loops
implement the [detector interface](../detectors/README.md).
