# Learning-rate schedulers

[choose_scheduler](builder.py) builds a scheduler from an optimizer, optional
configuration, and total scheduler-call count. A `None` configuration returns
`None`.

## Configuration

Every enabled scheduler requires `type`, `interval`, and `params`:

```yaml
scheduler:
  type: step
  interval: epoch
  params:
    step_size: 8
    gamma: 0.5
```

`type` selects the algorithm. `interval` controls when the trainer calls
`scheduler.step()`; it is not passed to the scheduler constructor. `params`
contains the constructor arguments. Missing or unsupported intervals raise
`ValueError`; use `scheduler: null` to disable scheduling.

| `interval` | Advancement |
|---|---|
| `step` | Once after the associated optimizer actually updates |
| `epoch` | Once after a complete training epoch, before saving the full training state |

| Case-sensitive `type` | Implementation | Configuration |
|---|---|---|
| `step` | `torch.optim.lr_scheduler.StepLR` | Constructor arguments in `params` |
| `cosine` | `torch.optim.lr_scheduler.CosineAnnealingLR` | Constructor arguments in `params` |
| `FauForensics_Customized` | [fauforensics_scheduler](customized_schedulers/fauforensics_scheduler.py) | `params.power`, `params.lr_decay`; builder supplies `total_iters` |

The `step` algorithm name denotes StepLR's staircase decay, independently of the
advancement interval. The example decays the learning rate every eight epochs.
Parameters such as `step_size` and `T_max` use the selected interval's units; the
trainer does not rescale them. Supplied FGMDF configurations use epoch-based
StepLR, while FauForensics configurations use its custom schedule per optimizer
update.

## Trainer integration

Basic training reads the top-level `scheduler`. Alternating training reads
`train_stages[].scheduler`, alongside each stage's optimizer. Each stage advances
only its own scheduler after an optimizer update when `interval: step`; epoch
schedulers advance at the end of the epoch.

For the custom schedule, `total_iters` is the total number of planned scheduler
calls: `num_epochs * len(train_data_loader)` for basic step scheduling, or
`num_epochs` for epoch scheduling. Alternating step schedules instead use each
stage's actual update count. The stage cycle restarts each epoch, so the trainer
counts both complete cycles and the remaining batches in that epoch. It does
not divide the total batch count by the number of stages.

The custom multiplier is
`max(0, 1 - step / total_iters) ** power * (1 - lr_decay) + lr_decay` and requires
positive `total_iters`. The trainers save and restore scheduler state together
with optimizer state in full training checkpoints. Resume with the same
scheduler algorithm, interval, parameters, and training-length settings.

## Adding a scheduler

Import the implementation and add a branch in `choose_scheduler`. Its author
must choose the appropriate `interval` and document the units of its parameters.
The trainer follows this explicit configuration rather than inferring the
interval from a scheduler's name. Schedulers requiring validation metrics need
additional trainer integration; the current interface supports `step()` without
metric arguments.
