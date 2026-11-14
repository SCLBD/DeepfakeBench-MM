# Shared utilities

These modules support training, detector construction, and evaluation.

| Module | Purpose |
|---|---|
| [registry.py](registry.py) | `BACKBONE`, `DETECTOR`, and `LOSSFUNC` component registration and lookup |
| [init_seed.py](init_seed.py) | Python/NumPy/PyTorch seeds, DataLoader worker seeds, and seeded generators |
| [logger.py](logger.py) | Process-global console/file logger with rank-zero output |
| [recorder.py](recorder.py) | Weighted running averages and distributed sum/count synchronization |
| [dict_utils.py](dict_utils.py) | Recursive in-place dictionary merging |
| [add_demographic_groups.py](../commands/fairness/add_demographic_groups.py) | Copy supplied directory labels or explicit group mappings into prepared dataset JSON before fairness inference |

## Component registration

`Registry.register_module(module_name=None)` decorates a component class. The
explicit name, or the class name when omitted, becomes its configuration lookup
key. For example, detectors use `@DETECTOR.register_module(module_name='...')`
and are constructed through `DETECTOR[config['model_name']]`.

The registry imports the backbone, detector, and loss packages after creating the
registry objects. Import each new implementation in its package initializer so
its decorator runs. Reusing a name replaces its existing entry; a missing lookup
raises `KeyError`.

## Seeding, logging, and recording

`set_seed` seeds the supported generators and optionally enables deterministic
backend settings. Trainers use `seed_worker` and `build_generator` for data
loading. Results can still depend on the hardware and dependency versions.

`get_logger` creates one logger on its first call. Rank zero writes to the console
and optionally `log_dir/log.txt`; later calls reuse those handlers and directory.

`Recorder.update(item, num)` accumulates `item * num` and its weight, skipping
`None` values. `average()` returns `None` before any weight is recorded.
`sync(device)` reduces sum and count when a process group is initialized.

## Dataset demographic annotations

`add_demographic_groups.py` annotates a selected manifest split while retaining
the other record fields and their order. Its built-in Mega-MMDF and FakeAVCeleb
rules read existing dataset directory annotations; other datasets can provide an
explicit path-to-group mapping. Existing group labels are retained unless
`--overwrite` is requested. In-place writes retain a first `.bak` backup.

Run it before inference so the loader can include `demographic_group` in saved
predictions. The [fairness commands](../commands/fairness/README.md) describe input
annotations, mappings, and the evaluation workflow.
