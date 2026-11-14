# Running experiments

Each directory contains its own instructions and any preparation or batch scripts.

| Experiment | Purpose |
| --- | --- |
| [train](train/README.md) | Train a detector and resume from a saved state. |
| [test](test/README.md) | Evaluate a checkpoint or export predictions and features. |
| [data_scale](data_scale/README.md) | Prepare Mega-MMDF data subsets and compare training scales. |
| [cross_pipeline](cross_pipeline/README.md) | Prepare pipeline manifests and evaluate transfer between forgery pipelines. |
| [mask_modality](mask_modality/README.md) | Evaluate with audio or video masked. |
| [misaligned](misaligned/README.md) | Evaluate changes under real-only audiovisual window misalignment. |
| [robustness](robustness/README.md) | Generate offline perturbations and evaluate robustness. |
| [fairness](fairness/README.md) | Add demographic group annotations and evaluate disparity across groups. |
