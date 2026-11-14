# Cross-pipeline settings

These configurations cover FGMDF, FRADE, and FauForensics on Mega-MMDF
pipelines 1–22. Each `{model}_train_on_pipeline{ID}.yaml` sets `train_datasets`
and `val_datasets` to `Mega-MMDF-pipeline{ID}`. These names refer to existing
prepared split manifests. The audio and video settings follow the detector’s
standard configuration. The [audio-video reference](../README.md) describes the
fields.

[Prepare these manifests](../../../../commands/cross_pipeline/README.md) before training.
