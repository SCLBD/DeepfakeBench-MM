# Reduced-data settings

These configurations cover 1%, 2%, 5%, 10%, 20%, and 50% of Mega-MMDF for seven
detectors. Each `{model}_train_on_megammdf_{percentage}.yaml` selects the matching
`Mega-MMDF-{percentage}%` manifest for training and for the Mega-MMDF part of
validation. The other four validation datasets remain configured. The
[`standard/`](../standard/) YAML files provide the full-data settings.

[Prepare these manifests](../../../../commands/data_scale/README.md) before training.
