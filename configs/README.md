# Configurations

[`path.yaml`](path.yaml) defines the roots for raw data, prepared media, metadata,
and logs. [`detectors/`](detectors/README.md) contains detector YAML files organized
by modality and setting.

```text
configs/
├── path.yaml
└── detectors/
    └── <modality>/
        ├── README.md
        └── <setting>/
            └── <configuration>.yaml
```
