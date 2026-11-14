# Datasets

This package provides data loading and related data operations, including input conversion, presentation, training augmentation and offline perturbation.

Dataset classes follow the inheritance chain `AbstractDataset` → modality Dataset → custom Dataset. [AbstractDataset](abstract_dataset.py) defines shared initialization and the Dataset interface. Each modality Dataset supplies its sampling and loading behavior; custom Dataset classes inherit from it to adapt the operations they need.

Image, video-only and audio-only Dataset classes are planned. Audio-video Dataset classes are implemented.

## Organization

| Location | Responsibility |
|---|---|
| [builder.py](builder.py) | Construct a registered Dataset class from configuration |
| [audio_video](audio_video/README.md) | Audio-video loading and custom Dataset classes |
| [utils](utils/README.md) | Shared data operations and utility overview |
| [utils/data_helpers](utils/data_helpers/README.md) | Read inputs, convert representations and check paired files |
| [utils/data_presenters](utils/data_presenters/README.md) | Visualize inputs and export playable media |
| [utils/augmentations](utils/augmentations/README.md) | Training transforms |
| [utils/perturbations](utils/perturbations/README.md) | Generate offline perturbation versions of prepared data |

## Construction entry point

```python
from datasets.builder import choose_dataset

dataset = choose_dataset(
    data_type='audio_video',
    data_config=data_config,       # Detector input/sampling settings.
    mode='test',                   # train, val or test.
    dataset_name='Mega-MMDF',      # Use dataset_list in training.
    aug_config=None,
    robustness_config=None,
)
sample = dataset[0]
```

The builder selects the registered Dataset class. Training takes `dataset_list`, and validation/testing take one `dataset_name`. See the [audio-video guide](audio_video/README.md) for data configuration and returned fields.

## Extension conventions

Add a custom Dataset class under its modality directory, inherit from that modality's Dataset and override the necessary operations. Register each new Dataset class in [builder.py](builder.py) and document its configuration and returned fields in the modality guide.

For another source dataset that already follows the standardized prepared format, supply its metadata without adding a new Dataset class. Place shared data operations in the appropriate utility directory listed above.
