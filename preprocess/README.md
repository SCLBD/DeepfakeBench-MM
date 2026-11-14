# Preprocessing

This package converts raw dataset annotations/media into the aligned media and shared metadata used by [datasets](../datasets/README.md). The implemented modality is audio-video; its eight dataset adapters are documented in the [audio-video guide](audio_video/README.md).

## Structure

```text
preprocess/
  abstract_preprocessor.py
  audio_video/
    audio_video_preprocessor.py
    <dataset>_preprocessor.py
  image/
    __init__.py
    image_preprocessor.py
  video/
    __init__.py
    video_preprocessor.py
  audio/
    __init__.py
    audio_preprocessor.py
  dlib_tools/
```

The image, video and audio packages contain empty `__init__.py` files and Todo placeholders for their modality preprocessors. Audio-video remains the only implemented modality.

Preprocessors follow a three-level inheritance structure:

```text
AbstractPreprocessor
  └── <Modality>Preprocessor
        └── <DatasetName>Preprocessor
```

- The abstract preprocessor defines the shared interface and lifecycle.
- A modality preprocessor implements media operations and output assembly for that modality.
- A dataset preprocessor inherits from the corresponding modality preprocessor and adapts a dataset's directory layout, annotations and split definitions to its expected input records.

Each dataset entrypoint follows the same lifecycle, shown here with FakeAVCeleb:

```python
preprocessor = FakeAVCelebPreprocessor(split=args.split)
preprocessor.preprocess_all_data(num_proc=args.num_proc)
preprocessor.assemble_to_json()
```

## Paths and split writing

Configure the roots in [configs/path.yaml](../configs/path.yaml) and run commands from the repository root.

| Setting | Contents |
|---|---|
| `original_dir` | Original datasets and their source annotations |
| `preprocessed_dir/<modality>/` | Media and auxiliary files produced by that modality's preprocessor |
| `json_dir/<modality>/` | Assembled metadata, with one `<dataset_name>.json` per dataset |

For example, these roots may be `./data/original`, `./data/preprocessed` and `./data/json`. Dataset-specific input layouts, resources and output examples belong in the corresponding modality guide.

`--split` accepts `all` (default), `train`, `val` or `test`; `--num_proc` sets the worker count (default: 1). Assembly overwrites the dataset JSON with all three split keys. Splits not collected by that invocation are empty, so use `--split all` to build a complete manifest.

In the current pipeline, `.done` and `.abort` mark completed and skipped items. Both are skipped on later runs; assembly includes completed items only. Remove an item's marker/cache intentionally when reprocessing it. Rerunning assembly replaces metadata without reprocessing completed media.

## Extending preprocessing

Add a dataset adapter under its corresponding modality package and inherit from that modality's preprocessor. Implement `collect_data_attributes()` to map the raw layout and annotations to the modality's input records, then expose the same preprocessing and assembly lifecycle.

The same raw dataset can have separate adapters for different modalities. Each adapter can produce the representation required by its detector while keeping processed media and metadata organized under that modality's directories.

Image, video and audio preprocessing are planned within this structure. Shared lifecycle logic remains in the abstract layer, modality-specific operations remain in the modality layer, and raw dataset adaptation remains in the dataset layer.
