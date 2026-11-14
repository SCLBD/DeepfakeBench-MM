# Fixed DigiFakeAV splits

The preprocessor reads the supplied `train.json`, `val.json` and `test.json`
directly from this directory. Each entry is a raw MP4 path relative to
`original_dir`, with a `DigiFakeAV/` prefix. The files do not need to be copied
into the dataset directory.

```bash
python preprocess/audio_video/digifakeav_preprocessor.py --split all --num_proc 8
```

| Split | Real | Fake video / real audio | Fake video / fake audio | Total |
|---|---:|---:|---:|---:|
| train | 8,000 | 20,000 | 20,000 | 48,000 |
| val | 1,000 | 2,500 | 2,500 | 6,000 |
| test | 1,000 | 2,500 | 2,500 | 6,000 |

The supplied lists contain 60,000 distinct paths, with no duplicate path within
a split and no path overlap between splits. These checks do not establish
identity or shared-source disjointness. The supplied split files are retained
byte-for-byte; only the preprocessor's default lookup location was changed.

Use `--split_dir /path/to/splits` for a separate explicit split directory.

This matches the lookup convention for
[FakeAVCeleb](../fakeavceleb_splits/README.md) and
[PolyGlotFake](../polyglotfake_splits/README.md): the default uses the bundled
split files directly, and `--split_dir` selects an explicit alternative. Raw
media still reside under `original_dir`; only split metadata are bundled.
