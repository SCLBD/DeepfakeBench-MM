# Fixed FakeAVCeleb splits

The preprocessor reads `train_meta_data.csv`, `val_meta_data.csv` and
`test_meta_data.csv` directly from this directory by default. No copying into
`original_dir/FakeAVCeleb_v1.2/` is required, and no new random split is generated.
Keep the raw media under `original_dir/FakeAVCeleb_v1.2/` in the category and
identity directories named in the CSVs.

```bash
python preprocess/audio_video/fakeavceleb_preprocessor.py --split all --num_proc 8
```

The supplied CSVs retain the benchmark's existing approximately 60%/20%/20%
split. Their bytes, row order, labels and forgery-method parsing are unchanged.
The preprocessor retains its original duplicate-record handling; CSV row counts
therefore need not equal the number of unique raw clips or successfully
preprocessed clips.

To use a separate explicit protocol, point `--split_dir` to another directory
containing all requested `{train,val,test}_meta_data.csv` files:

```bash
python preprocess/audio_video/fakeavceleb_preprocessor.py --split all --num_proc 8 --split_dir /path/to/splits
```

FakeAVCeleb, [PolyGlotFake](../polyglotfake_splits/README.md) and
[DigiFakeAV](../digifakeav_splits/README.md) use the same lookup convention:
bundled split files by default, an explicit `--split_dir` for an alternative.
