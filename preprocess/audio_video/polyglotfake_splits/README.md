# Fixed PolyGlotFake splits

The preprocessor reads `train.json`, `val.json` and `test.json` from this directory
by default. Each file is an ordered list of raw MP4 paths relative to
`original_dir`, including the `PolyGlotFake/` prefix. No copying into the raw
dataset directory and no runtime random split are required.

```bash
python preprocess/audio_video/polyglotfake_preprocessor.py --split all --num_proc 8
```

The original real/fake annotation JSONs are still required to obtain forgery
methods and real-source relationships. Their array order no longer affects
split membership or processing order. Missing listed annotations, overlapping
split paths, or fake videos assigned apart from their real source raise errors.
Use `--split_dir /path/to/splits` only for an explicitly different protocol.

## Origin and counts

The lists freeze the original benchmark rule: shuffle the 766 real annotation
entries once with `random.Random(42)`, allocate 80%/10%/the remainder to
train/val/test, and assign every fake to its `raw_video_name` source's split.
This is a source-clip grouping rule, not a claim of identity-disjoint splits.
Two exact duplicate fake records are represented only once.

| Split | Raw real | Raw fake | Raw total | Retained in the experiment manifest |
|---|---:|---:|---:|---:|
| train | 612 | 11,486 | 12,098 | 11,355 |
| val | 76 | 1,458 | 1,534 | 1,412 |
| test | 78 | 1,526 | 1,604 | 1,494 |
| total | 766 | 14,470 | 15,236 | 14,261 |

The lists were checked against all 14,261 records in the experiment's prepared
`PolyGlotFake.json`: every split assignment matches, and filtering the raw lists
to those retained records reproduces their order exactly. No split contains a
path from another split, and related real/fake clips stay in the same split.

Raw lists and retained-media counts serve different purposes: 975 listed raw
paths are absent from the prepared experiment manifest. Media preprocessing
still keeps only completed items, so fixed raw membership alone does not promise
identical retained counts across different decoding or face-detection setups.

[split_metadata.json](split_metadata.json) records source hashes, split-file
hashes, counts and verification results without machine-specific storage roots.
