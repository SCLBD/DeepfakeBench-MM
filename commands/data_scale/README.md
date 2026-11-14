# Reduced-data training and evaluation

The 42 configurations cover seven detectors at 1%, 2%, 5%, 10%, 20%, and 50%
of Mega-MMDF for the data scale experiment in Figure 3. Run the commands from
the repository root.

## Prepare the subsets

Configure [data paths](../../configs/path.yaml) and [preprocess Mega-MMDF](../../preprocess/README.md)
to create `json_dir/audio-video/Mega-MMDF.json`. The subset script reads the
prepared manifest and the original `original_dir/Mega-MMDF/pair.json`:

```bash
python commands/data_scale/prepare_small_scale.py
```

Use `--proportion 1 2` to generate only those two percentages. Supply
`--pair-json /path/to/pair.json` if the original file is elsewhere. Existing
pair lists can be used with `--pair-lists /path/to/pair_lists`, containing
`<split>_<skin>_<gender>.txt` files with one pair of clip IDs per line.

The script matches each pair against prepared real clips in the same split. It
samples whole pairs by demographic group and retains all matching records from
pipelines 0–21. Pipeline 22 does not use source pairs, so its records are sampled
by split and demographic group at the same rate. A fixed seed produces nested
subsets. The default seed 0 writes `Mega-MMDF-<percentage>%.json`; positive
`--seed` values add `-<seed>` to the filename and require updating the dataset
names in the configuration. Use `--overwrite` to replace existing manifests.

## Train and evaluate

Prepare all validation datasets selected in the [reduced-data settings](../../configs/detectors/audio_video/small_data_scale/README.md).
The [training commands](small_scale_train_commands.txt) use boolean `--resume`
to preserve and resume each named run.

```bash
bash commands/data_scale/small_scale_train_commands.txt
```

Remove `--resume` from a command to start fresh and clear its run directory.
Place the selected model-only checkpoints at the paths used by the
[test commands](small_scale_test_commands.txt), then run:

```bash
bash commands/data_scale/small_scale_test_commands.txt
```

The training and Mega-MMDF validation manifests use the selected subset. The
test commands evaluate the full eight test datasets at every scale. Use the
[standard configurations](../../configs/detectors/audio_video/standard/) for
100% training data and the [scaling figure guide](../../analysis/data_scale/README.md)
to plot the results.
