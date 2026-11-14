# Demographic fairness evaluation

Run the annotation script from the repository root after preprocessing and
manifest assembly:

```bash
python commands/fairness/add_demographic_groups.py --dataset Mega-MMDF --split test --in_place
python commands/fairness/add_demographic_groups.py --dataset FakeAVCeleb --split test --in_place
```

The script reads `json_dir/audio-video/<dataset>.json` using
[`configs/path.yaml`](../../configs/path.yaml) and copies the dataset's existing
annotations from `original_path` into `demographic_group`.

| Dataset | Directory labels | Stored group |
| --- | --- | --- |
| Mega-MMDF | `Female_Black`, `Male_White` | `Black - Female`, `White - Male` |
| FakeAVCeleb | `African/men`, `Asian (East)/women` | `African - men`, `Asian (East) - women` |

By default, only the test split is annotated. `--in_place` retains the original
manifest as a `.json.bak` file on the first write. Existing group labels are
preserved unless you pass `--overwrite`. Missing or ambiguous directory labels
cause an error before writing. Run this step again if you reassemble the manifest.

Use `--input_json` to select another manifest or `--output_json` instead of
`--in_place` to write an annotated copy. For another dataset, provide
`--mapping_json` with an object of exact `original_path` to group mappings.
Set `--key_field` to join on a different manifest field.

## Evaluate the annotated test split

```bash
python test.py \
  --detector_path configs/detectors/audio_video/standard/baseline_train_on_megammdf.yaml \
  --weight_path /path/to/model_only_checkpoint.pth \
  --test_datasets Mega-MMDF FakeAVCeleb \
  --log_dir logs/test/baseline_megammdf_fairness \
  --metrics 'video-level DP' 'video-level EO' 'video-level OAE' 'video-level MEO' 'video-level SDAUC'
```

Use a fresh log directory if your cached predictions were generated before
annotation. The loader carries `demographic_group` into the saved predictions.
SDAUC is the population standard deviation of the groups’ video-level ROC AUCs.
It weights groups equally and requires both real and fake videos in each group.
Lower values indicate less AUC variation across groups. If cached predictions
already contain `demographic_group`, rerun the command with the same `--log_dir`
to compute SDAUC from those predictions.

See the [metric guide](../../metrics/README.md) for the five metrics and the
[group performance analysis](../../analysis/fairness/README.md) for individual
group AUCs.
