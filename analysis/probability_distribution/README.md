# Predicted-score distributions

The scripts plot fake-score density histograms and report the count, mean, and standard deviation for each group.

| Entry point | Groups |
| --- | --- |
| [`run_real_vs_fake_distribution.py`](run_real_vs_fake_distribution.py) | Real and fake according to `label`. |
| [`run_modality_distribution.py`](run_modality_distribution.py) | RARV, RAFV, FARV, and FAFV according to `audio_label` and `video_label`. |

```bash
python analysis/probability_distribution/run_real_vs_fake_distribution.py --log_dir logs/test/AVFF_train_on_Mega-MMDF --dataset_name Mega-MMDF --level video
python analysis/probability_distribution/run_modality_distribution.py --log_dir logs/test/AVFF_train_on_Mega-MMDF --dataset_name Mega-MMDF --level video
```

Both scripts read `<log_dir>/predictions/<dataset_name>.json`. The default level is `sample`. Video mode groups windows by the parent of each `path`, takes the maximum fake score, and assigns a fake label if any evaluated window has that label. Each script writes one PNG under `outputs/` and uses a distinct default filename. Use `--output_dir`, `--title`, `--bins`, and `--dpi` to control the output.

`run_probability_distribution.py --group_by label|modality` remains available for existing commands. See [demographic subgroup analysis](../fairness/README.md) for comparisons across demographic groups.
