# Demographic subgroup analysis

Compares annotated subgroup metrics with the overall reference and plots subgroup-minus-overall deltas and cohort sizes.

```bash
python analysis/fairness/run_fairness.py --log_dir logs/test/AVFF_train_on_Mega-MMDF --dataset_name Mega-MMDF --level video --metric AUC
```

The input is a JSON list at `<log_dir>/predictions/<dataset_name>.json`, with binary `label` (0 real, 1 fake) and finite fake score `prob`. `--level sample` is the default. Video mode additionally uses native `path` identifiers: it groups by `dirname(path)`, ORs labels, and takes the maximum score across evaluated windows. Already-aggregated video fixtures should be used with their original video identities.

`--output_dir` defaults to this module's `outputs/`; `--title` controls the title/PNG filename and `--dpi` defaults to 300. The command saves one PNG and prints statistics. Appendix builders provide the paper's multi-panel PNG/PDF layouts.

`--metric` accepts `ACC` (default), `AUC`, `AP`, or `EER`. `--threshold` defaults to 0.5 and affects ACC only, with fake iff `prob > threshold`. AUC/AP/EER are NaN for a single-class subgroup. Every record needs an explicit `demographic_group`; video mode rejects inconsistent annotations within a video.

For example, a record can include `"demographic_group": "Black - Female"`. Group membership comes from supplied dataset annotations. See the [fairness protocol](../../commands/fairness/README.md) for the DP, EO, OAE, MEO, and SDAUC metrics.
