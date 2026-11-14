# ROC curve

Reports AUC, empirical EER, and its sampled ROC threshold. EER uses the discrete ROC vertex nearest FPR=FNR; both classes are required.

```bash
python analysis/roc/run_roc.py --log_dir logs/test/AVFF_train_on_Mega-MMDF --dataset_name Mega-MMDF --level video
```

The input is a JSON list at `<log_dir>/predictions/<dataset_name>.json`, with binary `label` (0 real, 1 fake) and finite fake score `prob`. `--level sample` is the default. Video mode additionally uses native `path` identifiers: it groups by `dirname(path)`, ORs labels, and takes the maximum score across evaluated windows. Already-aggregated video fixtures should be used with their original video identities.

`--output_dir` defaults to this module's `outputs/`; `--title` controls the title/PNG filename and `--dpi` defaults to 300. The command saves one PNG and prints statistics. Appendix builders provide the paper's multi-panel PNG/PDF layouts.
