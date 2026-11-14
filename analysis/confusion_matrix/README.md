# Confusion matrix

Reports TN/FP/FN/TP, accuracy, precision, recall, and F1. Fake classification uses `prob > threshold`.

```bash
python analysis/confusion_matrix/run_confusion_matrix.py --log_dir logs/test/AVFF_train_on_Mega-MMDF --dataset_name Mega-MMDF --level video
```

The input is a JSON list at `<log_dir>/predictions/<dataset_name>.json`, with binary `label` (0 real, 1 fake) and finite fake score `prob`. `--level sample` is the default. Video mode additionally uses native `path` identifiers: it groups by `dirname(path)`, ORs labels, and takes the maximum score across evaluated windows. Already-aggregated video fixtures should be used with their original video identities.

`--output_dir` defaults to this module's `outputs/`; `--title` controls the title/PNG filename and `--dpi` defaults to 300. The command saves one PNG and prints statistics. Appendix builders provide the paper's multi-panel PNG/PDF layouts.

`--threshold` defaults to 0.5. `--normalize` accepts `none` (raw counts), `true` (within true-label rows), `pred` (within predicted-label columns), or `all` (whole matrix). Normalization changes the displayed matrix.
