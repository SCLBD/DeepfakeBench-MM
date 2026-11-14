# Paired predicted-score changes

`run_predicted_prob_change_histogram.py` plots video-level `delta p = p(mask audio) - p(full)` for paired RARV/RAFV video IDs. It uses the export described in [mask analysis](../predicted_prob_histrogram/README.md).

```bash
python analysis/predicted_prob_change_histogram/run_predicted_prob_change_histogram.py --pairs /path/to/rafv_pairs.json.gz
```

`--pairs` is required. Options are `--datasets Mega-MMDF`, `--bin-width 0.01`, `--bandwidth 0.02`, and `--output-dir /path/to/figures`. The parser requires an RAFV task; FARV metrics use the adjacent module's entry point.

The displayed range is `[-1,1]`. Defaults produce 200 equal-width bins, separate RARV/RAFV density normalization, boundary-reflected Gaussian binned KDE, and dashed class means. Blue denotes RARV and red RAFV. A positive real-video change moves toward false-positive evidence; a negative fake-video change moves toward false-negative evidence.

PNG/PDF outputs go to `outputs/<dataset>/`, with `<model>` and `all_models` files. The shared implementation is `predicted_prob_histrogram/core.py`. Supply the full/masked paired cache separately; the complete-input appendix fixture lacks masked scores. Additional tasks require corresponding selection, label, layout, and guard changes.
