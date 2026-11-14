# Training-data scaling figure

`plot.py` reads the paper's seven-model scaling table and produces its figures.

```bash
python analysis/data_scale/plot.py --plot-only
```

The bundled `fixtures/data.json` contains the recorded AUCs for seven detectors at 1%, 2%, 5%, 10%, 20%, 50%, and 100% training data. The plotter uses `outputs/data.json` when a workbook has been extracted locally.

Workbook extraction requires `openpyxl`. Plotting the bundled JSON works with the standard project dependencies.

```bash
pip install openpyxl
python analysis/data_scale/extract.py --workbook /path/to/results.xlsx
```

`extract.py` reads `Small-scale scaling law!A8:P56` and saves `outputs/data.json`. Columns C and D supply the plotted values, and columns F through L supply the seven cross-dataset results. The output records the workbook hash and any discrepancies between recorded and recomputed means.

`plot.py --workbook <path>` runs extraction and plotting together. The existing `plot.py --workbook <path> --extract-only` command also works.

The module-local `outputs/` directory receives a report and four figures in PNG and PDF. The main figure compares within-dataset and cross-dataset AUC on a log axis. The delta figure subtracts each detector's AUC at 100% training data.

The [scaling configs](../../configs/detectors/audio_video/small_data_scale/README.md) cover the six reduced training and validation subsets. The 100% setting uses the full training configuration. The cross-dataset mean covers the seven other test datasets, four of which also participate in validation.
