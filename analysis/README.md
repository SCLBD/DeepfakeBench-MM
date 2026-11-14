# Analysis tools

These modules analyze saved detector predictions, extracted features, and dataset metadata.

| Module | Purpose |
|---|---|
| [roc](roc/README.md), [pr](pr/README.md) | ROC and precision–recall curves at sample or video level. |
| [confusion_matrix](confusion_matrix/README.md) | Thresholded predictions and confusion matrices. |
| [probability_distribution](probability_distribution/README.md) | Score distributions for authentic and forged samples or modality groups. |
| [fairness](fairness/README.md) | Score and performance summaries by demographic group. |
| [predicted_prob_histrogram](predicted_prob_histrogram/README.md) | Complete-input and modality-masked score distributions and paper figures. |
| [predicted_prob_change_histogram](predicted_prob_change_histogram/README.md) | Paired score changes after masking audio. |
| [misalignment](misalignment/README.md) | Paired score changes after shifting audio within real clips. |
| [pipeline_figures](pipeline_figures/README.md) | Pipeline-wise AUC figures for seven detectors and five training datasets. |
| [tsne](tsne/README.md) | t-SNE projections from saved detector features. |
| [data_scale](data_scale/README.md) | Within-dataset and cross-dataset results at different training data scales. |
| [dataset_statistics](dataset_statistics/README.md) | Clip counts, demographics, durations, and other dataset statistics. |
| [utils](utils/README.md) | Shared analysis utilities. |

Each module documents its inputs and provides a separate command for each analysis protocol. The standard prediction tools read
JSON exported by `test.py`. Paper figures may read bundled scores or coordinates from
`fixtures/`. Generated results go under `outputs/`.
