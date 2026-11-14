# Metrics

[MetricBuilder](builder.py) evaluates predictions, records per-dataset histories,
tracks best scores, and formats result tables for trainers and [test.py](../test.py).
The calculation functions document their required record fields.

| Metric suffix | Function | Preferred direction |
|---|---|---|
| `AUC` | [calculate_binary_auc](auc.py) | Higher |
| `AP` | [calculate_binary_ap](average_precision.py) | Higher |
| `accuracy` | [calculate_binary_accuracy](accuracy.py) | Higher |
| `EER` | [calculate_binary_eer](eer.py) | Lower |
| `DP` | [calculate_binary_dp](demographic_parity.py) | Lower |
| `EO` | [calculate_binary_eo](equal_odds.py) | Lower |
| `OAE` | [calculate_binary_oae](overall_accuracy_equality.py) | Lower |
| `MEO` | [calculate_binary_meo](maximum_equalized_odds.py) | Lower |
| `SDAUC` | [calculate_binary_sdauc](standard_deviation_auc.py) | Lower |

Use a case-sensitive level prefix, such as `video-level AUC`. SDAUC is available
as `video-level SDAUC`. The other table entries support both `sample-level`
and `video-level`. `loss` reads a [Recorder](../tools/recorder.py).

## Evaluation rules

Sample-level metrics treat each prediction record as an observation. Video-level
metrics group segments by their path's parent directory, use the maximum segment
score, and label a video fake when any segment is fake. Accuracy, DP, EO, OAE,
and MEO classify scores strictly above `0.5` as fake by default. The builder uses
that default. EER selects the empirical ROC point nearest equal FNR and FPR.

Fairness metrics require consistent demographic annotations and at least two
groups. DP measures the largest predicted-class rate gap; OAE measures the group
accuracy gap. EO sums group deviations from overall class-conditional rates and
skips undefined terms with a warning. MEO takes the largest group gap among the
four predicted/true-class cells, skips undefined cells, and returns NaN if none
is defined. Dataset coverage therefore matters when comparing these scores.

SDAUC computes ROC AUC within each demographic group and takes the population
standard deviation across those AUCs. Each group has equal weight, and the
variance uses the number of groups as its denominator. SDAUC uses continuous
scores and requires both real and fake videos in every evaluated group.

The trainers' validation records omit demographic annotations; [test.py](../test.py)
preserves them when provided by the dataset.

## Usage and extension

```python
from metrics.builder import MetricBuilder

metric = MetricBuilder('video-level AUC', ['ExampleDataset'])
metric.update(records, 'ExampleDataset')
updated_names = metric.update_best()
table = metric.parse_metrics(include_latest=True)
```

Update every configured dataset once per round before calling `update_best()`.
The `average` entry is an unweighted mean of dataset scores, with infinite values
propagated explicitly. Best scores change only on strict improvement.

To add a metric, implement its calculation function, import it in `builder.py`,
and add its exact name to the direction lists and dispatch branches. Document
aggregation, thresholds, and undefined cases beside the implementation.
