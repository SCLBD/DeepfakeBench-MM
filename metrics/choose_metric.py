import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from metrics.accuracy import binary_accuracy
from metrics.ap import binary_AP
from metrics.auc import binary_AUC
from metrics.eer import binary_EER

def choose_metric(config):
    """
    Choose metric function under ./metrics
    Args:
        config: [str] config containing 'metric_scoring'
    Returns:
        metric_func: [function] metric calculating function
    """
    metric_dict = {
        "acc": binary_accuracy,
        "auc": binary_AUC,
        "eer": binary_EER,
        "ap": binary_AP,
    }

    if config['metric_scoring'] not in metric_dict.keys():
        raise NotImplementedError(f"{config['metric_scoring']} has not been implemented")

    return metric_dict[config['metric_scoring']]