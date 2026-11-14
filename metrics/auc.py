import numpy as np
import os

from sklearn import metrics


def calculate_binary_auc(records, type):
    """Compute area under the empirical binary ROC curve.

    Args:
        records: List of mappings with label (0 real, 1 fake) and prob fields.
            Video-level evaluation also requires path and groups segments by
            its parent directory, using the maximum score and any-fake label.
        type: 'sample-level' for individual records or 'video-level' for aggregation.

    Returns:
        ROC area as a float.
    """
    if type == 'sample-level':
        ground_truths = np.array([record['label'] for record in records])
        probabilities = np.array([record['prob'] for record in records])
    elif type == 'video-level':
        # Aggregate video segment records
        video_to_records_dict = {}
        for record in records:
            video_name = os.path.dirname(record['path'])
            if video_name not in video_to_records_dict.keys():
                video_to_records_dict[video_name] = [record]
            else:
                video_to_records_dict[video_name].append(record)

        ground_truths = []
        probabilities = []
        for video_name, video_records in video_to_records_dict.items():
            video_segment_labels = [record['label'] for record in video_records]
            video_segment_probs = [record['prob'] for record in video_records]
            ground_truths.append(1 if np.any(np.array(video_segment_labels) == 1) else 0)
            probabilities.append(max(video_segment_probs))
        ground_truths = np.array(ground_truths)
        probabilities = np.array(probabilities)
    else:
        raise NotImplementedError(f'{type} AUC not implemented.')

    fpr, tpr, thresholds = metrics.roc_curve(ground_truths, probabilities, pos_label=1)
    return metrics.auc(fpr, tpr)
