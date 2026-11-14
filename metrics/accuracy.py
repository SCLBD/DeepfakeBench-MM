import numpy as np
import os

from sklearn.metrics import accuracy_score


def calculate_binary_accuracy(records, type, threshold=0.5):
    """Compute binary accuracy after applying a strict probability threshold.

    Args:
        records: List of mappings with label (0 real, 1 fake) and prob fields.
            Video-level evaluation also requires path and groups segments by
            its parent directory, using the maximum score and any-fake label.
        type: 'sample-level' for individual records or 'video-level' for aggregation.
        threshold: Scores strictly above this value are classified as fake.

    Returns:
        Accuracy as a float.
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

    if probabilities.max() > 1 or probabilities.min() < 0:
        raise ValueError('y_pred must be between 0 and 1. ')

    pred_labels = (probabilities > threshold).astype(int)
    return accuracy_score(ground_truths, pred_labels)
