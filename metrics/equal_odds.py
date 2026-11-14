import numpy as np
import os
import warnings


def calculate_binary_eo(records, type, threshold=0.5):
    """Sum demographic deviations from overall class-conditional prediction rates.

    Args:
        records: List of mappings with label (0 real, 1 fake) and prob fields.
            Video-level evaluation also requires path and groups segments by
            its parent directory, using the maximum score and any-fake label.
            Each record needs a nonempty demographic_group; at least two groups
            are required, and all segments of a video must share one group.
        type: 'sample-level' for individual records or 'video-level' for aggregation.
        threshold: Scores strictly above this value are classified as fake.

    Returns:
        Sum of absolute rate differences over groups and true classes.
        Undefined group/class terms are skipped with a warning.
    """
    # Extract ground-truth labels, probabilities, and demographic groups.
    if type == 'sample-level':
        Y = np.array([record['label'] for record in records])
        probabilities = np.array([record['prob'] for record in records])
        D = np.array([record.get('demographic_group') for record in records])
    elif type == 'video-level':
        # Aggregate video segment records.
        video_to_records_dict = {}
        for record in records:
            video_name = os.path.dirname(record['path'])
            if video_name not in video_to_records_dict.keys():
                video_to_records_dict[video_name] = [record]
            else:
                video_to_records_dict[video_name].append(record)

        Y = []
        probabilities = []
        D = []
        for video_name, video_records in video_to_records_dict.items():
            video_segment_labels = [record['label'] for record in video_records]
            video_segment_probs = [record['prob'] for record in video_records]
            video_groups = {record.get('demographic_group') for record in video_records}
            if len(video_groups) != 1:
                raise ValueError(
                    f'Video {video_name} has inconsistent demographic groups: {video_groups}'
                )
            Y.append(1 if np.any(np.array(video_segment_labels) == 1) else 0)
            probabilities.append(max(video_segment_probs))
            D.append(video_groups.pop())
        Y = np.array(Y)
        probabilities = np.array(probabilities)
        D = np.array(D)
    else:
        raise NotImplementedError(f'{type} fairness metric not implemented.')

    # Check demographic annotation.
    if any(group is None or (isinstance(group, str) and not group.strip()) for group in D):
        raise ValueError(
            "Fairness metric requires 'demographic_group' in each prediction record. "
            "Please make sure the dataset provides demographic annotations and the prediction "
            "JSON is regenerated with demographic_group saved by test.py."
        )

    # Convert probability to predicted label.
    Y_hat = (probabilities > threshold).astype(int)

    # Require at least two demographic groups.
    J = np.unique(D)
    if len(J) < 2:
        raise ValueError(
            f'At least two demographic groups are required for fairness metrics, '
            f'but only {len(J)} found: {list(J)}'
        )

    # F_EO = sum_{Jj in J} sum_{q in {0,1}}
    #            | P(Y_hat=1 | D=Jj, Y=q) - P(Y_hat=1 | Y=q) |
    overall_rates = {}
    for q in (0, 1):
        overall_mask = Y == q
        if overall_mask.sum() > 0:
            overall_rates[q] = (
                (Y_hat[overall_mask] == 1).sum() / overall_mask.sum()
            )
        else:
            overall_rates[q] = None

    F_EO = 0.0
    skipped_terms = False
    for J_j in J:
        for q in (0, 1):
            group_mask = (D == J_j) & (Y == q)
            if group_mask.sum() == 0 or overall_rates[q] is None:
                skipped_terms = True
                continue
            group_rate = (Y_hat[group_mask] == 1).sum() / group_mask.sum()
            F_EO += abs(group_rate - overall_rates[q])

    if skipped_terms:
        warnings.warn(
            'Equal Odds: some demographic groups do not contain samples of a required '
            'class, so those undefined terms are skipped.'
        )

    return F_EO
