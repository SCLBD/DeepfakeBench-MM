import numpy as np
import os


def calculate_binary_dp(records, type, threshold=0.5):
    """Compute the maximum demographic gap in predicted-class rates.

    Args:
        records: List of mappings with prob fields. Video-level evaluation also
            requires path and groups segments by its parent directory, using
            the maximum segment score.
            Each record needs a nonempty demographic_group; at least two groups
            are required, and all segments of a video must share one group.
        type: 'sample-level' for individual records or 'video-level' for aggregation.
        threshold: Scores strictly above this value are classified as fake.

    Returns:
        Maximum predicted-class rate gap across demographic groups.
    """
    # Extract probabilities and demographic groups.
    if type == 'sample-level':
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

        probabilities = []
        D = []
        for video_name, video_records in video_to_records_dict.items():
            video_segment_probs = [record['prob'] for record in video_records]
            video_groups = {record.get('demographic_group') for record in video_records}
            if len(video_groups) != 1:
                raise ValueError(
                    f'Video {video_name} has inconsistent demographic groups: {video_groups}'
                )
            probabilities.append(max(video_segment_probs))
            D.append(video_groups.pop())
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

    # F_DP = max_{q in {0,1}} [ max_{Jj in J} P(Y_hat=q | D=Jj)
    #                            - min_{Jj in J} P(Y_hat=q | D=Jj) ]
    F_DP_values = []
    for q in (0, 1):
        rates = []
        for J_j in J:
            mask = D == J_j
            rates.append((Y_hat[mask] == q).sum() / mask.sum())
        F_DP_values.append(max(rates) - min(rates))

    return max(F_DP_values)
