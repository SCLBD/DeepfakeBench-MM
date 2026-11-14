import os
import posixpath
import sys

from statistics import pstdev

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from metrics.auc import calculate_binary_auc


def calculate_binary_sdauc(records):
    """Return the population standard deviation of demographic-group video AUCs.

    Args:
        records: Segment predictions with path, binary label, fake probability
            (prob), and demographic_group. Segments sharing a parent directory
            form one video, which uses their maximum score and any-fake label.
            Each video must have one consistent group, and every group must
            contain both real and fake videos.
    Returns:
        float: Standard deviation across groups with equal weight and denominator
            equal to the number of groups. At least two groups are required.
    """
    videos = {}
    for record in records:
        group = record.get('demographic_group')
        if group is None or (isinstance(group, str) and not group.strip()):
            raise ValueError('SDAUC requires a nonempty demographic_group in every prediction record.')

        video_path = posixpath.dirname(record['path'].replace('\\', '/'))
        label = int(record['label'] == 1)
        probability = record['prob']
        if video_path not in videos:
            videos[video_path] = {'label': label, 'prob': probability, 'demographic_group': group}
            continue

        video = videos[video_path]
        if video['demographic_group'] != group:
            raise ValueError(f'Video {video_path!r} has inconsistent demographic groups.')
        video['label'] = max(video['label'], label)
        video['prob'] = max(video['prob'], probability)

    groups = {}
    for video in videos.values():
        groups.setdefault(video['demographic_group'], []).append(video)
    if len(groups) < 2:
        raise ValueError(f'SDAUC requires at least two demographic groups, found {len(groups)}.')

    aucs = []
    for group, group_videos in groups.items():
        if len({video['label'] for video in group_videos}) < 2:
            raise ValueError(f'SDAUC requires both real and fake videos in demographic group {group!r}.')
        # The segments have already been aggregated, so each input here is one video.
        aucs.append(calculate_binary_auc(group_videos, type='sample-level'))
    return pstdev(aucs)
