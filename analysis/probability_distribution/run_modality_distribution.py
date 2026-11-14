"""Plot predicted fake scores for the four audio/video authenticity groups."""

from __future__ import annotations

import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.probability_distribution.distribution import build_parser, run_distribution


MODALITY_GROUPS = [
    (0, 0, 'Real'),
    (1, 0, 'Video fake'),
    (0, 1, 'Audio fake'),
    (1, 1, 'Audio+Video fake'),
]


def build_groups(records: list[dict]) -> list[tuple[str, np.ndarray]]:
    """Group fake probabilities by the four audio/video authenticity combinations."""
    probabilities = np.asarray([record['prob'] for record in records], dtype=np.float64)
    if not all('video_label' in record and 'audio_label' in record for record in records):
        raise SystemExit('Modality grouping requires video_label and audio_label in prediction records.')
    groups = []
    for video_label, audio_label, name in MODALITY_GROUPS:
        mask = np.asarray([
            int(record['video_label']) == video_label and int(record['audio_label']) == audio_label
            for record in records
        ])
        if np.any(mask):
            groups.append((name, probabilities[mask]))
    return groups


def main() -> None:
    parser = build_parser(__doc__, title_suffix='Predicted Probability Distribution by Modality')
    run_distribution(parser.parse_args(), build_groups)


if __name__ == '__main__':
    main()
