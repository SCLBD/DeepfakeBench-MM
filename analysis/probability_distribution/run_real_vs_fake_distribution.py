"""Plot predicted fake scores for authentic and forged records."""

from __future__ import annotations

import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.probability_distribution.distribution import build_parser, run_distribution



def build_groups(records: list[dict]) -> list[tuple[str, np.ndarray]]:
    """Group fake probabilities by the overall real/fake label."""
    probabilities = np.asarray([record['prob'] for record in records], dtype=np.float64)
    labels = np.asarray([record['label'] for record in records], dtype=np.int64)
    return [
        ('Real', probabilities[labels == 0]),
        ('Fake', probabilities[labels == 1]),
    ]


def main() -> None:
    parser = build_parser(__doc__)
    run_distribution(parser.parse_args(), build_groups)


if __name__ == '__main__':
    main()
