"""Draw predicted fake-probability distributions from one prediction JSON."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.probability_distribution.distribution import build_parser, run_distribution
from analysis.probability_distribution.run_modality_distribution import build_groups as build_modality_groups
from analysis.probability_distribution.run_real_vs_fake_distribution import build_groups as build_label_groups


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument('--group_by', type=str, default='label', choices=['label', 'modality'],
                        help='Group probabilities by overall label or audio/video modality labels.')
    args = parser.parse_args()
    build_groups = build_modality_groups if args.group_by == 'modality' else build_label_groups
    run_distribution(args, build_groups)


if __name__ == '__main__':
    main()
