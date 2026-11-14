'''Run a small t-SNE example with twenty manually written 10-D samples.'''

from __future__ import annotations

import argparse
import numpy as np
import os
import sys

from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.tsne import TSNEConfig, TSNEGroup, plot_tsne_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Draw a demo t-SNE from manually grouped 10-D feature vectors.')
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).resolve().parent / 'outputs',
        help='Output directory. PNG and HTML files are saved here.')
    parser.add_argument('--title', type=str, default='tsne_demo',
                        help='Filename stem and plot title.')
    parser.add_argument('--max_samples_per_group', type=int, default=None,
                        help='Optional random sampling limit applied independently to every group.')

    parser.add_argument('--perplexity', type=float, default=5.0)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--tsne-seed', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=600)
    return parser.parse_args()


def build_groups() -> list[TSNEGroup]:
    '''Manually define grouping rules, features, hover strings, and styles here.'''

    real_dataset_a = np.array(
        [
            [0.12, 0.18, 0.09, 0.21, 0.15, 0.11, 0.24, 0.16, 0.08, 0.19],
            [0.15, 0.14, 0.12, 0.19, 0.18, 0.09, 0.20, 0.13, 0.11, 0.17],
            [0.09, 0.22, 0.14, 0.17, 0.13, 0.16, 0.21, 0.10, 0.15, 0.20],
            [0.18, 0.11, 0.16, 0.14, 0.21, 0.12, 0.17, 0.19, 0.10, 0.13],
            [0.11, 0.16, 0.20, 0.12, 0.17, 0.14, 0.19, 0.09, 0.18, 0.15],
        ], dtype=np.float32,
    )
    fake_dataset_a = np.array(
        [
            [1.22, 1.10, 1.18, 1.31, 1.16, 1.24, 1.08, 1.27, 1.13, 1.20],
            [1.17, 1.26, 1.11, 1.23, 1.29, 1.15, 1.21, 1.09, 1.25, 1.14],
            [1.28, 1.16, 1.25, 1.12, 1.20, 1.30, 1.14, 1.18, 1.08, 1.24],
            [1.13, 1.21, 1.30, 1.15, 1.09, 1.19, 1.27, 1.22, 1.16, 1.11],
            [1.25, 1.14, 1.20, 1.28, 1.12, 1.17, 1.23, 1.10, 1.29, 1.18],
        ], dtype=np.float32,
    )
    real_dataset_b = np.array(
        [
            [-1.18, 0.82, -1.10, 0.91, -1.24, 0.76, -1.05, 0.88, -1.16, 0.79],
            [-1.09, 0.75, -1.22, 0.86, -1.13, 0.92, -1.19, 0.81, -1.07, 0.89],
            [-1.25, 0.90, -1.14, 0.78, -1.08, 0.84, -1.21, 0.94, -1.12, 0.73],
            [-1.15, 0.87, -1.06, 0.74, -1.20, 0.89, -1.11, 0.77, -1.26, 0.93],
            [-1.21, 0.79, -1.17, 0.95, -1.10, 0.80, -1.23, 0.85, -1.04, 0.90],
        ], dtype=np.float32,
    )
    fake_dataset_b = np.array(
        [
            [0.74, -1.18, 0.82, -1.06, 0.91, -1.23, 0.78, -1.12, 0.86, -1.20],
            [0.88, -1.09, 0.76, -1.21, 0.84, -1.15, 0.93, -1.04, 0.80, -1.17],
            [0.79, -1.25, 0.90, -1.13, 0.75, -1.08, 0.87, -1.19, 0.94, -1.11],
            [0.92, -1.14, 0.81, -1.24, 0.77, -1.10, 0.85, -1.16, 0.73, -1.22],
            [0.83, -1.20, 0.95, -1.07, 0.80, -1.17, 0.74, -1.26, 0.89, -1.12],
        ], dtype=np.float32,
    )

    return [
        TSNEGroup(
            name='Dataset A / Real',
            features=real_dataset_a,
            hover_texts=[f'path=/demo/dataset_a/real/video_{i:03d}.mp4\nlabel=real\nsubset=A' for i in range(len(real_dataset_a))],
            color='#0072B2',
            marker='o',
            marker_size=62,
            alpha=0.86,
            edge_color='white',
            edge_width=0.8,
        ),
        TSNEGroup(
            name='Dataset A / Fake',
            features=fake_dataset_a,
            hover_texts=[f'path=/demo/dataset_a/fake/video_{i:03d}.mp4\nlabel=fake\nsubset=A' for i in range(len(fake_dataset_a))],
            color='#D55E00',
            marker='^',
            marker_size=76,
            alpha=0.86,
            edge_color='white',
            edge_width=0.8,
        ),
        TSNEGroup(
            name='Dataset B / Real',
            features=real_dataset_b,
            hover_texts=[f'path=/demo/dataset_b/real/subject_{i:03d}.mp4\nlabel=real\ndemographic_group=B1' for i in range(len(real_dataset_b))],
            color='#009E73',
            marker='s',
            marker_size=62,
            alpha=0.86,
            edge_color='white',
            edge_width=0.8,
        ),
        TSNEGroup(
            name='Dataset B / Fake',
            features=fake_dataset_b,
            hover_texts=[f'path=/demo/dataset_b/fake/subject_{i:03d}.mp4\nlabel=fake\ndemographic_group=B2' for i in range(len(fake_dataset_b))],
            color='#CC79A7',
            marker='D',
            marker_size=68,
            alpha=0.86,
            edge_color='white',
            edge_width=0.8,
        ),
    ]


def main() -> None:
    args = parse_args()
    result = plot_tsne_groups(
        build_groups(),
        args.output_dir / f'{args.title}.png',
        config=TSNEConfig(
            perplexity=args.perplexity,
            max_iter=args.max_iter,
            sample_seed=args.sample_seed,
            tsne_seed=args.tsne_seed,
            max_samples_per_group=args.max_samples_per_group,
        ),
        title=args.title,
        dpi=args.dpi,
    )
    print(f'Static t-SNE saved to: {args.output_dir / f"{args.title}.png"}')
    print(f'Interactive t-SNE saved to: {args.output_dir / f"{args.title}.html"}')
    print(f'Projected {result.total_selected_samples} samples from {len(result.groups)} groups.')


if __name__ == '__main__':
    main()
