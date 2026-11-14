'''Draw a t-SNE of Real plus every manipulation pipeline on Mega-MMDF.

Grouping rule:
- Real: prediction label == 0
- pipeline{x}: non-real samples whose path contains "pipeline{x}" (x > 0)

By default the pipeline indices are auto-discovered from the prediction
paths (e.g. pipeline0, pipeline1, ...); pass --pipelines to restrict or
reorder them explicitly.
'''

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import pickle
import re
import sys

from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.tsne import TSNEConfig, TSNEGroup, plot_tsne_groups

PIPELINE_RE = re.compile(r'pipeline(\d+)')
REAL_COLOR = '#111111'


def pipeline_index_of(path: object) -> int | None:
    match = PIPELINE_RE.search(str(path))
    return int(match.group(1)) if match else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Draw a t-SNE of Real plus every pipeline on Mega-MMDF.')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'analysis/tsne/outputs'),
        help='Static figure path. The optional HTML uses the same stem.')
    parser.add_argument('--log_dir', type=Path, required=True, help='test log directory with features saved.')
    parser.add_argument('--feature_name', type=str, required=True, help='feature name.')
    parser.add_argument('--max_samples_per_group', type=int, default=200,
                        help='Optional random sampling limit applied independently to every group.')
    parser.add_argument('--dataset_name', type=str, required=True, help='with-domain dataset name')
    parser.add_argument('--title', type=str, required=True, help='title of the figure and output file.')
    parser.add_argument('--pipelines', type=str, default=None,
                        help='Optional explicit pipeline indices, e.g. "0-22" or "0,3,7". Default: auto-detect.')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--tsne-seed', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=600)
    return parser.parse_args()


def parse_pipeline_arg(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def discover_pipelines(prediction_list: list[dict], explicit: str | None) -> list[int]:
    if explicit:
        return parse_pipeline_arg(explicit)
    indices = set()
    for item in prediction_list:
        for match in PIPELINE_RE.finditer(str(item.get('path', ''))):
            indices.add(int(match.group(1)))
    return sorted(indices)


def build_groups(args) -> list[TSNEGroup]:
    json_file_path = os.path.join(args.log_dir, 'predictions', f'{args.dataset_name}.json')
    prediction_list = json.load(open(json_file_path, 'r'))
    pickle_file_path = os.path.join(args.log_dir, 'features', args.feature_name, f'{args.dataset_name}.pkl')
    features = pickle.load(open(pickle_file_path, 'rb'))

    pipeline_indices = discover_pipelines(prediction_list, args.pipelines)
    print(f'Found {len(pipeline_indices)} pipelines: {pipeline_indices}')

    tsne_group_list = []

    def add_group(group_name: str, data_list: list[dict], color: str) -> None:
        if not data_list:
            print(f'Warning: no {group_name} samples found; this group will be skipped.')
            return
        feature_list = [features[data['path']][data['sample_frame_range']] for data in data_list]
        annotation_list = ['\n'.join([f'{key}: {value}' for key, value in data.items()]) for data in data_list]
        tsne_group_list.append(TSNEGroup(
            name=group_name,
            features=np.array(feature_list).reshape(len(feature_list), -1),
            hover_texts=annotation_list,
            color=color,
            alpha=0.7,
        ))
        print(f'  -> {group_name}: {len(data_list)} samples')

    print(f'Collecting {args.feature_name} of Real data in {args.dataset_name}...')
    real_data = [item for item in prediction_list if int(item.get('label', -1)) == 0]
    add_group('Real', real_data, REAL_COLOR)

    import matplotlib

    from matplotlib import cm
    from matplotlib.colors import to_hex
    palette = (
        list(cm.tab20.colors)
        + list(cm.tab20b.colors)
        + list(cm.tab20c.colors)
    )
    for rank, index in enumerate(pipeline_indices):
        if index == 0:
            continue  # Pipeline 0 is already represented by the Real group.
        group_name = f'pipeline{index}'
        print(f'Collecting {args.feature_name} of {group_name} data in {args.dataset_name}...')
        data_list = [item for item in prediction_list
                     if int(item.get('label', -1)) != 0
                     and pipeline_index_of(item.get('path')) == index]
        add_group(group_name, data_list, to_hex(palette[rank % len(palette)]))

    return tsne_group_list


def main() -> None:
    args = parse_args()
    result = plot_tsne_groups(
        build_groups(args),
        os.path.join(args.output_dir, f'{args.title}.png'),
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
    print(f't-SNE saved to: {os.path.join(args.output_dir, args.title)}')
    print(f'Projected {result.total_selected_samples} samples from {len(result.groups)} groups.')


if __name__ == '__main__':
    main()
