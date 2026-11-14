'''Run a small t-SNE example with twenty manually written 10-D samples.'''

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import pickle
import sys

from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.tsne import TSNEConfig, TSNEGroup, plot_tsne_groups


def parse_args() -> argparse.Namespace:
    # e.g. --log_root logs/test/AVFF_train_on_FakeAVCeleb --feature_name audio_feature --max_samples_per_group 1000
    #      --title "AVFF train on FakeAVCeleb - Real v.s. Fake on FakeAVCeleb - Audio Feature"
    parser = argparse.ArgumentParser(description='Draw a demo t-SNE from manually grouped 10-D feature vectors.')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'analysis/tsne/outputs'),
        help='Static figure path. The optional HTML uses the same stem.')
    parser.add_argument('--log_dir', type=Path, required=True, help='test log directory with features saved.')
    parser.add_argument('--feature_name', type=str, required=True, help='feature name.')
    parser.add_argument('--max_samples_per_group', type=int, default=None,
                        help='Optional random sampling limit applied independently to every group.')
    parser.add_argument('--dataset_name', type=str, required=True, help='with-domain dataset name')
    parser.add_argument('--title', type=str, required=True, help='title of the figure and output file.')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--tsne-seed', type=int, default=0)
    
    parser.add_argument('--dpi', type=int, default=600)
    return parser.parse_args()


def build_groups(args) -> list[TSNEGroup]:
    group_setting = {
        'Real': {'label': 0, 'color': 'orange'},
        'Fake': {'label': 1, 'color': 'green'},
    }
    tsne_group_list = []

    json_file_path = os.path.join(args.log_dir, 'predictions', f'{args.dataset_name}.json')
    prediction_list = json.load(open(json_file_path, 'r'))
    pickle_file_path = os.path.join(args.log_dir, 'features', args.feature_name, f'{args.dataset_name}.pkl')
    features = pickle.load(open(pickle_file_path, 'rb'))

    for group_name in group_setting.keys():
        print(f'Collecting {args.feature_name} of {group_name} data in {args.dataset_name}...')
        data_list = [item for item in prediction_list if item['label'] == group_setting[group_name]['label']]
        feature_list = [features[data['path']][data['sample_frame_range']] for data in data_list]
        annotation_list = ['\n'.join([f'{key}: {value}' for key, value in data.items()]) for data in data_list]
        tsne_group_list.append(TSNEGroup(
            name=group_name,
            features=np.array(feature_list).reshape(len(feature_list), -1),
            hover_texts=annotation_list,
            color=group_setting[group_name]['color'],
            alpha=0.7,
        ))

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
