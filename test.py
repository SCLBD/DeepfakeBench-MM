"""Save detector predictions/features and evaluate requested metrics, reusing complete outputs."""

import argparse
import datetime
import json
import numpy as np
import os
import pickle
import sys
import torch
import yaml

from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.builder import choose_dataset, resolve_data_config
from datasets.utils.data_helpers.audio_video_helpers import validate_audio_video_files
from datasets.utils.perturbations.audio_video_perturbations import AUDIO_VIDEO_OPERATIONS
from metrics.builder import MetricBuilder
from tools.logger import get_logger
from tools.registry import DETECTOR


def arg_parse():
    """
    Parse testing options, including an offline operation's actual parameter.
    Returns:
        args (argparse.Namespace): Validated prediction, metric and data-version settings.
    """
    parser = argparse.ArgumentParser(description='Deepfake Detection Testing Command Line Settings')
    parser.add_argument('--detector_path', required=True, help='Path to detector configuration.')
    parser.add_argument('--weight_path', required=True, help='Path to final detector checkpoint.')
    parser.add_argument('--log_dir', help='Output directory; overrides the path configuration.')
    parser.add_argument('--test_datasets', nargs='+', required=True, help='Dataset names to evaluate.')
    parser.add_argument('--test_batch_size', type=int, help='Override validation batch size.')
    parser.add_argument('--metrics', nargs='+', default=[], help='Optional metrics; omit for prediction-only evaluation.')
    parser.add_argument('--feature_keys', nargs='*', default=[], help='Feature keys returned by the detector.')
    parser.add_argument(
        '--robustness_type',
        choices=[*AUDIO_VIDEO_OPERATIONS, 'mask_video', 'mask_audio'],
        help='Previously generated offline perturbation, or a test-time modality mask.',
    )
    parser.add_argument('--robustness_param', help='Actual offline parameter, matching the generator --param value.')
    args = parser.parse_args()
    if args.robustness_type in AUDIO_VIDEO_OPERATIONS:
        if args.robustness_param is None:
            parser.error('--robustness_param is required for offline robustness tests.')
        try:
            args.robustness_param = AUDIO_VIDEO_OPERATIONS[args.robustness_type][2](args.robustness_param)
        except ValueError as error:
            parser.error(str(error))
    elif args.robustness_param is not None:
        parser.error('--robustness_param requires an offline --robustness_type; modality masks take no parameter.')
    return args


def get_prediction_save_path(prediction_save_root, dataset_name):
    return os.path.join(prediction_save_root, f'{dataset_name}.json')


def load_detector_checkpoint(model, weight_path):
    """Allow checkpoint-managed evaluation adapters; retain strict ordinary loads."""
    if callable(getattr(model, 'load_checkpoint', None)):
        model.load_checkpoint(weight_path)
    else:
        ckpt = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(ckpt, strict=True)


def add_prediction_metadata(record, data_dict, index):
    """Persist pairing metadata, including tensors produced by default_collate."""
    for key in (
        'demographic_group',
        'original_audio_label',
        'original_video_label',
        'original_label',
        'audio_sample_frame_range',
        'video_sample_frame_range',
        'misalignment_applied',
        'misalignment_modality',
        'misalignment_shift',
        'misalignment_shift_frames',
        'original_sample_index',
        'donor_sample_index',
        'num_samples_in_data',
    ):
        if key in data_dict:
            value = data_dict[key][index]
            record[key] = value.detach().cpu().item() if isinstance(value, torch.Tensor) else value


def get_feature_pickle_path(feature_save_root, feature_key, dataset_name):
    return os.path.join(feature_save_root, feature_key, f'{dataset_name}.pkl')


def get_feature_npy_path(feature_save_root, feature_key, data_path, sample_frame_range):
    return os.path.join(feature_save_root, feature_key, os.path.dirname(data_path), sample_frame_range + '.npy')


def remove_empty_parent_dirs(path, stop_dir):
    parent = os.path.dirname(os.path.abspath(path))
    stop_dir = os.path.abspath(stop_dir)
    while os.path.commonpath([parent, stop_dir]) == stop_dir and parent != stop_dir:
        try:
            os.rmdir(parent)
        except OSError:
            break
        parent = os.path.dirname(parent)


def build_test_plan(dataset_names, feature_keys, prediction_save_root, feature_save_root):
    """
    Build a resumable testing plan using dataset-level prediction JSON and feature pickle files.
    A dataset is complete only when its prediction JSON and all requested feature pickle files exist.
    """
    test_plan = {}
    for dataset_name in dataset_names:
        prediction_path = get_prediction_save_path(prediction_save_root, dataset_name)
        missing_feature_keys = [
            feature_key
            for feature_key in feature_keys
            if not os.path.exists(get_feature_pickle_path(feature_save_root, feature_key, dataset_name))
        ]
        prediction_exists = os.path.exists(prediction_path)
        test_plan[dataset_name] = {
            'prediction_exists': prediction_exists,
            'missing_feature_keys': missing_feature_keys,
            'complete': prediction_exists and len(missing_feature_keys) == 0,
        }

    return test_plan


def remove_dataset_feature_npys(dataset_name, feature_save_root, feature_key, logger, existing_npy_paths):
    feature_dir = os.path.join(feature_save_root, feature_key)
    if len(existing_npy_paths) == 0:
        return 0

    removed_npy_count = 0
    for npy_path in tqdm(existing_npy_paths, desc=f'Removing npy {dataset_name}::{feature_key}', dynamic_ncols=True):
        if os.path.exists(npy_path):
            os.remove(npy_path)
            remove_empty_parent_dirs(npy_path, feature_dir)
            removed_npy_count += 1

    logger.info(f'Temporary npy cleanup finished: dataset={dataset_name}, feature={feature_key}, removed_npy={removed_npy_count}')
    return removed_npy_count


def aggregate_dataset_features(dataset_name, prediction_records, feature_save_root, feature_key, logger):
    """
    Aggregate temporary per-sample npy feature files into one dataset-level pickle.
    Remove source npy files only after pickle is successfully created.
    Existing pickle is trusted as complete and skipped without reading it.
    """
    save_path = get_feature_pickle_path(feature_save_root, feature_key, dataset_name)
    temp_save_path = save_path + '.tmp'
    if os.path.exists(save_path):
        logger.info(f'Feature pickle found, skip aggregation: {save_path}')
        return

    aggregated = OrderedDict()
    missing_count = 0
    npy_paths_to_remove = []
    for record in tqdm(prediction_records, desc=f'Aggregating {dataset_name}::{feature_key}', dynamic_ncols=True):
        npy_path = get_feature_npy_path(
            feature_save_root=feature_save_root,
            feature_key=feature_key,
            data_path=record['path'],
            sample_frame_range=record['sample_frame_range'],
        )
        if not os.path.exists(npy_path):
            missing_count += 1
            continue

        feature = np.load(npy_path)
        if record['path'] not in aggregated:
            aggregated[record['path']] = OrderedDict()
        aggregated[record['path']][record['sample_frame_range']] = feature
        npy_paths_to_remove.append(npy_path)

    if missing_count > 0:
        raise RuntimeError(
            f'Missing {missing_count} feature npy files for {dataset_name}::{feature_key}; '
            f'will not create incomplete pickle: {save_path}'
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(temp_save_path, 'wb') as f:
        pickle.dump(aggregated, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_save_path, save_path)

    removed_npy_count = remove_dataset_feature_npys(
        dataset_name=dataset_name,
        feature_save_root=feature_save_root,
        feature_key=feature_key,
        logger=logger,
        existing_npy_paths=list(dict.fromkeys(npy_paths_to_remove)),
    )

    logger.info(
        f'Feature aggregation finished: dataset={dataset_name}, feature={feature_key}, '
        f'videos={len(aggregated)}, missing_npy={missing_count}, removed_npy={removed_npy_count}, saved={save_path}'
    )


def main():
    # Todo: Distributed testing.
    args = arg_parse()

    with open(os.path.join(os.path.dirname(__file__), 'configs/path.yaml'), 'r') as f:
        path_config = yaml.safe_load(f)
    with open(args.detector_path, 'r') as f:
        detector_config = yaml.safe_load(f)

    if args.log_dir is not None:
        path_config['log_dir'] = args.log_dir
    else:
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        path_config['log_dir'] = os.path.join(path_config['log_dir'], f"test/{detector_config['model_name']}_{time_now}")

    config = {
        **detector_config,
        **{'path_config': path_config},
        'weight_path': args.weight_path,
        'metrics': args.metrics,
        'test_datasets': args.test_datasets,
        'test_batch_size': detector_config['val_batch_size'] if args.test_batch_size is None else args.test_batch_size,
        'robustness_config': (
            {
                'type': args.robustness_type,
                'param': args.robustness_param,
            }
            if args.robustness_type is not None
            else {}
        ),
    }
    config['data_config'] = resolve_data_config(config['data_type'], config['data_config'], 'test', config['robustness_config'])
    logger = get_logger(config['path_config']['log_dir'])
    # logger = get_logger(log_dir, rank=config['rank'], world_size=config['world_size'])
    logger.info(f'Save log to {config["path_config"]["log_dir"]}')
    logger.info(f'Config: {json.dumps(config, indent=4)}')
    if not args.metrics:
        logger.info('No metrics requested; saving predictions without metric evaluation.')

    # run testing, save feature if specified, and record predicted probability
    prediction_save_root = os.path.join(config['path_config']['log_dir'], 'predictions')
    feature_save_root = os.path.join(config['path_config']['log_dir'], 'features')
    test_plan = build_test_plan(
        dataset_names=config['test_datasets'],
        feature_keys=args.feature_keys,
        prediction_save_root=prediction_save_root,
        feature_save_root=feature_save_root,
    )
    # Only pending inference needs prepared files; cached metric-only evaluation stays available.
    directory = config['data_config'].get('data_directory', 'audio-video')
    if directory != 'audio-video':
        for name, plan in test_plan.items():
            if plan['complete']:
                continue
            json_path = os.path.join(path_config['json_dir'], 'audio-video', f'{name}.json')
            with open(json_path, encoding='utf-8') as handle:
                records = json.load(handle)['test']
            validate_audio_video_files(
                records,
                os.path.join(path_config['preprocessed_dir'], directory),
                with_landmark=config['data_config'].get('with_landmark', False),
            )
    # Cached metric-only runs do not need a model or checkpoint loaded.
    if any(not plan['complete'] for plan in test_plan.values()):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DETECTOR[config['model_name']](config['model_config'])
        try:
            load_detector_checkpoint(model, args.weight_path)
            logger.info('Successfully loaded checkpoint from {}'.format(config['weight_path']))
        except RuntimeError as e:
            logger.error('Failed to load checkpoint: {}'.format(e))
            raise
        model.to(device)
        model.eval()
    metric_factories = {
        metric_name: MetricBuilder(metric_name, config['test_datasets']) for metric_name in args.metrics
    }

    for dataset_name in config['test_datasets']:
        dataset_test_plan = test_plan[dataset_name]
        prediction_save_path = get_prediction_save_path(prediction_save_root, dataset_name)
        feature_keys_to_save = dataset_test_plan['missing_feature_keys']

        if dataset_test_plan['complete']:
            logger.info(f'Prediction and requested feature pickles on {dataset_name} found, skip inference.')
            with open(prediction_save_path, 'r') as f:
                prediction_records = json.load(f)
        else:
            logger.info(
                f'Start testing on {dataset_name}. '
                f'Prediction exists={dataset_test_plan["prediction_exists"]}, '
                f'Missing feature keys: {feature_keys_to_save}'
            )
            dataset = choose_dataset(
                data_type=config['data_type'],
                dataset_name=dataset_name,
                data_config=config['data_config'],
                mode='test',
                robustness_config=config['robustness_config'],
            )
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=config['test_batch_size'],
                shuffle=False,
                num_workers=config.get('num_workers', 4),
                pin_memory=True,
            )
            log_steps = np.linspace(0, len(dataloader), min(10, len(dataloader)) + 1).astype(np.int32).tolist()[1:]

            prediction_records = []
            pbar = tqdm(total=len(dataloader), desc=f'Testing {dataset_name}', dynamic_ncols=True)
            with torch.no_grad():
                for iteration, data_dict in enumerate(dataloader):
                    for key in data_dict.keys():
                        if isinstance(data_dict[key], torch.Tensor):
                            data_dict[key] = data_dict[key].to(device)

                    batch_size = data_dict['label'].shape[0]
                    predictions = model(data_dict, inference=True)

                    for i in range(batch_size):
                        prediction_record = {
                            'path': data_dict['path'][i],
                            'sample_frame_range': data_dict['sample_frame_range'][i],
                            'label': data_dict['label'][i].detach().cpu().item(),
                            'video_label': data_dict['video_label'][i].detach().cpu().item(),
                            'audio_label': data_dict['audio_label'][i].detach().cpu().item(),
                            'prob': predictions['prob'][i].detach().cpu().item(),
                        }
                        add_prediction_metadata(prediction_record, data_dict, i)
                        prediction_records.append(prediction_record)

                        for feature_key in feature_keys_to_save:
                            feature_save_path = get_feature_npy_path(
                                feature_save_root=feature_save_root,
                                feature_key=feature_key,
                                data_path=data_dict['path'][i],
                                sample_frame_range=data_dict['sample_frame_range'][i],
                            )
                            os.makedirs(os.path.dirname(feature_save_path), exist_ok=True)
                            np.save(feature_save_path, predictions[feature_key][i].detach().cpu().numpy())

                    pbar.update(1)
                    if iteration in log_steps:
                        logger.info(f'Prediction on {dataset_name} finished {int(round(iteration / len(dataloader) * 100))}%.')

            pbar.close()

            os.makedirs(os.path.dirname(prediction_save_path), exist_ok=True)
            with open(prediction_save_path, 'w') as f:
                json.dump(prediction_records, f, indent=4)
            logger.info(f'Prediction file saved: {prediction_save_path}')

        for metric_name, metric_factory in metric_factories.items():
            metric_factory.update(prediction_records, dataset_name)

        if not dataset_test_plan['complete']:
            for feature_key in feature_keys_to_save:
                aggregate_dataset_features(
                    dataset_name=dataset_name,
                    prediction_records=prediction_records,
                    feature_save_root=feature_save_root,
                    feature_key=feature_key,
                    logger=logger,
                )

    for metric_name, metric_factory in metric_factories.items():
        metric_factory.update_best()
        logger.info(f'{metric_name} Testing Result:\n{metric_factory.parse_metrics()}')


if __name__ == '__main__':
    main()
