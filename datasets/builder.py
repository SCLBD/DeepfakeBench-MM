"""Explicit dataset construction and selection of prepared data versions."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.audio_video.audio_video_dataset import AudioVideoDataset
from datasets.audio_video.audio_video_mask_modality_dataset import AudioVideoMaskModalityDataset
from datasets.audio_video.audio_video_misaligned_real_dataset import AudioVideoMisalignedRealDataset
from datasets.audio_video.audio_video_real_dataset import AudioVideoRealDataset
from datasets.utils.perturbations.audio_video_perturbations import AUDIO_VIDEO_OPERATIONS, get_perturbation_directory


def select_dataset_class(data_type, mode, data_config, aug_config, robustness_config):
    """
    Select an ordinary or specialized loader without changing its constructor behavior.
    Args:
        data_type (str): Explicit dataset selector.
        mode (str): train, val or test.
        data_config (dict): Sampling settings, including the legacy only_real_data flag.
        aug_config (str, dict or None): Ordinary/random-mask training settings.
        robustness_config (dict): Test mask or offline perturbation settings.
    Returns:
        dataset_class (type): Requested implemented class. Image/audio/video classes
            are currently placeholders and are not registered as working loaders.
    """
    classes = {
        'audio_video': AudioVideoDataset,
        'audio_video_mask_modality': AudioVideoMaskModalityDataset,
        'audio_video_real': AudioVideoRealDataset,
        'audio_video_misaligned_real': AudioVideoMisalignedRealDataset,
    }
    if data_type not in classes:
        raise NotImplementedError(f'{data_type} dataset has not been implemented.')
    if data_type != 'audio_video':
        return classes[data_type]
    if mode == 'test' and robustness_config.get('type') in ['mask_audio', 'mask_video']:
        return AudioVideoMaskModalityDataset
    if mode == 'train' and isinstance(aug_config, dict):
        audiovisual = aug_config.get('audiovisual')
        if isinstance(audiovisual, dict) and audiovisual.get('mask_modality'):
            return AudioVideoMaskModalityDataset
    if mode in ['train', 'val'] and data_config.get('only_real_data', False):
        return AudioVideoRealDataset
    return AudioVideoDataset


def resolve_data_config(data_type, data_config, mode, robustness_config=None):
    """
    Select an offline perturbation directory without mutating the input config.
    Args:
        data_type (str): Dataset selector, currently an implemented audio-video type.
        data_config (dict): Ordinary data settings, optionally with data_directory.
        mode (str): Dataset mode; robustness requests are test-only.
        robustness_config (dict or None): Empty, a modality mask, or a mapping with
            type and actual param. Numerical strength presets are not supported.
    Returns:
        resolved (dict): Original mapping for clean/masked inputs, or a shallow copy
            with data_directory set to audio-video_<type>_<param> for offline inputs.
    """
    robustness_config = robustness_config or {}
    kind = robustness_config.get('type')
    if not kind:
        return data_config
    if mode != 'test':
        raise ValueError('robustness_config is supported only in test mode.')
    if data_type == 'audio_video_misaligned_real':
        raise ValueError('Do not combine window misalignment with robustness interventions.')
    if kind in ['mask_audio', 'mask_video']:
        return data_config
    if kind not in AUDIO_VIDEO_OPERATIONS:
        raise ValueError(f'Unsupported robustness type: {kind!r}.')
    if robustness_config.get('param') is None:
        raise ValueError('Offline robustness tests require an actual param, provided by --robustness_param.')
    directory = get_perturbation_directory(kind, robustness_config['param'])
    if data_config.get('data_directory') not in [None, 'audio-video', directory]:
        raise ValueError('data_directory conflicts with the requested robustness type and parameter.')
    return {**data_config, 'data_directory': directory}


def choose_dataset(data_type, data_config, mode, dataset_list=None, dataset_name=None, aug_config=None, robustness_config=None):
    """
    Build a dataset using ordinary settings or a previously generated data version.
    Args:
        data_type (str): Implemented loader selector, e.g., audio_video or audio_video_real.
        data_config (dict): Input, sampling and optional data_directory settings.
        mode (str): train, val or test.
        dataset_list (list or None): Dataset names combined in train mode.
        dataset_name (str or None): Single dataset name used in val/test mode.
        aug_config (str, dict or None): Ordinary/random-mask training augmentation.
        robustness_config (dict or None): Test mask or offline operation type/param.
    Returns:
        dataset (AbstractDataset): Selected loader. Offline perturbations reuse the
            ordinary loading path and must already exist in the selected directory.
    """
    assert (mode == 'train' and dataset_list is not None) or (mode in ['val', 'test'] and dataset_name is not None)
    robustness_config = robustness_config or {}
    data_config = resolve_data_config(data_type, data_config, mode, robustness_config)
    dataset_class = select_dataset_class(data_type, mode, data_config, aug_config, robustness_config)
    kwargs = dict(mode=mode, data_config=data_config, aug_config=aug_config, dataset_list=dataset_list, dataset_name=dataset_name)
    if dataset_class in [AudioVideoMaskModalityDataset, AudioVideoMisalignedRealDataset]:
        kwargs['robustness_config'] = robustness_config
    dataset = dataset_class(**kwargs)
    if (
        mode in ['train', 'val']
        and data_config.get('only_real_data', False)
        and not isinstance(dataset, AudioVideoRealDataset)
    ):
        dataset.data_info_list = AudioVideoRealDataset._filter_real_samples(dataset.data_info_list)
    return dataset
