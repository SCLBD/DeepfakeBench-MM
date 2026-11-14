"""Video augmentation with shared random transforms across frames and landmarks."""

import numpy as np
import os
import sys

from albumentations import (
    FancyPCA,
    GaussianBlur,
    HorizontalFlip,
    HueSaturationValue,
    ImageCompression,
    KeypointParams,
    OneOf,
    RandomBrightnessContrast,
    ReplayCompose,
    Rotate,
    VerticalFlip,
)
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def init_video_aug(aug_cfg, with_landmark=False):
    """
    Create video transformation function based on augmentation method.
    Args:
        aug_cfg (dict): config of augmentation settings.
        with_landmark (bool): whether to use landmark or not.
    Returns:
        transform (albumentations.core.composition.Compose): Augmentation function.
    """
    trans = []
    if aug_cfg.get('flip'):
        if aug_cfg['flip'].get('type', 'horizontal') == 'horizontal':
            trans.append(HorizontalFlip(p=aug_cfg['flip'].get('prob', 0.5)))
        elif aug_cfg['flip'].get('type', 'horizontal') == 'vertical':
            trans.append(VerticalFlip(p=aug_cfg['flip'].get('prob', 0.5)))
        else:
            raise NotImplementedError(f'{aug_cfg["flip"]["type"]} flip has not been supported.')
    if aug_cfg.get('rotate'):
        trans.append(Rotate(limit=aug_cfg['rotate'].get('range', [-10, 10]), p=aug_cfg['rotate'].get('prob', 0.5)))
    if aug_cfg.get('gaussian_blur', None) is not None:
        trans.append(
            GaussianBlur(blur_limit=aug_cfg['gaussian_blur'].get('range', [3, 7]), p=aug_cfg['gaussian_blur'].get('prob', 0.5))
        )
    if aug_cfg.get('color', None) is not None:
        trans.append(
            OneOf(
                [
                    RandomBrightnessContrast(
                        brightness_limit=aug_cfg['color'].get('brightness_range', [-0.1, 0.1]),
                        contrast_limit=aug_cfg['color'].get('contrast_range', [-0.1, 0.1]),
                    ),
                    FancyPCA(),
                    HueSaturationValue(),
                ],
                p=aug_cfg['color'].get('prob', 0.5),
            )
        )
    if aug_cfg.get('compression', None) is not None:
        strength = aug_cfg['compression'].get('range', [40, 100])
        trans.append(
            ImageCompression(quality_lower=strength[0], quality_upper=strength[1], p=aug_cfg['compression'].get('prob', 0.5))
        )

    keypoint_params = KeypointParams(format='xy', remove_invisible=False) if with_landmark else None
    return ReplayCompose(trans, keypoint_params=keypoint_params)


def apply_video_aug(transform, frames, landmarks=None):
    """
    Apply predefined visual augmentations.
    Args:
        transform (albumentations.core.composition.ReplayCompose): Initialized video augmentation pipeline.
        frames (np.ndarray): uint8 RGB frames in (T,H,W,C), range [0,255]. Grayscale keeps C=1;
            color-specific transforms use RGB semantics, so convert BGR before calling.
        landmarks (np.ndarray or None): Per-frame landmarks in shape (T, 68, 2), or None.
    Returns:
        augmented_frames (np.ndarray or list): augmented frames.
        augmented_landmarks (np.ndarray or list): augmented landmarks if landmarks provided.
    """
    if frames.ndim == 4:
        replay_params = None
        augmented_frames, augmented_landmarks = [], []
        for i in range(len(frames)):
            kwargs = {'image': frames[i]}
            if landmarks is not None:
                kwargs['keypoints'] = landmarks[i]

            if i == 0:  # Record augment parameters at the first time
                augmented = transform(**kwargs)
                replay_params = augmented['replay']
            else:
                augmented = ReplayCompose.replay(replay_params, **kwargs)

            augmented_frames.append(augmented['image'])
            if landmarks is not None:
                augmented_landmarks.append(np.array(augmented['keypoints']))

        return augmented_frames, np.array(augmented_landmarks) if landmarks is not None else None
    else:
        raise RuntimeError(f'Cannot apply visual augmentations to variable `frames`: {type(frames)}')


if __name__ == '__main__':
    from pathlib import Path

    from datasets.utils.data_presenters.audio_video_presenter import present_audio_video

    unit_test_params = {
        'aug_config': {'flip': {'type': 'horizontal', 'prob': 1.0}},
        'output_directory': Path('outputs/augmentation_examples'),
    }
    # A colored gradient makes the horizontal flip easy to inspect.
    x, y = np.meshgrid(np.arange(96), np.arange(64))
    frame = np.stack((x * 2, y * 3, x + y), axis=-1).astype(np.uint8)
    frames = np.repeat(frame[np.newaxis], 4, axis=0)
    waveform = np.sin(np.arange(16000) * 2 * np.pi * 220 / 16000).astype(np.float32) * 0.2
    transform = init_video_aug(unit_test_params['aug_config'])
    augmented, _ = apply_video_aug(transform, frames)
    output_directory = unit_test_params['output_directory']
    output_directory.mkdir(parents=True, exist_ok=True)
    present_audio_video(frames.transpose(3, 0, 1, 2), waveform, 1, output_directory / 'video_before.png')
    present_audio_video(np.asarray(augmented).transpose(3, 0, 1, 2), waveform, 1, output_directory / 'video_after.png')
