import os
import sys

from audiomentations import AddGaussianNoise, BandPassFilter, Compose, Gain, PitchShift
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def init_audio_aug(aug_cfg):
    """
    Create audio transformation function based on augmentation method.
    Args:
        aug_cfg (dict): config of augmentation settings.
    Returns:
        transform (audiomentations.core.composition.Compose): Augmentation function.
    """
    trans = []
    if aug_cfg.get('gaussian_noise', None) is not None:
        strength = aug_cfg['gaussian_noise'].get('range', [0.001, 0.01])
        trans.append(
            AddGaussianNoise(min_amplitude=strength[0], max_amplitude=strength[1], p=aug_cfg['gaussian_noise'].get('prob', 0.5))
        )
    if aug_cfg.get('pitch_shift', None) is not None:
        strength = aug_cfg['pitch_shift'].get('range', [-2, 2])
        trans.append(PitchShift(min_semitones=strength[0], max_semitones=strength[1], p=aug_cfg['pitch_shift'].get('prob', 0.5)))
    if aug_cfg.get('volume_gain', None) is not None:
        strength = aug_cfg['volume_gain'].get('range', [-6, 6])
        trans.append(Gain(min_gain_db=strength[0], max_gain_db=strength[1], p=aug_cfg['volume_gain'].get('prob', 0.5)))

    return Compose(trans)


def apply_audio_aug(transform, audio, sr):
    """
    Apply predefined audio augmentations.
    Args:
        transform (audiomentations.core.composition.Compose): Augmentation function.
        audio (np.ndarray): float32 mono waveform (L,), normally in [-1,1].
        sr (int): sample rate of audio.
    Returns:
        augmented_audio (np.ndarray): float32 waveform (L,); gain/noise can exceed [-1,1].
    """
    return transform(samples=audio, sample_rate=sr)


if __name__ == '__main__':
    import numpy as np

    from pathlib import Path

    from datasets.utils.data_presenters.audio_video_presenter import present_audio_video, save_audio_video

    unit_test_params = {
        'aug_config': {'volume_gain': {'range': [-6, -6], 'prob': 1.0}},
        'audio_sample_rate': 16000,
        'output_directory': Path('outputs/augmentation_examples'),
    }
    sample_rate = unit_test_params['audio_sample_rate']
    waveform = np.sin(np.arange(sample_rate) * 2 * np.pi * 220 / sample_rate).astype(np.float32) * 0.2
    transform = init_audio_aug(unit_test_params['aug_config'])
    augmented = apply_audio_aug(transform, waveform.copy(), sample_rate)
    frames = np.zeros((3, 4, 64, 96), dtype=np.uint8)
    frames[1] = 100
    output_directory = unit_test_params['output_directory']
    output_directory.mkdir(parents=True, exist_ok=True)
    present_audio_video(frames, waveform, 1, output_directory / 'audio_before.png')
    present_audio_video(frames, augmented, 1, output_directory / 'audio_after.png')
    save_audio_video(frames, augmented, output_directory / 'audio_after.mp4', video_fps=4, audio_sample_rate=sample_rate)
