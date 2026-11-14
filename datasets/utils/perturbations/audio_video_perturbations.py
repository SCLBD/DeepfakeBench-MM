"""Operations that change temporal correspondence between audio and video."""

import numpy as np
import os
import sys

from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.utils.perturbations.audio_perturbations import audio_aac, audio_dropout, read_audio_file, write_audio_file
from datasets.utils.perturbations.common_perturbations import (
    bitrate_param,
    crf_param,
    numeric_param,
    probability_param,
    ratio_param,
)
from datasets.utils.perturbations.video_perturbations import video_downscale, video_frame_drop, video_h264


def video_delay(input_path, output_path, param):
    """
    Shift the complete audio track relative to an unchanged video timeline.
    Args:
        input_path (str or Path): Prepared mono WAV paired with video segments.
        output_path (str or Path): Shifted WAV of identical sample rate/length.
        param (str or float): Signed offset in seconds. Positive d produces
            output[t] = input[t+d], matching the existing future-audio reading
            convention for delayed video. Out-of-bounds positions are silent.
    Returns:
        None. This is a fixed temporal offset, not the model-window-dependent
        circular real-window reassignment used in the misalignment analysis.
    """
    seconds = float(numeric_param(param))
    audio, sample_rate = read_audio_file(input_path)
    shift = int(round(seconds * sample_rate))
    shifted = np.zeros_like(audio)
    source_start = max(0, shift)
    output_start = max(0, -shift)
    length = min(len(audio) - source_start, len(audio) - output_start)
    if length > 0:
        shifted[output_start : output_start + length] = audio[source_start : source_start + length]
    write_audio_file(output_path, shifted, sample_rate)


# Manually register functions supported by prepared audio-video datasets.
# name -> (affected file kind, operation function, actual-parameter parser)
AUDIO_VIDEO_OPERATIONS = {
    'video_h264': ('video', video_h264, crf_param),
    'video_downscale': ('video', video_downscale, ratio_param),
    'video_frame_drop': ('video', video_frame_drop, probability_param),
    'audio_aac': ('audio', audio_aac, bitrate_param),
    'audio_dropout': ('audio', audio_dropout, probability_param),
    'video_delay': ('audio', video_delay, numeric_param),
}


def get_perturbation_directory(perturbation_type, param):
    """
    Resolve the shared directory convention from an operation and actual parameter.
    Args:
        perturbation_type (str): Name manually registered in AUDIO_VIDEO_OPERATIONS.
        param (str, int or float): Actual parameter, not a severity level.
    Returns:
        directory (str): Folder name audio-video_<type>_<normalized_param> beneath preprocessed_dir.
            The runner and test loader use the same parameter parser and naming rule.
    """
    if perturbation_type not in AUDIO_VIDEO_OPERATIONS:
        raise ValueError(f'Unsupported audio-video perturbation: {perturbation_type!r}.')
    parameter = AUDIO_VIDEO_OPERATIONS[perturbation_type][2](param)
    return f'audio-video_{perturbation_type}_{parameter}'


if __name__ == '__main__':
    from pathlib import Path

    # Edit these example paths/parameters to match your local prepared data.
    unit_test_params = {
        'input_path': Path('data/examples/audio.wav'),
        'output_path': Path('outputs/perturbation_examples/audio_offset.wav'),
        'param': 0.16,
    }
    video_delay(**unit_test_params)
    print(f'Saved shifted audio to {unit_test_params["output_path"]}')
