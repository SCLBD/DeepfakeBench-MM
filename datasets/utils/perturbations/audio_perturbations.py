"""Offline AAC round trips and waveform packet-loss simulation."""

import json
import numpy as np
import os
import subprocess
import sys
import tempfile

from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.utils.perturbations.common_perturbations import atomic_output, bitrate_param, probability_param, run_ffmpeg


def read_audio_file(input_path):
    """
    Decode one prepared mono WAV to float32 without changing its sample rate.
    Args:
        input_path (str or Path): Prepared mono waveform file.
    Returns:
        audio (np.ndarray): Mono float32 samples, shape (L,).
        sample_rate (int): Original sample rate. No model-specific resampling
            or mean removal is performed by offline perturbation generation.
    """
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate,channels',
                '-of', 'json', os.fspath(input_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.decode('utf-8', errors='replace').strip()) from error
    streams = json.loads(result.stdout).get('streams', [])
    if not streams or streams[0].get('channels') != 1:
        raise ValueError(f'Expected preprocessed mono audio: {input_path}.')
    sample_rate = int(streams[0]['sample_rate'])
    decoded = run_ffmpeg(['-i', os.fspath(input_path), '-map', '0:a:0', '-vn', '-f', 'f32le', '-c:a', 'pcm_f32le', 'pipe:1'])
    audio = np.frombuffer(decoded, dtype='<f4').copy()
    return audio, sample_rate


def write_audio_file(output_path, audio, sample_rate):
    """
    Atomically save float32 mono samples as a lossless floating-point WAV.
    Args:
        output_path (str or Path): Final WAV destination.
        audio (np.ndarray): float32 mono waveform (L,), normally in [-1,1].
        sample_rate (int): Original sampling rate to preserve.
    Returns:
        None. No additional quantization, resampling or audio compression occurs.
    """
    with atomic_output(output_path) as temporary_path:
        run_ffmpeg(
            [
                '-y', '-f', 'f32le', '-ar', str(sample_rate), '-ac', '1', '-i', 'pipe:0', '-c:a', 'pcm_f32le',
                os.fspath(temporary_path),
            ],
            np.asarray(audio, dtype='<f4').tobytes(),
        )


def audio_aac(input_path, output_path, param):
    """
    Encode a prepared waveform as AAC-LC and decode back to a PCM WAV.
    Args:
        input_path (str or Path): Prepared mono WAV.
        output_path (str or Path): Decoded PCM WAV at the original sample rate.
        param (str): Actual bitrate in bits/s or k/M notation, e.g., '64k'.
    Returns:
        None. The original sample count is preserved by trimming/padding the
        decoded result; the WAV contains AAC artifacts, not another AAC stream.
    """
    bitrate = bitrate_param(param)
    audio, sample_rate = read_audio_file(input_path)
    with (
        atomic_output(output_path) as temporary_path,
        tempfile.TemporaryDirectory(prefix='.aac_', dir=Path(output_path).parent) as work_dir,
    ):
        encoded_path = Path(work_dir) / 'encoded.m4a'
        run_ffmpeg(
            [
                '-y', '-i', os.fspath(input_path), '-map', '0:a:0', '-vn', '-ar', str(sample_rate), '-ac', '1',
                '-c:a', 'aac', '-profile:a', 'aac_low', '-b:a', bitrate, '-movflags', '+faststart',
                os.fspath(encoded_path),
            ]
        )
        run_ffmpeg(
            [
                '-y', '-i', os.fspath(encoded_path), '-map', '0:a:0', '-vn',
                '-af', f'apad=whole_len={len(audio)},atrim=end_sample={len(audio)}', '-ar', str(sample_rate),
                '-ac', '1', '-c:a', 'pcm_s16le', os.fspath(temporary_path),
            ]
        )


def audio_dropout(input_path, output_path, param, seed=0, packet_duration_ms=20, fade_duration_ms=2):
    """
    Replace random fixed-duration audio packets with silence and fade boundaries.
    Args:
        input_path (str or Path): Prepared mono WAV.
        output_path (str or Path): WAV with the original length and sample rate.
        param (str or float): Packet-loss probability in [0, 1].
        seed (int): Stable per-file seed supplied by the runner.
        packet_duration_ms (float): Packet length in milliseconds; must be positive.
        fade_duration_ms (float): Boundary fade duration; zero disables fading.
    Returns:
        None. Packet boundaries are defined on the complete prepared waveform,
        independently of a detector's input windows.
    """
    probability = float(probability_param(param))
    audio, sample_rate = read_audio_file(input_path)
    packet_size = max(1, int(round(sample_rate * packet_duration_ms / 1000)))
    packet_count = (len(audio) + packet_size - 1) // packet_size
    dropped = np.random.default_rng(seed).random(packet_count) < probability
    mask = np.repeat(dropped, packet_size)[: len(audio)]
    audio[mask] = 0
    fade_size = int(round(sample_rate * fade_duration_ms / 1000))
    if fade_duration_ms > 0:
        fade_size = max(1, fade_size)
        boundaries = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
        starts, ends = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
        for start, end in zip(starts, ends):
            before = min(fade_size, start)
            if before:
                audio[start - before : start] *= np.linspace(1, 0, before + 1, dtype=np.float32)[1:]
            after = min(fade_size, len(audio) - end)
            if after:
                audio[end : end + after] *= np.linspace(0, 1, after + 1, dtype=np.float32)[:-1]
    write_audio_file(output_path, audio, sample_rate)


if __name__ == '__main__':
    unit_test_params = {
        'input_path': Path('data/examples/audio.wav'),
        'output_path': Path('outputs/perturbation_examples/audio_aac.wav'),
        'param': '64k',
    }
    audio_aac(**unit_test_params)
    print(f'Saved perturbed audio to {unit_test_params["output_path"]}')
