"""Offline H.264 compression, spatial downscaling and frame-loss concealment."""

import cv2
import numpy as np
import os
import subprocess
import sys
import tempfile

from contextlib import suppress
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.utils.perturbations.common_perturbations import atomic_output, crf_param, probability_param, ratio_param, run_ffmpeg


def video_h264(input_path, output_path, param):
    """
    Apply an additional H.264 encoding pass to one prepared video segment.
    Args:
        input_path (str or Path): Prepared video file, without the separate WAV.
        output_path (str or Path): Destination video file with the same timeline.
        param (str, int or float): Actual CRF value, validated in [0, 51].
    Returns:
        None.
    """
    parameter = crf_param(param)
    with atomic_output(output_path) as temporary_path:
        arguments = [
            '-y', '-i', os.fspath(input_path), '-map', '0:v:0', '-an', '-c:v', 'libx264', '-preset', 'medium',
            '-crf', parameter, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', os.fspath(temporary_path),
        ]
        run_ffmpeg(arguments)


def transform_video_file(input_path, output_path, transform):
    """
    Stream decoded frames through a transform and save lossless RGB H.264.
    Args:
        input_path (str or Path): Prepared fixed-frame-rate video segment.
        output_path (str or Path): Output with original dimensions/FPS/frame count.
        transform (callable): Function (BGR uint8 frame, zero-based index) ->
            transformed BGR uint8 frame with the same shape and [0,255] range.
    Returns:
        None. RGB lossless storage avoids adding a second lossy codec treatment
        to spatial downscaling or frame-loss experiments. Audio stays separate.
    """
    capture = cv2.VideoCapture(os.fspath(input_path))
    if not capture.isOpened():
        raise RuntimeError(f'Failed to open video: {input_path}.')
    process = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        with atomic_output(output_path) as temporary_path, tempfile.TemporaryFile() as errors:
            command = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-nostdin', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s:v', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an',
                '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'medium', '-threads', '1', '-movflags', '+faststart',
                os.fspath(temporary_path),
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            frame_count = 0
            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    frame = transform(frame, frame_count)
                    process.stdin.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
                    frame_count += 1
            except BrokenPipeError:
                pass
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                with suppress(OSError):
                    process.stdin.close()
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(errors.read().decode('utf-8', errors='replace').strip())
            if frame_count == 0 or (expected_frames > 0 and frame_count != expected_frames):
                raise RuntimeError(
                    f'Incomplete video decoding: {input_path}; read {frame_count}, expected {expected_frames} frames.'
                )
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        capture.release()


def video_downscale(input_path, output_path, param):
    """
    Downscale each decoded frame and resize it back to the prepared dimensions.
    Args:
        input_path (str or Path): Prepared video segment.
        output_path (str or Path): Output video with original dimensions/timeline.
        param (str or float): Width/height multiplier in (0, 1]. Uses AREA for
            downscaling and LINEAR for restoring the original dimensions.
    Returns:
        None.
    """
    ratio = float(ratio_param(param))

    def transform(frame, index):
        height, width = frame.shape[:2]
        size = max(1, int(round(width * ratio))), max(1, int(round(height * ratio)))
        reduced = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)

    transform_video_file(input_path, output_path, transform)


def video_frame_drop(input_path, output_path, param, seed=0):
    """
    Conceal random frame losses by repeating the last received frame.
    Args:
        input_path (str or Path): Prepared video segment.
        output_path (str or Path): Output with unchanged FPS/frame count/dimensions.
        param (str or float): Frame-loss probability in [0, 1]. The first frame
            is always retained. Loss decisions cover all source frames before
            any model-specific stride or window sampling.
        seed (int): Stable per-file seed supplied by the runner.
    Returns:
        None.
    """
    probability = float(probability_param(param))
    generator = np.random.default_rng(seed)
    previous = None

    def transform(frame, index):
        nonlocal previous
        dropped = generator.random() < probability
        if index == 0 or not dropped:
            previous = frame.copy()
        return previous

    transform_video_file(input_path, output_path, transform)


if __name__ == '__main__':
    # One manual example; change the function/parameter to inspect another operation.
    unit_test_params = {
        'input_path': Path('data/examples/0_25.mp4'),
        'output_path': Path('outputs/perturbation_examples/video_h264.mp4'),
        'param': 23,
    }
    video_h264(**unit_test_params)
    print(f'Saved perturbed video to {unit_test_params["output_path"]}')
