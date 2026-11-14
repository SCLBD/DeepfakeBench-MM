"""Shared parameter validation, FFmpeg execution and atomic output writing."""

import hashlib
import os
import re
import shutil
import subprocess
import tempfile

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path


def numeric_param(value):
    """
    Normalize a finite numeric argument without selecting a severity preset.
    Args:
        value (str, int or float): Actual operation parameter supplied by the user.
    Returns:
        parameter (str): Canonical decimal text, also safe as a directory suffix.
            For example, '0.10' and '1e-1' both become '0.1'.
    """
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f'Expected a numeric parameter, received {value!r}.') from error
    if not number.is_finite():
        raise ValueError(f'Expected a finite parameter, received {value!r}.')
    return '0' if number == 0 else format(number.normalize(), 'f')


def crf_param(value):
    """
    Validate an actual H.264 CRF parameter against the encoder's numeric range.
    Args:
        value (str, int or float): CRF value; this is not a benchmark strength ID.
    Returns:
        parameter (str): Canonical CRF text in the range [0, 51].
    """
    parameter = numeric_param(value)
    if not 0 <= float(parameter) <= 51:
        raise ValueError('H.264 CRF must be between 0 and 51.')
    return parameter


def ratio_param(value):
    """
    Validate a spatial downscaling ratio.
    Args:
        value (str, int or float): Multiplier applied to width and height.
    Returns:
        parameter (str): Canonical ratio in the range (0, 1].
    """
    parameter = numeric_param(value)
    if not 0 < float(parameter) <= 1:
        raise ValueError('Downscaling ratio must be greater than 0 and at most 1.')
    return parameter


def probability_param(value):
    """
    Validate a frame-loss or packet-loss probability.
    Args:
        value (str, int or float): Probability of dropping one frame or packet.
    Returns:
        parameter (str): Canonical probability in the range [0, 1].
    """
    parameter = numeric_param(value)
    if not 0 <= float(parameter) <= 1:
        raise ValueError('Drop probability must be between 0 and 1.')
    return parameter


def bitrate_param(value):
    """
    Validate an explicit FFmpeg audio bitrate.
    Args:
        value (str): Positive bitrate in bits/s, or with a k/M SI suffix.
    Returns:
        parameter (str): Normalized bitrate text, e.g., '64000', '64k' or '1M'.
    """
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([kKmM]?)', str(value).strip())
    if match is None or Decimal(match[1]) <= 0:
        raise ValueError('Bitrate must be positive, for example 64000, 64k or 1M.')
    suffix = {'': '', 'k': 'k', 'K': 'k', 'm': 'M', 'M': 'M'}[match[2]]
    return numeric_param(match[1]) + suffix


def run_ffmpeg(arguments, input_data=None):
    """
    Run FFmpeg with checked errors and optional binary stdin.
    Args:
        arguments (list): FFmpeg arguments excluding the executable name.
        input_data (bytes or None): Optional raw signal bytes supplied on stdin.
    Returns:
        output (bytes): Captured stdout, used for raw audio decoding when needed.
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-threads', '1'] + arguments,
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.decode('utf-8', errors='replace').strip()) from error
    return result.stdout


@contextmanager
def atomic_output(output_path):
    """
    Write to a temporary sibling and publish only a successfully completed file.
    Args:
        output_path (str or Path): Final output file. Its extension is retained.
    Yields:
        temporary_path (Path): Unique temporary filename in the output directory.
            On successful exit, it atomically replaces the final output file.
            Failed and interrupted runs never expose this path as a final file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f'.{output_path.stem}.', suffix=f'.tmp{output_path.suffix}', dir=output_path.parent)
    os.close(descriptor)
    temporary_path = Path(name)
    try:
        yield temporary_path
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def stable_file_seed(seed, relative_path):
    """
    Derive a file-based seed independent of dataset traversal order and model sampling.
    Args:
        seed (int): User-specified base seed.
        relative_path (str or Path): Prepared file path relative to the clean data directory.
    Returns:
        file_seed (int): Reproducible unsigned 64-bit seed for NumPy random generators.
    """
    digest = hashlib.sha256(Path(relative_path).as_posix().encode('utf-8')).digest()
    return (seed + int.from_bytes(digest[:8], 'big')) % (2**64)


def copy_file(input_path, output_path):
    """
    Copy an unaffected file and publish its final filename only after copying succeeds.
    Args:
        input_path (str or Path): Clean stream or annotation file.
        output_path (str or Path): Matching destination in the perturbed data directory.
    Returns:
        None.
    """
    with atomic_output(output_path) as temporary_path:
        shutil.copy2(input_path, temporary_path)
