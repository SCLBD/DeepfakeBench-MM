"""Generate offline perturbations from the shared preprocessed audio-video layout."""

import argparse
import json
import os
import shutil
import sys
import yaml

from pathlib import Path, PureWindowsPath

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.utils.perturbations.audio_video_perturbations import AUDIO_VIDEO_OPERATIONS, get_perturbation_directory
from datasets.utils.perturbations.common_perturbations import copy_file, stable_file_seed


OPERATIONS = AUDIO_VIDEO_OPERATIONS


def parse_args(argv=None):
    """
    Parse an operation and its actual parameter, defaulting to test data.
    Args:
        argv (list or None): Optional CLI argument list; None reads sys.argv.
    Returns:
        args (argparse.Namespace): Validated operation, normalized parameter,
            dataset/split selection, path configuration and stochastic options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--type', choices=tuple(OPERATIONS), required=True, help='Operation name.')
    parser.add_argument('--param', required=True, help='Actual parameter value; no severity presets.')
    parser.add_argument('--datasets', nargs='+', help='Dataset names; default: all JSON files under json_dir/audio-video.')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test', help='Split to process; default: test.')
    parser.add_argument(
        '--path_config',
        type=Path,
        default=Path(project_root) / 'configs/path.yaml',
        help='Path settings; default: configs/path.yaml.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='Optional explicit output root; default: preprocessed_dir/audio-video_<type>_<param>.',
    )
    parser.add_argument('--seed', type=int, default=0, help='Base seed for file-based random perturbations; default: 0.')
    parser.add_argument(
        '--packet_duration_ms', type=float, default=20, help='Audio dropout packet length in milliseconds; default: 20.'
    )
    parser.add_argument(
        '--fade_duration_ms',
        type=float,
        default=2,
        help='Audio dropout boundary fade in milliseconds; default: 2; 0 disables it.',
    )
    args = parser.parse_args(argv)
    try:
        args.param = OPERATIONS[args.type][2](args.param)
    except ValueError as error:
        parser.error(str(error))
    return args


def resolve_root(value):
    """
    Resolve a configured root relative to the repository rather than the runner.
    Args:
        value (str or Path): Absolute path or path relative to project_root.
    Returns:
        root (Path): Resolved absolute path.
    """
    value = Path(value).expanduser()
    return (value if value.is_absolute() else Path(project_root) / value).resolve()


def collect_split_files(json_root, source_root, dataset_names=None, split='test'):
    """
    Collect unique files from the selected standardized dataset split records.
    Args:
        json_root (Path): Directory containing audio-video dataset JSON files.
        source_root (Path): Clean preprocessed audio-video root.
        dataset_names (list or None): Dataset names without .json; None selects all.
        split (str): Split key to traverse; test by default.
    Returns:
        files (list): Sorted (kind, relative_path) pairs, where kind is 'video',
            'audio' or 'landmarks'. Shared WAVs and repeated video references
            appear once. Existing matching NPY files are copied as auxiliary
            source annotations; no landmark detection is rerun.
    """
    json_root, source_root = Path(json_root), Path(source_root)
    if not json_root.is_dir():
        raise FileNotFoundError(f'Dataset JSON directory does not exist: {json_root}.')
    names = (
        sorted(set(dataset_names))
        if dataset_names is not None
        else sorted(path.stem for path in json_root.glob('*.json'))
    )
    if not names:
        raise ValueError(f'No datasets selected or no JSON files found under {json_root}.')
    files = set()
    for name in names:
        if not name or Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError(f'Expected a dataset name without directory components: {name!r}.')
        json_path = json_root / f'{name}.json'
        with json_path.open('r', encoding='utf-8') as handle:
            records = json.load(handle)[split]
        print(f'{name}: {len(records)} {split} records', flush=True)
        for record in records:
            relative = record['path'].replace('\\', '/')
            video_path = Path(relative)
            if video_path.is_absolute() or PureWindowsPath(relative).drive or '..' in video_path.parts:
                raise ValueError(f'Dataset paths must be relative to the clean root: {relative!r}.')
            files.add(('video', video_path))
            files.add(('audio', Path(f'{video_path.parent.as_posix()}.wav')))
            landmarks = video_path.with_suffix('.npy')
            if (source_root / landmarks).is_file():
                files.add(('landmarks', landmarks))
    return sorted(files, key=lambda item: (item[1].as_posix(), item[0]))


def generate(args):
    """
    Traverse selected split files and generate one offline perturbation version.
    Args:
        args (argparse.Namespace): Validated parse_args result. Reads path.yaml
            automatically; files already present at their final paths are skipped.
    Returns:
        statistics (dict): Generated, copied, skipped and failed file counts.
            File/encoder failures are reported while other files continue. Unexpected programming
            errors propagate immediately. No Dataset or detector is constructed.
    """
    with args.path_config.open('r', encoding='utf-8') as handle:
        path_config = yaml.safe_load(handle)
    preprocessed_root = resolve_root(path_config['preprocessed_dir'])
    source_root = preprocessed_root / 'audio-video'
    json_root = resolve_root(path_config['json_dir']) / 'audio-video'
    output_root = (
        resolve_root(args.output_dir)
        if args.output_dir
        else preprocessed_root / get_perturbation_directory(args.type, args.param)
    )
    if not source_root.is_dir():
        raise FileNotFoundError(f'Clean preprocessed directory does not exist: {source_root}.')
    if output_root.resolve() == source_root.resolve():
        raise ValueError('Output directory must be different from the clean source root.')
    for executable in ['ffmpeg', 'ffprobe']:
        if shutil.which(executable) is None:
            raise FileNotFoundError(f'{executable} is not available in PATH.')
    files = collect_split_files(json_root, source_root, args.datasets, args.split)
    target_kind, operation, _ = OPERATIONS[args.type]
    print(
        f'Input: {source_root}\nOutput: {output_root}\n'
        f'Operation: {args.type}; parameter: {args.param}; files: {len(files)}',
        flush=True,
    )
    statistics = {'generated': 0, 'copied': 0, 'skipped': 0, 'failed': 0}
    for index, (kind, relative_path) in enumerate(files, 1):
        input_path, output_path = source_root / relative_path, output_root / relative_path
        if output_path.is_file():
            statistics['skipped'] += 1
            continue
        try:
            if kind == target_kind:
                # Pass only options that this operation actually uses.
                options = {}
                if args.type in ['video_frame_drop', 'audio_dropout']:
                    options['seed'] = stable_file_seed(args.seed, relative_path)
                if args.type == 'audio_dropout':
                    options.update(packet_duration_ms=args.packet_duration_ms, fade_duration_ms=args.fade_duration_ms)
                operation(input_path, output_path, args.param, **options)
                statistics['generated'] += 1
            else:
                copy_file(input_path, output_path)
                statistics['copied'] += 1
        except (OSError, RuntimeError) as error:
            statistics['failed'] += 1
            print(f'FAILED {relative_path}: {error}', file=sys.stderr, flush=True)
        if index % 100 == 0:
            print(f'{index}/{len(files)} files: {statistics}', flush=True)
    print(json.dumps(statistics, ensure_ascii=False), flush=True)
    if statistics['failed']:
        raise RuntimeError(f'{statistics["failed"]} files failed; rerun the same command to retry missing outputs.')
    return statistics


def main():
    """
    Run the audio-video offline generator from the command line.
    Returns:
        None.
    """
    generate(parse_args())


if __name__ == '__main__':
    main()
