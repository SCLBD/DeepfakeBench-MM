"""Build the Mega-MMDF P0 + Pi manifests used in cross-pipeline experiments."""

import argparse
import heapq
import json
import os
import re
import tempfile

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train', 'val', 'test')
PIPELINES = tuple(range(1, 23))
PIPELINE_PART = re.compile(r'pipeline(0|[1-9][0-9]*)')


def group_records(manifest):
    """Index each source split by its exact pipeline directory, in source order."""
    if not isinstance(manifest, dict):
        raise ValueError('The source manifest must be an object with train, val and test lists.')

    grouped = {}
    for split in SPLITS:
        records = manifest.get(split)
        if not isinstance(records, list):
            raise ValueError(f'The source manifest must contain a {split!r} list.')
        groups = {pipeline: [] for pipeline in (0, *PIPELINES)}
        for index, record in enumerate(records):
            path = record['path']
            matches = [match for part in path.replace('\\', '/').split('/') if (match := PIPELINE_PART.fullmatch(part))]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one pipeline directory in {path!r}.')
            pipeline = int(matches[0][1])
            if pipeline not in groups or record['label'] != int(pipeline != 0):
                raise ValueError(f'Invalid pipeline or overall label in {path!r}.')
            groups[pipeline].append((index, record))
        grouped[split] = groups
    return grouped


def write_manifest(manifest, path, overwrite):
    """Publish a complete JSON file, only replacing it on request."""
    stream = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=4)
            stream.write('\n')
        if overwrite:
            os.replace(temporary, path)
        elif os.name == 'nt':
            os.rename(temporary, path)  # Windows rename rejects an existing destination.
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path-config', type=Path, default=PROJECT_ROOT / 'configs/path.yaml')
    parser.add_argument('--source-json', type=Path, help='Prepared Mega-MMDF.json. Defaults to json_dir/audio-video/Mega-MMDF.json.')
    parser.add_argument('--output-dir', type=Path, help='Destination directory. Defaults to json_dir/audio-video.')
    parser.add_argument('--pipelines', type=int, nargs='+', choices=PIPELINES, default=PIPELINES,
                        help='Fake pipeline IDs to prepare. Defaults to 1 through 22.')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing pipeline manifests.')
    args = parser.parse_args()

    if len(args.pipelines) != len(set(args.pipelines)):
        parser.error('Provide each pipeline ID once.')
    if args.source_json is None or args.output_dir is None:
        import yaml

        with args.path_config.open(encoding='utf-8') as stream:
            json_directory = PROJECT_ROOT / Path(yaml.safe_load(stream)['json_dir']).expanduser() / 'audio-video'
    source = args.source_json or json_directory / 'Mega-MMDF.json'
    output = args.output_dir or json_directory
    destinations = {pipeline: output / f'Mega-MMDF-pipeline{pipeline}.json' for pipeline in args.pipelines}

    for destination in destinations.values():
        if destination.resolve() == source.resolve():
            parser.error(f'Output would replace the source manifest: {destination}')
        if destination.exists() and not args.overwrite:
            parser.error(f'{destination} already exists. Use --overwrite to replace the selected manifests.')

    with source.open(encoding='utf-8') as stream:
        grouped = group_records(json.load(stream))
    for split in SPLITS:
        for pipeline in (0, *args.pipelines):
            if not grouped[split][pipeline]:
                parser.error(f'The {split} split has no records from P{pipeline}.')

    output.mkdir(parents=True, exist_ok=True)
    for pipeline, destination in destinations.items():
        manifest = {
            split: [record for _, record in heapq.merge(grouped[split][0], grouped[split][pipeline])]
            for split in SPLITS
        }
        try:
            write_manifest(manifest, destination, args.overwrite)
        except FileExistsError:
            parser.error(f'{destination} already exists. Use --overwrite to replace the selected manifests.')
        counts = ', '.join(f'{split}={len(records):,}' for split, records in manifest.items())
        print(f'{destination} ({counts})')


if __name__ == '__main__':
    main()
