"""Copy supplied demographic labels into a prepared dataset manifest before inference."""

import argparse
import json
import os
import re
import shutil
import tempfile
import yaml

from collections import Counter
from pathlib import Path


project_root = str(Path(__file__).resolve().parents[2])
MEGA_GROUP = re.compile(r'^(Male|Female)_(White|Black|Yellow|Brown)$')
FAKEAV_RACES = {'African', 'Asian (East)', 'Asian (South)', 'Caucasian (American)', 'Caucasian (European)'}
FAKEAV_GENDERS = {'men', 'women'}


def group_from_path(path, dataset):
    """Read existing dataset directory labels, without estimating attributes from media.

    Args:
        path (str): Original clip path, with either slash convention.
        dataset (str): Mega-MMDF or FakeAVCeleb. Mega-MMDF uses a directory such as
            Female_Black; FakeAVCeleb uses separate race/gender directories.

    Returns:
        str: Existing research convention: 'Black - Female' for Mega-MMDF or
            'African - men' for FakeAVCeleb.
    """
    parts = path.replace('\\', '/').split('/')
    if dataset == 'Mega-MMDF':
        groups = {match.groups() for part in parts if (match := MEGA_GROUP.fullmatch(part))}
        if 'Mega-MMDF' in parts and len(groups) == 1:
            gender, skin_tone = groups.pop()
            return f'{skin_tone} - {gender}'
    elif dataset == 'FakeAVCeleb':
        races, genders = set(parts) & FAKEAV_RACES, set(parts) & FAKEAV_GENDERS
        if {'FakeAVCeleb', 'FakeAVCeleb_v1.2'} & set(parts) and len(races) == len(genders) == 1:
            return f'{races.pop()} - {genders.pop()}'
    else:
        raise ValueError(f'No directory-label rule for {dataset!r}; provide --mapping_json.')
    raise ValueError(f'No unambiguous {dataset} directory annotation in {path!r}.')


def annotate_demographic_groups(manifest, dataset, split='test', mapping=None, key_field='original_path', overwrite=False):
    """Add group labels to selected records, validating coverage before modifying the manifest.

    Args:
        manifest (dict): Prepared train/val/test JSON object; unrelated fields are retained.
        dataset (str): Dataset name used for the built-in directory-label rule when mapping is None.
        split (str): train, val, test, or all. Only selected split records are changed.
        mapping (dict | None): Exact {record[key_field]: group_string} annotations.
            Backslashes are normalized to slashes; basenames or path prefixes are not guessed.
        key_field (str): Record field to look up; default original_path, optionally path or another explicit field.
        overwrite (bool): Replace existing group values using the supplied source. False preserves valid existing labels.

    Returns:
        dict: Counts for records, added, preserved, overwritten, and resulting demographic groups.
            Sample order, all other record fields, and unselected splits remain unchanged.
    """
    if not isinstance(manifest, dict):
        raise ValueError('Expected an assembled dataset JSON object with train/val/test lists.')
    splits = ('train', 'val', 'test') if split == 'all' else (split,)
    if split not in ('train', 'val', 'test', 'all'):
        raise ValueError('split must be train, val, test, or all.')
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise ValueError('--mapping_json must contain an object mapping exact paths to group strings.')
        normalized = {}
        for key, group in mapping.items():
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f'Mapping value for {key!r} must be a nonempty group string.')
            key = key.replace('\\', '/')
            if key in normalized and normalized[key] != group:
                raise ValueError(f'Conflicting annotations for normalized path {key!r}.')
            normalized[key] = group
        mapping = normalized

    counts = {'records': 0, 'added': 0, 'preserved': 0, 'overwritten': 0}
    groups, updates, errors = Counter(), [], []
    error_count = 0
    for selected_split in splits:
        if selected_split not in manifest or not isinstance(manifest[selected_split], list):
            raise ValueError(f'Manifest must contain a {selected_split!r} list.')
        for index, record in enumerate(manifest[selected_split]):
            counts['records'] += 1
            existing = record.get('demographic_group')
            if existing is not None and (not isinstance(existing, str) or not existing.strip()):
                raise ValueError(f'{selected_split}[{index}] has an invalid demographic_group value.')
            if existing is not None and not overwrite:
                counts['preserved'] += 1
                groups[existing] += 1
                continue
            path = record.get(key_field)
            try:
                if not isinstance(path, str) or not path:
                    raise ValueError(f'Missing {key_field!r}.')
                if mapping is None:
                    group = group_from_path(path, dataset)
                else:
                    key = path.replace('\\', '/')
                    if key not in mapping:
                        raise ValueError(f'No mapping entry for {key_field}={path!r}.')
                    group = mapping[key]
            except ValueError as error:
                error_count += 1
                if len(errors) < 10:
                    errors.append(f'{selected_split}[{index}]: {error}')
                continue
            groups[group] += 1
            if existing is None:
                counts['added'] += 1
                updates.append((record, group))
            elif existing != group:
                counts['overwritten'] += 1
                updates.append((record, group))
            else:
                counts['preserved'] += 1
    if not counts['records']:
        raise ValueError('Selected split contains no records.')
    if error_count:
        raise ValueError(f'Cannot annotate {error_count} records; no output was written.\n' + '\n'.join(errors))
    for record, group in updates:
        record['demographic_group'] = group
    return {**counts, 'groups': dict(sorted(groups.items()))}


def update_manifest(input_json, dataset, output_json=None, in_place=False, split='test', mapping_json=None,
                    key_field='original_path', overwrite=False):
    """Load annotations and write a prepared manifest, optionally preserving a first in-place backup.

    Args:
        input_json (str | Path): Prepared dataset manifest, not a prediction file.
        dataset (str): Dataset name for directory annotations; arbitrary names work with mapping_json.
        output_json (str | Path | None): Explicit output file, mutually exclusive with in_place.
        in_place (bool): Replace input_json and retain its first backup as <input>.bak.
        split (str): Split to annotate; defaults to test.
        mapping_json (str | Path | None): Optional exact path-to-group JSON object.
        key_field (str): Record field used to join annotations; defaults to original_path.
        overwrite (bool): Recompute existing labels only when explicitly requested.

    Returns:
        dict: Annotation counts and the output path. Existing in-place backup files are never overwritten.
    """
    source = Path(input_json).resolve()
    if bool(output_json) == bool(in_place):
        raise ValueError('Choose exactly one of --output_json or --in_place.')
    destination = source if in_place else Path(output_json).resolve()
    if destination == source and not in_place:
        raise ValueError('Use --in_place explicitly to replace the input manifest.')
    with source.open('r', encoding='utf-8') as stream:
        manifest = json.load(stream)
    mapping = None
    if mapping_json is not None:
        with open(mapping_json, 'r', encoding='utf-8') as stream:
            mapping = json.load(stream)
    result = annotate_demographic_groups(manifest, dataset, split, mapping, key_field, overwrite)
    if in_place and not result['added'] and not result['overwritten']:
        return {**result, 'output_json': str(destination)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if in_place:
        backup = source.with_name(source.name + '.bak')
        if not backup.exists():
            shutil.copy2(source, backup)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=destination.parent,
                                         prefix='.' + destination.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, indent=4)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            shutil.copymode(destination, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {**result, 'output_json': str(destination)}


def parse_args(arguments=None):
    """Parse dataset, annotation-source, and output choices; optionally accept an explicit argument list."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='Prepared dataset name; built-in labels: Mega-MMDF/FakeAVCeleb.')
    parser.add_argument('--input_json', help='Prepared manifest; defaults to json_dir/audio-video/<dataset>.json.')
    parser.add_argument('--path_config', default=os.path.join(project_root, 'configs/path.yaml'),
                        help='Path configuration used when --input_json is omitted.')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--mapping_json', help='Explicit JSON object: exact record path -> demographic group.')
    parser.add_argument('--key_field', default='original_path', help='Record field for annotation lookup.')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing group labels using the annotation source.')
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument('--output_json', help='Output manifest path; use --in_place to replace the input.')
    output.add_argument('--in_place', action='store_true', help='Update the input and retain its first .bak backup.')
    return parser.parse_args(arguments)


def main(arguments=None):
    """Resolve the configured manifest and annotate it before detector inference; return the printed summary."""
    args = parse_args(arguments)
    input_json = args.input_json
    if input_json is None:
        with open(args.path_config, 'r', encoding='utf-8') as stream:
            paths = yaml.safe_load(stream)
        json_root = os.path.join(project_root, paths['json_dir'])
        input_json = os.path.join(json_root, 'audio-video', args.dataset + '.json')
    result = update_manifest(input_json, args.dataset, args.output_json, args.in_place, args.split,
                             args.mapping_json, args.key_field, args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
