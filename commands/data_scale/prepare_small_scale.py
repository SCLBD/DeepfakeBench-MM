"""Build pair-stratified Mega-MMDF subset manifests from the full manifest."""

import argparse
import json
import math
import random
import re
import tempfile

from collections import defaultdict
from pathlib import Path, PurePosixPath

import yaml


SPLITS = ('train', 'val', 'test')
SKINS = ('Black', 'Brown', 'White', 'Yellow')
GENDERS = ('Female', 'Male')
GROUPS = tuple((split, f'{gender}_{skin}') for split in SPLITS for skin in SKINS for gender in GENDERS)
PIPELINE_DIR = re.compile(r'(?:^|/)pipeline(\d+)/([^/]+)/([^/]+)(?:/|$)')
CLIP_ID = re.compile(r'\d{5}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path-config', type=Path, default=Path('configs/path.yaml'))
    parser.add_argument('--proportion', type=float, nargs='+', default=[1, 2, 5, 10, 20, 50],
                        help='Percentages to retain, for example 1 2 5. Defaults to all six paper settings.')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--pair-json', type=Path, help='Defaults to original_dir/Mega-MMDF/pair.json.')
    inputs.add_argument('--pair-lists', type=Path, help='Use existing <split>_<skin>_<gender>.txt files.')
    parser.add_argument('--seed', type=int, default=0, help='Seed 0 writes unsuffixed manifests.')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing subset manifests.')
    args = parser.parse_args()
    if any(not 0 < value <= 100 for value in args.proportion):
        parser.error('--proportion must be in (0, 100].')
    if args.seed < 0:
        parser.error('--seed must be nonnegative.')
    return args


def load_pairs(pair_json, pair_lists, real_splits):
    """Read the original pair order and keep pairs whose real clips were both prepared."""
    if pair_lists is not None:
        groups = {}
        for split, group in GROUPS:
            gender, skin = group.split('_')
            pair_file = pair_lists / f'{split}_{skin}_{gender}.txt'
            pairs = []
            for line in pair_file.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    left, right = line.split()
                    pairs.append((left, right))
            groups[split, group] = pairs
        return groups

    groups = {key: [] for key in GROUPS}
    with pair_json.open(encoding='utf-8') as source:
        for left, right in json.load(source):
            left_path = PurePosixPath(left.replace('\\', '/'))
            right_path = PurePosixPath(right.replace('\\', '/'))
            group = left_path.parent.name
            if group != right_path.parent.name:
                raise ValueError(f'Pair members belong to different demographic groups: {left}, {right}')
            left_split = real_splits.get((group, left_path.stem))
            right_split = real_splits.get((group, right_path.stem))
            if left_split is None or right_split is None:
                continue
            if left_split != right_split:
                raise ValueError(f'Pair members belong to different splits: {left}, {right}')
            groups[left_split, group].append((left_path.stem, right_path.stem))
    if not any(groups.values()):
        raise ValueError(f'No pairs from {pair_json} matched the prepared real records.')
    return groups


def index_records(source):
    """Index real clip membership, pair-linked rows, and unpaired P22 rows."""
    real_splits = {}
    paired_rows = {}
    unpaired_rows = defaultdict(list)
    for split in SPLITS:
        records = paired_rows[split] = {}
        for index, row in enumerate(source[split]):
            path = row['path'].replace('\\', '/')
            match = PIPELINE_DIR.search(path)
            if match is None:
                raise ValueError(f'Invalid Mega-MMDF manifest path: {path}')
            pipeline_str, group, clip_dir = match.groups()
            pipeline = int(pipeline_str)
            if pipeline == 22:
                unpaired_rows[split, group].append(index)
            elif 0 <= pipeline <= 21:
                if pipeline == 0:
                    previous_split = real_splits.setdefault((group, clip_dir), split)
                    if previous_split != split:
                        raise ValueError(f'Real clip appears in multiple splits: {group}/{clip_dir}')
                    clip_ids = (clip_dir,)
                else:
                    clip_ids = tuple(CLIP_ID.findall(clip_dir))
                records[index] = (group, clip_ids)
            else:
                raise ValueError(f'Unexpected pipeline in manifest path: {path}')
    return real_splits, paired_rows, unpaired_rows


def prefix_count(size, proportion):
    """Round down and retain at least one unit from each nonempty group."""
    return max(1, math.floor(size * proportion)) if size else 0


def build_subset(source, paired_rows, shuffled_pairs, shuffled_unpaired, proportion):
    """Keep linked records for selected pairs and sample P22 independently."""
    selected_ids = {
        key: {clip_id for pair in pairs[:prefix_count(len(pairs), proportion)] for clip_id in pair}
        for key, pairs in shuffled_pairs.items()
    }
    unpaired_indices = {split: set() for split in SPLITS}
    for (split, group), rows in shuffled_unpaired.items():
        unpaired_indices[split].update(rows[:prefix_count(len(rows), proportion)])

    output = {}
    for split in SPLITS:
        selected_by_group = {group: ids for (kind, group), ids in selected_ids.items() if kind == split}
        kept = unpaired_indices[split]
        for index, (group, clip_ids) in paired_rows[split].items():
            if any(clip_id in selected_by_group[group] for clip_id in clip_ids):
                kept.add(index)
        output[split] = [source[split][index] for index in sorted(kept)]
    return output


def write_manifest(path, data, overwrite):
    """Publish a complete JSON file after writing it in the same directory."""
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(data, file, ensure_ascii=False)
        except Exception:
            file.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    with args.path_config.open(encoding='utf-8') as file:
        config = yaml.safe_load(file)
    manifest_dir = Path(config['json_dir']) / 'audio-video'
    pair_json = args.pair_json or Path(config['original_dir']) / 'Mega-MMDF' / 'pair.json'
    source_path = manifest_dir / 'Mega-MMDF.json'
    suffix = f'-{args.seed}' if args.seed else ''
    destinations = {value: manifest_dir / f'Mega-MMDF-{value:g}%{suffix}.json' for value in args.proportion}
    existing = [path.name for path in destinations.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f'Output exists: {", ".join(existing)}. Use --overwrite to regenerate it.')

    with source_path.open(encoding='utf-8') as file:
        source = json.load(file)
    real_splits, paired_rows, unpaired_rows = index_records(source)
    pairs = load_pairs(pair_json, args.pair_lists, real_splits)
    rng = random.Random(args.seed)
    for key in GROUPS:
        rng.shuffle(pairs[key])
    for key in sorted(unpaired_rows):
        rng.shuffle(unpaired_rows[key])

    for value, path in destinations.items():
        subset = build_subset(source, paired_rows, pairs, unpaired_rows, value / 100)
        write_manifest(path, subset, args.overwrite)
        counts = ', '.join(f'{split}={len(subset[split]):,}' for split in SPLITS)
        print(f'{path} ({counts})', flush=True)


if __name__ == '__main__':
    main()
