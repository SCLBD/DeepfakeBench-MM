import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocess.audio_video.audio_video_preprocessor import AudioVideoPreprocessor


class PolyGlotFakePreprocessor(AudioVideoPreprocessor):
    def __init__(self, split, split_dir=None):
        self.split_dir = os.path.abspath(split_dir or os.path.join(os.path.dirname(__file__), 'polyglotfake_splits'))
        self._split_data_cache = None
        super(PolyGlotFakePreprocessor, self).__init__(dataset_name='PolyGlotFake', split=split)

    def _load_videos_from_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # The original real and fake annotations use different keys.
        videos = data.get('videos', None)
        if videos is None:
            videos = data.get('video', [])
        return videos

    def _load_split_paths(self):
        split_paths, owners = {}, {}
        for split in ('train', 'val', 'test'):
            split_path = os.path.join(self.split_dir, f'{split}.json')
            with open(split_path, 'r', encoding='utf-8') as f:
                paths = json.load(f)
            if not isinstance(paths, list):
                raise ValueError(f'Expected a list of video paths in {split_path}.')
            split_paths[split] = []
            for path in paths:
                if not isinstance(path, str):
                    raise ValueError(f'Expected a string path in {split_path}: {path!r}')
                path = path.replace('\\', '/')
                parts = path.split('/')
                if (len(parts) < 4 or parts[0] != 'PolyGlotFake'
                        or any(part in ('', '.', '..') for part in parts)
                        or not path.lower().endswith('.mp4')):
                    raise ValueError(f'Invalid PolyGlotFake path in {split_path}: {path!r}')
                if path in owners:
                    raise ValueError(f'Duplicate split path {path!r}: {owners[path]} and {split}.')
                owners[path] = split
                split_paths[split].append(path)
        return split_paths, owners

    def _make_data(self, relative_path, label, method):
        return {
            'video_label': label,
            'audio_label': label,
            'label': label,
            'original_path': os.path.join(self.path['original_dir'], relative_path).replace('\\', '/'),
            'output_path': os.path.join(self.path['preprocessed_dir'], self.dataset_type, relative_path[:-4]).replace('\\', '/'),
            'method': method,
        }

    def collect_data_attributes(self, data_split):
        """Read the published split lists; never reshuffle the raw annotations."""
        if self._split_data_cache is None:
            real_json_path = os.path.join(self.path['original_dir'], 'PolyGlotFake/json_file/real_json_file/all_real_video.json')
            fake_json_path = os.path.join(self.path['original_dir'], 'PolyGlotFake/json_file/fake_Json_file/all_fake_video.json')
            real_videos = self._load_videos_from_json(real_json_path)
            fake_videos = self._load_videos_from_json(fake_json_path)
            split_paths, owners = self._load_split_paths()

            records, real_paths, fake_sources = {}, {}, {}
            for item in real_videos:
                filename = item.get('filename')
                if not filename:
                    continue
                path = f'PolyGlotFake/real/{filename[:2]}/{filename}'
                real_paths[filename] = path
                records[path] = self._make_data(path, 0, 'real')

            for item in fake_videos:
                filename = item['filename']
                source_name = item['raw_video_name']
                if source_name not in real_paths:
                    raise ValueError(f'Unknown real source {source_name!r} for fake video {filename!r}.')
                path = f"PolyGlotFake/fake/to_{item['target_lang']}/{filename}"
                method = f"{item.get('tts_technique')} -> {item.get('sync_tech')}"
                record = self._make_data(path, 1, method)
                if path in records and records[path] != record:
                    raise ValueError(f'Conflicting metadata for {path!r}.')
                records[path] = record
                fake_sources[path] = real_paths[source_name]

            split_data = {}
            for split, paths in split_paths.items():
                split_data[split] = []
                for path in paths:
                    if path not in records:
                        raise ValueError(f'Split video is missing from the raw annotations: {path!r}')
                    if path in fake_sources and owners.get(fake_sources[path]) != split:
                        raise ValueError(f'Fake video and its real source must share a split: {path!r}')
                    split_data[split].append(records[path])
            self._split_data_cache = split_data

        return self._split_data_cache[data_split]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val', 'test'],
                        help='Which split to preprocess')
    parser.add_argument('--num_proc', type=int, default=1, help='number of processes to use')
    parser.add_argument('--split_dir', default=None,
                        help='Split JSON directory; defaults to the bundled polyglotfake_splits directory')
    args = parser.parse_args()

    preprocessor = PolyGlotFakePreprocessor(split=args.split, split_dir=args.split_dir)
    preprocessor.preprocess_all_data(num_proc=args.num_proc)
    preprocessor.assemble_to_json()
