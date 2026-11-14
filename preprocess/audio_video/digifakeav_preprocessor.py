import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocess.audio_video.audio_video_preprocessor import AudioVideoPreprocessor


class DigiFakeAVPreprocessor(AudioVideoPreprocessor):
    AUDIO_METHOD = 'CosyVoice 2'

    def __init__(self, split, split_dir=None):
        self.split_dir = os.path.abspath(split_dir or os.path.join(os.path.dirname(__file__), 'digifakeav_splits'))
        super(DigiFakeAVPreprocessor, self).__init__(dataset_name='DigiFakeAV', split=split)

    def collect_data_attributes(self, data_split):
        """Convert a DigiFakeAV split file to the unified audio-video format.

        Each split JSON contains paths relative to ``original_dir``. The first
        component below ``DigiFakeAV`` identifies the modality labels:
            - FVFA: fake video and fake audio
            - FVRA: fake video and real audio
            - real: real video and real audio

        For fake-video folders, the text between the first two underscores is
        the video forgery method.
        """
        split_path = os.path.join(self.split_dir, f'{data_split}.json')
        with open(split_path, 'r', encoding='utf-8') as f:
            relative_paths = json.load(f)

        if not isinstance(relative_paths, list):
            raise ValueError(f'Expected a list of video paths in {split_path}.')

        label_map = {
            'FVFA': (1, 1),
            'FVRA': (1, 0),
            'real': (0, 0),
        }
        data_list = []

        for relative_path in relative_paths:
            if not isinstance(relative_path, str):
                raise ValueError(f'Expected a string path in {split_path}, got {relative_path!r}.')

            normalized_path = os.path.normpath(relative_path)
            path_parts = normalized_path.split(os.sep)
            if len(path_parts) < 3 or path_parts[0] != 'DigiFakeAV':
                raise ValueError(f'Unexpected DigiFakeAV path in {split_path}: {relative_path!r}')

            folder_name = path_parts[1]
            category = folder_name.split('_', 1)[0]
            if category not in label_map:
                raise ValueError(f'Unknown DigiFakeAV category in path: {relative_path!r}')

            video_label, audio_label = label_map[category]
            if category == 'real':
                method = 'real'
            else:
                folder_parts = folder_name.split('_')
                if len(folder_parts) < 3 or not folder_parts[1]:
                    raise ValueError(f'Cannot extract the video method from folder: {folder_name!r}')
                video_method = folder_parts[1]
                method = (f'{self.AUDIO_METHOD} -> {video_method}'
                          if category == 'FVFA' else video_method)

            relative_stem = os.path.splitext(normalized_path)[0]
            data_list.append({
                'video_label': video_label,
                'audio_label': audio_label,
                'label': int(video_label or audio_label),
                'original_path': os.path.join(self.path['original_dir'], normalized_path).replace('\\', '/'),
                'output_path': os.path.join(self.path['preprocessed_dir'], self.dataset_type, relative_stem).replace('\\', '/'),
                'method': method,
            })

        return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val', 'test'],
                        help='Which split to preprocess')
    parser.add_argument('--num_proc', type=int, default=1, help='number of processes to use')
    parser.add_argument('--split_dir', default=None,
                        help='Split JSON directory; defaults to the bundled digifakeav_splits directory')
    args = parser.parse_args()

    preprocessor = DigiFakeAVPreprocessor(split=args.split, split_dir=args.split_dir)
    preprocessor.preprocess_all_data(num_proc=args.num_proc)
    preprocessor.assemble_to_json()
