import argparse
import json
import os
import sys

from pathlib import Path
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocess.audio_video.audio_video_preprocessor import AudioVideoPreprocessor


class TalkingHeadBenchPreprocessor(AudioVideoPreprocessor):
    def __init__(self, split):
        super(TalkingHeadBenchPreprocessor, self).__init__(dataset_name='TalkingHeadBench', split=split)

    def collect_data_attributes(self, data_split):
        """
        Collect TalkingHeadBench data attributes for the given split.

        The dataset contains two kinds of data:
            1) Fake data (real audio, fake video), located under 'TalkingHeadBench/fake'. The split is decided by the
               file path:
                 - 'additional_dataset/<method>/*.mp4': all belong to the 'test' split, folder name is the 'method'.
                 - '<method>/{train,val,test}/*.mp4': folder name is the 'method', the sub-folder name is the split.
            2) Real data (real audio, real video), whose split is defined in
               'TalkingHeadBench/real/real_dataset_split.json'. Two source datasets are used:
                 - CelebV-HQ: single mp4 file (audio + video) under 'CelebV-HQ/35666'.
                 - FaceForensics++: video under 'FaceForensics++/original_sequences/youtube/raw/videos' and audio under
                   'TalkingHeadBench/audio/ff++' (same stem, '.wav' suffix). Only data whose audio and video files
                   both exist are kept.

        Returns:
            data_list: [list] each element is a dictionary containing attributes of this data.
        """
        data_list = []
        data_list.extend(self._collect_fake_data(data_split))
        data_list.extend(self._collect_real_data(data_split))
        return data_list

    def _collect_fake_data(self, data_split):
        original_dir = self.path['original_dir']
        fake_root = os.path.join(original_dir, 'TalkingHeadBench', 'fake')

        data_list = []
        for method_name in sorted(os.listdir(fake_root)):
            method_dir = os.path.join(fake_root, method_name)
            if not os.path.isdir(method_dir):
                continue

            if method_name == 'additional_dataset':
                # every data in additional_dataset is used for testing only
                if data_split != 'test':
                    continue
                for sub_method in sorted(os.listdir(method_dir)):
                    sub_method_dir = os.path.join(method_dir, sub_method)
                    if not os.path.isdir(sub_method_dir):
                        continue
                    for file_name in sorted(os.listdir(sub_method_dir)):
                        if not file_name.endswith('.mp4'):
                            continue
                        rel_path = os.path.join('TalkingHeadBench/fake/additional_dataset', sub_method, file_name).replace('\\', '/')
                        data_list.append(self._make_fake_data(original_dir, rel_path, sub_method))
            else:
                split_dir = os.path.join(method_dir, data_split)
                if not os.path.isdir(split_dir):
                    continue
                for file_name in sorted(os.listdir(split_dir)):
                    if not file_name.endswith('.mp4'):
                        continue
                    rel_path = os.path.join('TalkingHeadBench/fake', method_name, data_split, file_name).replace('\\', '/')
                    data_list.append(self._make_fake_data(original_dir, rel_path, method_name))

        return data_list

    def _make_fake_data(self, original_dir, rel_path, method):
        return {
            'video_label': 1,
            'audio_label': 0,
            'label': 1,
            'original_path': os.path.join(original_dir, rel_path).replace('\\', '/'),
            'output_path': os.path.join(self.path['preprocessed_dir'], self.dataset_type, rel_path[:-4]).replace('\\', '/'),
            'method': method,
        }

    def _collect_real_data(self, data_split):
        original_dir = self.path['original_dir']
        split_map = {'train': 'Train', 'val': 'Val', 'test': 'Test'}

        json_path = os.path.join(original_dir, 'TalkingHeadBench', 'real', 'real_dataset_split.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            real_split = json.load(f)

        group = real_split[split_map[data_split]]
        data_list = []

        # CelebV-HQ: single mp4 file containing both audio and video
        for file_name in group.get('CelebV-HQ', []):
            video_path = os.path.join(original_dir, 'CelebV-HQ', '35666', file_name)
            if not os.path.exists(video_path):
                continue
            rel_out = os.path.join('TalkingHeadBench', 'real', 'CelebV-HQ', file_name[:-4]).replace('\\', '/')
            data_list.append({
                'video_label': 0,
                'audio_label': 0,
                'label': 0,
                'original_path': video_path.replace('\\', '/'),
                'output_path': os.path.join(self.path['preprocessed_dir'], self.dataset_type, rel_out).replace('\\', '/'),
                'method': 'real',
            })

        # FaceForensics++: video and audio are provided as separate files
        for file_name in group.get('FaceForensics++', []):
            video_path = os.path.join(original_dir, 'FaceForensics++/original_sequences/youtube/raw/videos', file_name)
            audio_path = os.path.join(original_dir, 'TalkingHeadBench', 'audio', 'ff++', file_name[:-4] + '.wav')
            if not (os.path.exists(video_path) and os.path.exists(audio_path)):
                continue
            rel_out = os.path.join('TalkingHeadBench', 'real', 'FaceForensics++', file_name[:-4]).replace('\\', '/')
            data_list.append({
                'video_label': 0,
                'audio_label': 0,
                'label': 0,
                'original_path_video': video_path.replace('\\', '/'),
                'original_path_audio': audio_path.replace('\\', '/'),
                'output_path': os.path.join(self.path['preprocessed_dir'], self.dataset_type, rel_out).replace('\\', '/'),
                'method': 'real',
            })

        return data_list

    def assemble_data_split(self, data_split, data_list):
        """
        Assemble specified data split into a list of dictionaries. This overrides the base implementation to also
        support data that provides separate audio and video files (i.e., FaceForensics++ real data) instead of a
        single 'original_path'.
        Args:
            data_split: [str] data split name, options: ['train', 'val', 'test']
            data_list: [list] list of prefetched data information list.
        Returns:
            json_list: [list] list of dictionaries, which contains the final data information.
        """
        json_list = []
        for data in tqdm(data_list, desc=f'Assembling {data_split}', dynamic_ncols=True):
            if not os.path.exists(data['output_path'] + '.done'):  # fail to preprocess
                continue

            segment_list = list(Path(data['output_path']).glob('*.mp4'))
            assert len(segment_list) == 1
            if data.get('original_path'):
                data['original_path'] = str(Path(data['original_path']).relative_to(self.path['original_dir']))
            else:
                data['original_path_video'] = str(Path(data['original_path_video']).relative_to(self.path['original_dir']))
                data['original_path_audio'] = str(Path(data['original_path_audio']).relative_to(self.path['original_dir']))
            data['path'] = str(segment_list[0].relative_to(Path(self.path['preprocessed_dir']) / 'audio-video'))
            del data['output_path']
            json_list.append(data)

        return json_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val', 'test'],
                        help='Which split to preprocess')
    parser.add_argument('--num_proc', type=int, default=1, help='number of processes to use')
    args = parser.parse_args()

    preprocessor = TalkingHeadBenchPreprocessor(split=args.split)
    preprocessor.preprocess_all_data(num_proc=args.num_proc)
    preprocessor.assemble_to_json()
