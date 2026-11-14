import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocess.audio_video.audio_video_preprocessor import AudioVideoPreprocessor


class MegaMMDFPreprocessor(AudioVideoPreprocessor):
    def __init__(self, split):
        super(MegaMMDFPreprocessor, self).__init__(dataset_name='Mega-MMDF', split=split)

    def collect_data_attributes(self, data_split):
        """
        Deal with different data labeling format. Output a list of dictionary containing attributes of each data.
        Returns:
            data_list: [list] each element is a dictionary containing attributes of this data.
        """
        with open(os.path.join(self.path['original_dir'], 'Mega-MMDF', f'{data_split}.json'), 'r') as f:
            data_list = json.load(f)

        with open(os.path.join(self.path['original_dir'], 'Mega-MMDF', 'attributes.json'), 'r') as f:
            data_attributes = json.load(f)

        for i in range(len(data_list)):
            data = {
                'label': 1 if 'RARV' not in data_list[i] else 0,
                'video_label': 1 if 'FV' in data_list[i] else 0,
                'audio_label': 1 if 'FA' in data_list[i] else 0,
                'original_path': os.path.join(self.path['original_dir'], data_list[i].replace('\\', '/')),
                'output_path': os.path.join(self.path['preprocessed_dir'], self.dataset_type, data_list[i][:-4].replace('\\', '/')),
            }
            original_file_path = data_list[i].replace('\\', '/')
            data['method'] = ' -> '.join(data_attributes[original_file_path]['method'])
            if data_attributes[original_file_path].get('partial_fake_segments', None) is not None:
                start_frame = int(round(data_attributes[original_file_path]['partial_fake_segments'][0] * self.fps))
                end_frame = int(round(data_attributes[original_file_path]['partial_fake_segments'][1] * self.fps))
                data['partial_fake_type'] = 'audio'
                data['fake_frame_range_list'] = [[start_frame, end_frame + 1]]

            data_list[i] = data

        return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val', 'test'],
                        help='Which split to preprocess')
    parser.add_argument('--num_proc', type=int, default=1, help='number of processes to use')
    args = parser.parse_args()

    preprocessor = MegaMMDFPreprocessor(split=args.split)
    preprocessor.preprocess_all_data(num_proc=args.num_proc)
    preprocessor.assemble_to_json()
