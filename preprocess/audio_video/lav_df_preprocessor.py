import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocess.audio_video.audio_video_preprocessor import AudioVideoPreprocessor


class LAVDFPreprocessor(AudioVideoPreprocessor):
    def __init__(self, split):
        super(LAVDFPreprocessor, self).__init__(dataset_name='LAV-DF', split=split)

    def collect_data_attributes(self, data_split):
        """
        Deal with different data labeling format. Output a list of dictionary containing attributes of each data.
        Returns:
            data_list: [list] each element is a dictionary containing attributes of this data.
        """
        with open(os.path.join(self.path['original_dir'], 'LAV-DF', f'metadata.json'), 'r') as f:
            data_list = json.load(f)

        if data_split == 'train':
            data_list = [data for data in data_list if data['split'] == 'train']
        elif data_split == 'val':
            data_list = [data for data in data_list if data['split'] == 'dev']
        elif data_split == 'test':
            data_list = [data for data in data_list if data['split'] == 'test']
        else:
            raise NotImplementedError(f'Unsupported data split: {data_split}')

        for i in range(len(data_list)):
            data = {
                'label': 1 if data_list[i]['modify_video'] or data_list[i]['modify_audio'] else 0,
                'video_label': 1 if data_list[i]['modify_video'] else 0,
                'audio_label': 1 if data_list[i]['modify_audio'] else 0,
                'original_path': os.path.join(self.path['original_dir'], 'LAV-DF', data_list[i]['file']).replace('\\', '/'),
                'output_path': os.path.join(self.path['preprocessed_dir'], self.dataset_type, 'LAV-DF', data_list[i]['file'][:-4]).replace('\\', '/')
            }
            # Todo: data['method']

            if len(data_list[i]['fake_periods']) != 0:
                data['partial_fake_type'] = 'audio-video'
                data['fake_frame_range_list'] = []
                for fake_periods in data_list[i]['fake_periods']:
                    start_time_stamp, end_time_stamp = fake_periods
                    start_frame, end_frame = int(round(start_time_stamp * self.fps)), int(round(end_time_stamp * self.fps))
                    data['fake_frame_range_list'].append([start_frame, end_frame + 1])

            data_list[i] = data
        return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val', 'test'],
                        help='Which split to preprocess')
    parser.add_argument('--num_proc', type=int, default=1, help='number of processes to use')
    args = parser.parse_args()

    preprocessor = LAVDFPreprocessor(split=args.split)
    preprocessor.preprocess_all_data(num_proc=args.num_proc)
    preprocessor.assemble_to_json()