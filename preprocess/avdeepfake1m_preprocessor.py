import argparse
import json
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocess.base_preprocessor import BasePreprocessor

class AVDeepfake1MPreprocessor(BasePreprocessor):
    def __init__(self, config):
        super().__init__(config, 'AVDeepfake1M')

    def collect_data_attributes(self, data_split):
        """
        Deal with different data labeling format. Output a list of dictionary containing attributes of each data.
        Returns:
            data_list: [list] each element is a dictionary containing attributes of this data.
        """
        if data_split == 'train':
            with open(os.path.join(self.config['original_dir'], 'AVDeepfake1M', f'train_metadata.json'), 'r') as f:
                data_list = json.load(f)

            for i in range(len(data_list)):
                save_root = self.config['transcode_dir'] if self.config.get('transcode', False) else self.config['preprocess_dir']
                data = {
                    'transcode': self.config.get('transcode', False),
                    'FPS': self.config.get('FPS', 25),
                    'sample_rate': self.config.get('sample_rate', 16000),
                    'duration': self.config.get('length', 1.0),
                    'clip_amount': self.config.get('clip_amount', 1),
                    'label': 1 if data_list[i]['modify_type'] in ['visual_modified', 'audio_modified', 'both_modified']
                               else 0,
                    'video_label': 1 if data_list[i]['modify_type'] in ['visual_modified', 'both_modified'] else 0,
                    'audio_label': 1 if data_list[i]['modify_type'] in ['audio_modified', 'both_modified'] else 0,
                    'original_path': os.path.join(self.config['original_dir'],
                                                  'AVDeepfake1M', 'train', data_list[i]['file']),
                    'video_path': os.path.join(save_root, 'AVDeepfake1M', 'train', data_list[i]['file']),
                    'audio_path': os.path.join(save_root, 'AVDeepfake1M', 'train', data_list[i]['file'][:-4] + '.wav')
                }
                # Todo: data['method']

                if len(data_list[i]['fake_segments']) != 0:
                    data['fake_segments'] = data_list[i]['fake_segments']

                data_list[i] = data
        else:   # since no test label released, use val data as test data
            with open(os.path.join(self.config['original_dir'], 'AVDeepfake1M', f'val_metadata.json'), 'r') as f:
                data_list = json.load(f)

            for i in range(len(data_list)):
                save_root = self.config['transcode_dir'] if self.config.get('transcode', False) else self.config['preprocess_dir']
                data = {
                    'transcode': self.config.get('transcode', False),
                    'FPS': self.config.get('FPS', 25),
                    'sample_rate': self.config.get('sample_rate', 16000),
                    'duration': self.config.get('length', 1.0),
                    'clip_amount': self.config.get('clip_amount', 1),
                    'label': 1 if data_list[i]['modify_type'] in ['visual_modified', 'audio_modified', 'both_modified']
                               else 0,
                    'video_label': 1 if data_list[i]['modify_type'] in ['visual_modified', 'both_modified'] else 0,
                    'audio_label': 1 if data_list[i]['modify_type'] in ['audio_modified', 'both_modified'] else 0,
                    'original_path': os.path.join(self.config['original_dir'],
                                                  'AVDeepfake1M', 'val', data_list[i]['file']),
                    'video_path': os.path.join(save_root, 'AVDeepfake1M', 'val', data_list[i]['file']),
                    'audio_path': os.path.join(save_root, 'AVDeepfake1M', 'val', data_list[i]['file'][:-4] + '.wav')
                }
                # Todo: data['method']

                if len(data_list[i]['fake_segments']) != 0:
                    data['fake_segments'] = data_list[i]['fake_segments']

                data_list[i] = data
        return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose dataset split to preprocess')
    parser.add_argument('--split', type=str, default='all',
                        help='which data split to preprocess, options: train, val, test, all')
    parser.add_argument('--transcode', action='store_true', help='override options in config.yaml')
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs/path.yaml'), 'r') as f:
        path_config = yaml.safe_load(f)

    with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    config.update(path_config)
    config.update(vars(args))
    print(config)

    preprocessor = AVDeepfake1MPreprocessor(config)
    preprocessor.split_clip()
    preprocessor.assemble()
