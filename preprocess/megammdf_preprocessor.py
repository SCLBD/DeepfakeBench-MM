import argparse
import json
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocess.base_preprocessor import BasePreprocessor

class MegaMMDFPreprocessor(BasePreprocessor):
    def __init__(self, config):
        super().__init__(config, 'Mega-MMDF')

    def collect_data_attributes(self, data_split):
        """
        Deal with different data labeling format. Output a list of dictionary containing attributes of each data.
        Returns:
            data_list: [list] each element is a dictionary containing attributes of this data.
        """
        with open(os.path.join(self.config['original_dir'], 'Mega-MMDF', f'{data_split}.json'), 'r') as f:
            data_list = json.load(f)

        with open(os.path.join(self.config['original_dir'], 'Mega-MMDF', 'attributes.json'), 'r') as f:
            data_attributes = json.load(f)

        for i in range(len(data_list)):
            save_root = self.config['transcode_dir'] if self.config.get('transcode', False) else self.config['preprocess_dir']
            data = {
                'transcode': self.config.get('transcode', False),
                'FPS': self.config.get('FPS', 25),
                'sample_rate': self.config.get('sample_rate', 16000),
                'duration': self.config.get('length', 1.0),
                'clip_amount': self.config.get('clip_amount', 1),
                'label': 1 if 'RARV' not in data_list[i] else 0,
                'video_label': 1 if 'FV' in data_list[i] else 0,
                'audio_label': 1 if 'FA' in data_list[i] else 0,
                'original_path': os.path.join(self.config['original_dir'], data_list[i]).replace('\\', '/'),
                'video_path': os.path.join(save_root, data_list[i]).replace('\\', '/'),
                'audio_path': os.path.join(save_root, data_list[i][:-4] + '.wav').replace('\\', '/')
            }
            # Todo: data['method']
            original_file_path = data_list[i].replace('\\', '/')
            if data_attributes[original_file_path].get('partial_fake_segments', None) is not None:
                data['fake_segments'] = [data_attributes[original_file_path]['partial_fake_segments']]

            data_list[i] = data
        return data_list


if __name__ == '__main__':
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

        preprocessor = MegaMMDFPreprocessor(config)
        preprocessor.split_clip()
        preprocessor.assemble()
