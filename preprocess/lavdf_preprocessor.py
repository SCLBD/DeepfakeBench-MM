import argparse
import json
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocess.base_preprocessor import BasePreprocessor

class LAVDFPreprocessor(BasePreprocessor):
    def __init__(self, config):
        super().__init__(config, 'LAV-DF')

    def collect_data_attributes(self, data_split):
        """
        Deal with different data labeling format. Output a list of dictionary containing attributes of each data.
        Returns:
            data_list: [list] each element is a dictionary containing attributes of this data.
        """
        with open(os.path.join(self.config['original_dir'], 'LAV-DF', f'metadata.json'), 'r') as f:
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
            save_root = self.config['transcode_dir'] if self.config.get('transcode', False) else self.config['preprocess_dir']
            data = {
                'transcode': self.config.get('transcode', False),
                'FPS': self.config.get('FPS', 25),
                'sample_rate': self.config.get('sample_rate', 16000),
                'duration': self.config.get('length', 1.0),
                'clip_amount': self.config.get('clip_amount', 1),
                'label': 1 if data_list[i]['modify_video'] or data_list[i]['modify_audio'] else 0,
                'video_label': 1 if data_list[i]['modify_video'] else 0,
                'audio_label': 1 if data_list[i]['modify_audio'] else 0,
                'original_path': os.path.join(self.config['original_dir'], 'LAV-DF', data_list[i]['file']).replace('\\', '/'),
                'video_path': os.path.join(save_root, 'LAV-DF', data_list[i]['file']).replace('\\', '/'),
                'audio_path': os.path.join(save_root, 'LAV-DF', data_list[i]['file'][:-4] + '.wav').replace('\\', '/')
            }
            # Todo: data['method']

            if len(data_list[i]['fake_periods']) != 0:
                data['fake_segments'] = data_list[i]['fake_periods']

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

    preprocessor = LAVDFPreprocessor(config)
    # preprocessor.split_clip()
    preprocessor.assemble()

