import argparse
import os
import pandas as pd
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocess.base_preprocessor import BasePreprocessor

class FakeAVCelebPreprocessor(BasePreprocessor):
    def __init__(self, config):
        super().__init__(config, 'FakeAVCeleb')

    def collect_data_attributes(self, data_split):
        """
        Deal with different data labeling format. Output a list of dictionary containing attributes of each data.
        Returns:
            data_list: [list] each element is a dictionary containing attributes of this data.
        """
        df = pd.read_csv(os.path.join(self.config['original_dir'], 'FakeAVCeleb_v1.2', f'{data_split}_meta_data.csv'))
        df.columns.values[-1] = 'parent_path'
        df['parent_path'] = df['parent_path'].str.replace('FakeAVCeleb', 'FakeAVCeleb_v1.2')
        df['original_path'] = df.iloc[:, -1] + '/' + df['path']
        duplicated_paths = df[df['original_path'].duplicated(keep='first')]['original_path'].to_list()
        # Remarks: In this `meta_data.csv` file, 22 files have 2 row records, indicating in FakeAudio-FakeVideo category,
        #          this data passed the forgery pipeline 'faceswap-wav2lip' -> 'wav2lip'.

        data_list = []
        for index, row in df.iterrows():
            save_root = self.config['transcode_dir'] if self.config.get('transcode', False) else self.config['preprocess_dir']
            data = {
                'transcode': self.config.get('transcode', False),
                'FPS': self.config.get('FPS', 25),
                'sample_rate': self.config.get('sample_rate', 16000),
                'duration': self.config.get('length', 1.0),
                'clip_amount': self.config.get('clip_amount', 1),
                'video_label': 1 if row['category'] in ['C', 'D'] else 0,
                'audio_label': 1 if row['category'] in ['B', 'D'] else 0,
                'label': 1 if row['category'] in ['B', 'C', 'D'] else 0,
                # Duplicate rows have the same forgery pipeline, i.e., 'faceswap-wav2lip' -> 'wav2lip'.
                'method': [row['method']] if row['original_path'] not in duplicated_paths
                else ['faceswap-wav2lip', 'wav2lip'],
                'original_path': os.path.join(self.config['original_dir'], row['original_path']).replace('\\', '/'),
                'video_path': os.path.join(save_root, row['original_path']).replace('\\', '/'),
                'audio_path': os.path.join(save_root, row['original_path'][:-4] + '.wav').replace('\\', '/')
            }

            if data not in data_list:
                data_list.append(data)

        # remove duplicated items
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

    preprocessor = FakeAVCelebPreprocessor(config)
    # preprocessor.split_clip()
    preprocessor.assemble()

