import argparse
import os
import pandas as pd
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocess.base_preprocessor import BasePreprocessor


class IDForgePreprocessor(BasePreprocessor):
    def __init__(self, config):
        super().__init__(config, 'IDForge')

    def collect_data_attributes(self, data_split):
        """
        Deal with different data labeling format. Output a list of dictionary containing attributes of each data.
        Returns:
            data_list: [list] each element is a dictionary containing attributes of this data.
        """
        df = pd.read_csv(os.path.join(self.config['original_dir'], 'IDForge_v1', 'main', f'{data_split}.csv'), header=None)

        data_list = []
        for index, row in df.iterrows():
            save_root = self.config['transcode_dir'] if self.config.get('transcode', False) else self.config[
                'preprocess_dir']
            data = {
                'FPS': self.config.get('FPS', 25),
                'sample_rate': self.config.get('sample_rate', 16000),
                'duration': self.config.get('length', 1.0),
                'clip_amount': self.config.get('clip_amount', 1),
                'label': 1 if not row[0].startswith('pristine') else 0,
                'original_path': os.path.join(self.config['original_dir'], 'IDForge_v1', 'main', row[0]).replace('\\', '/'),
                'video_path': os.path.join(save_root, 'IDForge_v1', 'main', row[0]).replace('\\', '/'),
                'audio_path': os.path.join(save_root, 'IDForge_v1', 'main', row[0][:-4] + '.wav').replace('\\', '/')
            }
            if row[0].startswith('face_audiomismatch_textmismatch'):
                data['video_label'], data['audio_label'] = 1, 0
                data['method'] = ['audio_shuffling', row[0].split('/')[-1].split('.')[0].split('_')[-1], 'Wav2Lip']
            elif row[0].startswith('face_rvc_textmismatch'):
                data['video_label'], data['audio_label'] = 1, 1
                data['method'] = ['RVC', row[0].split('/')[-1].split('.')[0].split('_')[-1], 'Wav2Lip']
            elif row[0].startswith('face_tts'):
                data['video_label'], data['audio_label'] = 1, 1
                data['method'] = ['TorToiSe', row[0].split('/')[-1].split('.')[0].split('_')[-1], 'Wav2Lip']
            elif row[0].startswith('face_tts_textgen'):
                data['video_label'], data['audio_label'] = 1, 1
                data['method'] = ['TorToiSe', row[0].split('/')[-1].split('.')[0].split('_')[-1], 'Wav2Lip']
            elif row[0].startswith('lip_audiomismatch_textmismatch'):
                data['video_label'], data['audio_label'] = 1, 0
                data['method'] = ['audio_shuffling', 'Wav2Lip']
            elif row[0].startswith('lip_rvc_textmismatch'):
                data['video_label'], data['audio_label'] = 1, 1
                data['method'] = ['RVC', 'Wav2Lip']
            elif row[0].startswith('lip_tts_textgen'):
                data['video_label'], data['audio_label'] = 1, 1
                data['method'] = ['TorToiSe', 'Wav2Lip']
            elif row[0].startswith('pristine'):
                data['video_label'], data['audio_label'] = 0, 0
                data['method'] = ['Real']
            elif row[0].startswith('rvc_textmismatch'):
                data['video_label'], data['audio_label'] = 0, 1
                data['method'] = ['RVC']
            elif row[0].startswith('tts_textgen'):
                data['video_label'], data['audio_label'] = 0, 1
                data['method'] = ['TorToiSe']
            elif row[0].startswith('tts_textmismatch'):
                data['video_label'], data['audio_label'] = 0, 1
                data['method'] = ['TorToiSe']
            else:
                raise AssertionError(f'No such category: {row[0]}')

            data_list.append(data)

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

    preprocessor = IDForgePreprocessor(config)
    preprocessor.split_clip()
    preprocessor.assemble()
