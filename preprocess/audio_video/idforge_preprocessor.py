import argparse
import os
import pandas as pd
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocess.audio_video.audio_video_preprocessor import AudioVideoPreprocessor


class IDForgePreprocessor(AudioVideoPreprocessor):
    def __init__(self, split):
        super(IDForgePreprocessor, self).__init__(dataset_name='IDForge', split=split)

    def collect_data_attributes(self, data_split):
        """
        Deal with different data labeling format. Output a list of dictionary containing attributes of each data.
        Returns:
            data_list: [list] each element is a dictionary containing attributes of this data.
        """
        df = pd.read_csv(os.path.join(self.path['original_dir'], 'IDForge_v1', 'main', f'{data_split}.csv'), header=None)

        data_list = []
        for index, row in df.iterrows():
            data = {
                'label': 1 if not row[0].startswith('pristine') else 0,
                'original_path': os.path.join(self.path['original_dir'], 'IDForge_v1', 'main', row[0]).replace('\\', '/'),
                'output_path': os.path.join(self.path['preprocessed_dir'], self.dataset_type, 'IDForge', 'main', row[0][:-4].replace('\\', '/'))
            }
            if row[0].startswith('face_audiomismatch_textmismatch'):
                data['video_label'], data['audio_label'] = 1, 0
                data['method'] = f'audio_shuffling -> {row[0].split("/")[-1].split(".")[0].split("_")[-1]} -> Wav2Lip'
            elif row[0].startswith('face_rvc_textmismatch'):
                data['video_label'], data['audio_label'] = 1, 1
                data['method'] = f'RVC -> {row[0].split("/")[-1].split(".")[0].split("_")[-1]} -> Wav2Lip'
            elif row[0].startswith('face_tts'):
                data['video_label'], data['audio_label'] = 1, 1
                data['method'] = f'TorToiSe -> {row[0].split("/")[-1].split(".")[0].split("_")[-1]} -> Wav2Lip'
            elif row[0].startswith('face_tts_textgen'):
                data['video_label'], data['audio_label'] = 1, 1
                data['method'] = f'TorToiSe -> {row[0].split("/")[-1].split(".")[0].split("_")[-1]} -> Wav2Lip'
            elif row[0].startswith('lip_audiomismatch_textmismatch'):
                data['video_label'], data['audio_label'] = 1, 0
                data['method'] = 'audio_shuffling -> Wav2Lip'
            elif row[0].startswith('lip_rvc_textmismatch'):
                data['video_label'], data['audio_label'] = 1, 1
                data['method'] = 'RVC -> Wav2Lip'
            elif row[0].startswith('lip_tts_textgen'):
                data['video_label'], data['audio_label'] = 1, 1
                data['method'] = 'TorToiSe -> Wav2Lip'
            elif row[0].startswith('pristine'):
                data['video_label'], data['audio_label'] = 0, 0
                data['method'] = 'Real'
            elif row[0].startswith('rvc_textmismatch'):
                data['video_label'], data['audio_label'] = 0, 1
                data['method'] = 'RVC'
            elif row[0].startswith('tts_textgen'):
                data['video_label'], data['audio_label'] = 0, 1
                data['method'] = 'TorToiSe'
            elif row[0].startswith('tts_textmismatch'):
                data['video_label'], data['audio_label'] = 0, 1
                data['method'] = 'TorToiSe'
            else:
                raise AssertionError(f'No such category: {row[0]}')

            data_list.append(data)

        return data_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val', 'test'],
                        help='Which split to preprocess')
    parser.add_argument('--num_proc', type=int, default=1, help='number of processes to use')
    args = parser.parse_args()

    preprocessor = IDForgePreprocessor(split=args.split)
    preprocessor.preprocess_all_data(num_proc=args.num_proc)
    preprocessor.assemble_to_json()
