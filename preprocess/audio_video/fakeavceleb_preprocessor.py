import argparse
import os
import pandas as pd
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocess.audio_video.audio_video_preprocessor import AudioVideoPreprocessor


class FakeAVCelebPreprocessor(AudioVideoPreprocessor):
    def __init__(self, split, split_dir=None):
        self.split_dir = os.path.abspath(split_dir or os.path.join(os.path.dirname(__file__), 'fakeavceleb_splits'))
        super(FakeAVCelebPreprocessor, self).__init__(dataset_name='FakeAVCeleb', split=split)

    def collect_data_attributes(self, data_split):
        """
        Deal with different data labeling format. Output a list of dictionary containing attributes of each data.
        Returns:
            data_list: [list] each element is a dictionary containing attributes of this data.
        """
        df = pd.read_csv(os.path.join(self.split_dir, f'{data_split}_meta_data.csv'))
        df.columns.values[-1] = 'parent_path'
        df['parent_path'] = df['parent_path'].str.replace('FakeAVCeleb', 'FakeAVCeleb_v1.2')
        df['original_path'] = df.iloc[:, -1] + '/' + df['path']
        duplicated_paths = df[df['original_path'].duplicated(keep='first')]['original_path'].to_list()
        # Remarks: In this `meta_data.csv` file, 22 files have 2 row records, indicating in FakeAudio-FakeVideo category,
        #          this data passed the forgery pipeline 'faceswap-wav2lip' -> 'wav2lip'.

        data_list = []
        for index, row in df.iterrows():
            data = {
                'video_label': 1 if row['category'] in ['C', 'D'] else 0,
                'audio_label': 1 if row['category'] in ['B', 'D'] else 0,
                'label': 1 if row['category'] in ['B', 'C', 'D'] else 0,
                'original_path': os.path.join(self.path['original_dir'], row['original_path']).replace('\\', '/'),
                'output_path': os.path.join(self.path['preprocessed_dir'], self.dataset_type,
                                            row['original_path'][:-4]).replace('\\', '/')
            }
            # Duplicate rows have the same forgery pipeline, i.e., 'faceswap-wav2lip' -> 'wav2lip'.
            if row['original_path'] not in duplicated_paths:
                data['method'] = row['method']
            else:
                data['method'] = 'faceswap-wav2lip -> wav2lip'

            if data not in data_list:
                data_list.append(data)

        # remove duplicated items
        return data_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val', 'test'],
                        help='Which split to preprocess')
    parser.add_argument('--num_proc', type=int, default=1, help='number of processes to use')
    parser.add_argument('--split_dir', default=None,
                        help='Split CSV directory; defaults to the bundled fakeavceleb_splits directory')
    args = parser.parse_args()

    preprocessor = FakeAVCelebPreprocessor(split=args.split, split_dir=args.split_dir)
    preprocessor.preprocess_all_data(num_proc=args.num_proc)
    preprocessor.assemble_to_json()
