import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocess.audio_video.audio_video_preprocessor import AudioVideoPreprocessor


class MMDFPreprocessor(AudioVideoPreprocessor):
    def __init__(self, split):
        super(MMDFPreprocessor, self).__init__(dataset_name='MMDF', split=split)

    def collect_data_attributes(self, data_split):
        """
        Collect MMDF data attributes for the given split.

        The dataset has a unified layout: MMDF/<split>/<label>/<model_id>/<clip_id>.mp4 where
            - split: one of 'train', 'val', 'test'.
            - label: 'real' or 'fake', which is the video modality label (audio modality is always real).
            - model_id: the forgery method, used as the 'method' label.
            - clip_id: video id, all files are '.mp4' containing both audio and video streams.

        Returns:
            data_list: [list] each element is a dictionary containing attributes of this data.
        """
        data_list = []
        split_dir = os.path.join(self.path['original_dir'], 'MMDF', data_split)
        if not os.path.isdir(split_dir):
            return data_list

        for label in ['real', 'fake']:
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue

            video_label = 1 if label == 'fake' else 0
            for model_id in sorted(os.listdir(label_dir)):
                model_dir = os.path.join(label_dir, model_id)
                if not os.path.isdir(model_dir):
                    continue

                for file_name in sorted(os.listdir(model_dir)):
                    if not file_name.endswith('.mp4'):
                        continue

                    rel_path = os.path.join('MMDF', data_split, label, model_id, file_name).replace('\\', '/')
                    data_list.append({
                        'video_label': video_label,
                        'audio_label': 0,
                        'label': video_label,
                        'original_path': os.path.join(self.path['original_dir'], rel_path).replace('\\', '/'),
                        'output_path': os.path.join(self.path['preprocessed_dir'], self.dataset_type,
                                                    rel_path[:-4]).replace('\\', '/'),
                        'method': model_id,
                    })

        return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val', 'test'],
                        help='Which split to preprocess')
    parser.add_argument('--num_proc', type=int, default=1, help='number of processes to use')
    args = parser.parse_args()

    preprocessor = MMDFPreprocessor(split=args.split)
    preprocessor.preprocess_all_data(num_proc=args.num_proc)
    preprocessor.assemble_to_json()
