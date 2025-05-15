import copy
import json
import os
import sys

from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocess.utils.tools import split_video_and_audio_multiprocess, video_and_audio_segmentation_multiprocess

class BasePreprocessor:
    def __init__(self, config, output_name):
        self.config = config
        self.output_name = output_name
        print(f'Start to collect data.')
        self.train_data_list, self.val_data_list, self.test_data_list = [], [], []
        if config['split'] in ['all', 'train']:
            self.train_data_list = self.collect_data_attributes('train')
            print(f'Totally training {len(self.train_data_list)} audio-video found.')
        if config['split'] in ['all', 'val']:
            self.val_data_list = self.collect_data_attributes('val')
            print(f'Totally validation {len(self.val_data_list)} audio-video found.')
        if config['split'] in ['all', 'test']:
            self.test_data_list = self.collect_data_attributes('test')
            print(f'Totally testing {len(self.test_data_list)} audio-video found.')
        self.data_list = self.train_data_list + self.val_data_list + self.test_data_list

    def split_clip(self):
        print(f'Start to separate video and audio stream from each file. Adjust video to {self.config["FPS"]} FPS audio '
              f'to {self.config["sample_rate"]}Hz sampling rate.')
        split_video_and_audio_multiprocess(self.data_list, self.config.get('num_proc', os.cpu_count()))

        print(f'Start to detect, align, and crop face for each frame. Segment video and audio.')
        video_and_audio_segmentation_multiprocess(self.data_list, self.config.get('num_proc', os.cpu_count()))

    def assemble(self):
        def assemble_data_split(data_split, data_list):
            all_data_json[data_split] = []
            for i in tqdm(range(len(data_list)), dynamic_ncols=True):
                data = data_list[i]
                save_root = data['video_path'][:-4]
                # No segment saved in preprocess()
                if not os.path.exists(save_root):
                    continue

                # rearrange json dict for every clip, record relative path instead of absolute path
                for segment_name in list(Path(save_root).glob('*')):
                    clip_json = copy.deepcopy(data)
                    if clip_json['transcode']:
                        clip_json['path'] = str(segment_name.relative_to(self.config['transcode_dir']))
                    else:
                        clip_json['path'] = str(segment_name.relative_to(self.config['preprocess_dir']))
                    # clip_json['video'] = str((segment_name / 'frames.npz').relative_to(self.config['transcode_dir']))
                    # clip_json['audio'] = str((segment_name / 'audio.wav').relative_to(self.config['transcode_dir']))
                    if clip_json.get('original_path', None) is not None:
                        clip_json['original_path'] = str(Path(clip_json['original_path']).relative_to(self.config['original_dir']))
                    elif clip_json.get('original_video_path', None) is not None and clip_json.get('original_audio_path', None) is not None:
                        clip_json['original_video_path'] = str(Path(clip_json['original_video_path']).relative_to(self.config['original_dir']))
                        clip_json['original_audio_path'] = str(Path(clip_json['original_audio_path']).relative_to(self.config['original_dir']))
                    del clip_json['video_path'], clip_json['audio_path'], clip_json['transcode'], clip_json['clip_amount']
                    all_data_json[data_split].append(clip_json)

        print(f'Assemble preprocessed data with labels and metadata into json')
        all_data_json = {}
        assemble_data_split('train', self.train_data_list)
        assemble_data_split('val', self.val_data_list)
        assemble_data_split('test', self.test_data_list)

        os.makedirs(os.path.join(self.config['json_dir']), exist_ok=True)
        with open(os.path.join(self.config['json_dir'], f'{self.output_name}.json'), 'w') as f:
            json.dump(all_data_json, f, indent=4)
