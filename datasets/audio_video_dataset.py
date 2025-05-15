import albumentations as A
import cv2
import json
import librosa
import numpy as np
import os
import python_speech_features
import torch
import torch.nn.functional as F
import warnings
import yaml

warnings.filterwarnings('ignore')

from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class AudioVideoDataset(Dataset):
    def __init__(self, config=None, mode='train'):
        assert mode in ['train', 'val', 'test']
        self.config = config
        self.mode = mode
        self.transform = self.init_data_aug_method()
        self.data_list = []
        if mode == 'train':     # Only in train mode, a list of dataset name provided in config['train_dataset'].
            dataset_list = config['train_dataset']
            for dataset_name in dataset_list:
                with open(os.path.join(config['json_dir'], f'{dataset_name}.json'), 'r') as f:
                    self.data_list.extend(json.load(f)['train'])
        elif mode == 'val':     # In val and test mode, a string of dataset name is provided in it.
            with open(os.path.join(config['json_dir'], f'{config["val_dataset"]}.json'), 'r') as f:
                self.data_list.extend(json.load(f)[mode])
        elif mode == 'test':    # In val and test mode, a string of dataset name is provided in it.
            with open(os.path.join(config['json_dir'], f'{config["test_dataset"]}.json'), 'r') as f:
                self.data_list.extend(json.load(f)[mode])

        # Convert audio into spectrogram to save data loading cost.
        if config.get('audio_conversion', False):
            print(f'Convert audio into {config["audio_conversion"]} format and save to {config["temp_dir"]}.')
            for data in tqdm(self.data_list):
                if config['audio_conversion'] == 'MFCC':  # Convert to MFCC using python_speech_features
                    save_path = os.path.join(config["temp_dir"], data['path'], 'mfcc.npy')
                    if os.path.exists(save_path):
                        continue

                    if config.get('use_transcoded', False):
                        audio = np.load(os.path.join(config['transcode_dir'], data['path'], 'data.npz'))['audio']
                        sr = data['sample_rate']
                    else:
                        sr, audio = wavfile.read(os.path.join(config['preprocess_dir'], data['path'], 'audio.wav'))
                    mfcc = python_speech_features.mfcc(audio, sr)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, mfcc.astype(np.float32))
                    data['mfcc'] = save_path
                else:   # Todo: More spectrogram should be implemented here.
                    raise NotImplementedError(f'Unsupported audio conversion format: {config["audio_conversion"]}.')

    def init_data_aug_method(self):
        augmentation_config = self.config['augmentations']
        trans = []
        if augmentation_config.get('flip', None) is not None:
            if augmentation_config['flip'].get('type', 'horizontal') == 'horizontal':
                trans.append(A.HorizontalFlip(p=augmentation_config['flip'].get('prob', 0.5)))
            else:
                raise NotImplementedError(f'{augmentation_config["flip"]["type"]} is not supported.')
        if augmentation_config.get('rotate', None) is not None:
            trans.append(A.Rotate(limit=augmentation_config['rotate'].get('rotate_limit', [-10, 10]),
                                  p=augmentation_config['rotate'].get('prob', 0.5)))
        if augmentation_config.get('gaussian_blur', None) is not None:
            trans.append(A.GaussianBlur(blur_limit=augmentation_config['gaussian_blur'].get('blur_limit', [3, 7]),
                                        p=augmentation_config['gaussian_blur'].get('prob', 0.5)))
        if augmentation_config.get('color', None) is not None:
            trans.append(A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=augmentation_config['color'].get('brightness_limit', [-0.1, 0.1]),
                    contrast_limit=augmentation_config['color'].get('contrast_limit', [-0.1, 0.1])
                ),
                A.FancyPCA(),
                A.HueSaturationValue()], p=augmentation_config['color'].get('prob', 0.5)))
        if augmentation_config.get('quality', None) is not None:
            trans.append(A.ImageCompression(quality_lower=augmentation_config['quality'].get('quality_lower', 40),
                                            quality_upper=augmentation_config['quality'].get('quality_upper', 100)))

        return A.ReplayCompose(trans,
                               keypoint_params=A.KeypointParams(format='xy')
                                               if self.config.get('with_landmark', False) else None)

    def __getitem__(self, index):
        # Todo: Fix bug of landmarks changes along with augmentations.
        if self.config.get('with_landmarks', False):
            raise NotImplementedError('[Todo] landmarks transformation.')

        data = self.data_list[index]
        if self.config.get('use_transcoded', False):
            tmp = np.load(os.path.join(self.config['transcode_dir'], data['path'], 'data.npz'))
            frames, landmarks, audio = tmp['frames'], tmp['landmarks'], tmp['audio']
        else:
            frames = []
            cap = cv2.VideoCapture(os.path.join(self.config['preprocess_dir'], data['path'], 'frames.mp4'))
            if not cap.isOpened:
                raise RuntimeError(f'Failed to open video {os.path.join(self.config["preprocess_dir"], data["path"], "frames.mp4")}')
            for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f'Fail to open {i + 1}-th frame in {os.path.join(self.config["preprocess_dir"], data["path"], "frames.mp4")}')
                frames.append(frame)
            frames = np.stack(frames)
            landmarks = np.load(os.path.join(self.config["preprocess_dir"], data["path"], "landmarks.npy"))
            audio, sr = librosa.load(os.path.join(self.config['preprocess_dir'], data['path'], 'audio.wav'), sr=None)
        # frames = np.load(os.path.join(self.config['encoded_dir'], 'audio-video', data['video']))['frames']
        # landmarks = np.load(os.path.join(self.config['encoded_dir'], data['video']))['landmarks'] if self.config.get(
        #     'with_landmarks', False) else None

        # Convert BGR to RGB
        for i in range(frames.shape[0]):
            frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)

        # data augmentation
        if self.mode == 'train' and self.config.get('augmentations', None) is not None:
            for i in range(frames.shape[0]):
                kwargs = {'image': frames[i]}
                # if self.config.get('with_landmarks', False):
                #     kwargs['keypoints'] = landmarks[i]
                #     kwargs['keypoint_params'] = A.KeypointParams(format='xy')

                if i == 0:
                    augmented = self.transform(**kwargs)
                    replay_params = augmented['replay']
                else:
                    augmented = A.ReplayCompose.replay(replay_params, **kwargs)

                frames[i] = augmented['image']
                # if self.config.get('with_landmarks', False):
                #     landmarks[i] = np.array(augmented['keypoints'])


        # Resize to video resolution setting
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        frames = F.interpolate(frames, size=self.config['video_resolution'], mode='bilinear', align_corners=False)
        # if self.config.get('with_landmarks', False):
        #     landmarks = landmarks * self.config['video_resolution'] / 256

        # normalize
        frames = frames.permute(1, 0, 2, 3)
        frames = (frames / 255.0 - torch.tensor(self.config['mean']).reshape(3, 1, 1, 1)) / torch.tensor(
            self.config['std']).reshape(3, 1, 1, 1)


        if self.config.get('audio_conversion', None) is not None:
            if self.config['audio_conversion']:
                audio = np.load(os.path.join(self.config["temp_dir"], data['path'], 'mfcc.npy'))
            else:
                print('[Todo] More spectrogram should be implemented here.')

        return {'video': frames,
                'audio': audio,
                # 'landmarks': landmarks,
                'video_label': data['video_label'],
                'audio_label': data['audio_label'],
                'label': data['label'],
                'path': data['path']}

    def __len__(self):
        return len(self.data_list)

    # @staticmethod
    # def collate_fn(batch):
    #     return data_dict['video']


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'path.yaml'), 'r') as f:
        path_config = yaml.safe_load(f)

    test_config = {
        'use_transcoded': True,
        'audio_conversion': 'MFCC',
        'video_resolution': 256,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'train_dataset': ['FakeAVCeleb'],
        'val_dataset': ['FakeAVCeleb'],
        'with_landmarks': False,
        'augmentations': {
            'flip': {'type': 'horizontal', 'prob': 0.5},
            'rotate': {'rotate_limit': [-50, 50], 'prob': 1},
            'gaussian_blur': {'blur_limit': [3, 7], 'blur_prob': 0.5},
            'color': {'brightness_limit': [-0.1, 0.1], 'contrast_limit': [-0.1, 0.1], 'prob': 0.5},
            'quality': {'quality_lower': 40, 'quality_upper': 100},
        },
    }

    dataset = AudioVideoDataset(config={**path_config, **test_config}, mode='train')
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, num_workers=4)
    for idx, batch_data in enumerate(dataloader):
        for key, value in batch_data.items():
            if key in ['audio', 'video']:
                print(key, value.shape)
            else:
                print(key, value)
        for i in range(batch_data['video'].shape[2]):
            img = batch_data['video'].permute(0, 2, 3, 4, 1).numpy()
            img = img * np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 1, 3) + np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 1, 3)
            img = (img * 255).astype(np.uint8)
            img = img[0, i, :, :, ::-1].astype(np.uint8)    # to BGR
            cv2.imwrite(f'temp_{i:04d}.png', img)
        break
