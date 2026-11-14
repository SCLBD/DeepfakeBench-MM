"""Ordinary audio-video loading for training, validation and testing."""

import copy
import cv2
import json
import numpy as np
import os
import random
import sys
import torch

from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.abstract_dataset import AbstractDataset
from datasets.utils.augmentations.audio_augmentations import apply_audio_aug, init_audio_aug
from datasets.utils.augmentations.video_augmentations import apply_video_aug, init_video_aug
from datasets.utils.data_helpers.audio_helpers import convert_audio_spectrogram, load_audio_segment, spectrogram_normalization
from datasets.utils.data_helpers.audio_video_helpers import validate_audio_video_files
from datasets.utils.data_helpers.video_helpers import (
    load_landmarks_fixed_stride,
    load_video_frames_fixed_stride,
    video_normalization,
)


class AudioVideoDataset(AbstractDataset):
    dataset_type = 'audio-video'
    default_aug = {
        'video': {
            'flip': {'type': 'horizontal', 'prob': 0.5},
            'rotate': {'range': [-10, 10], 'prob': 0.5},
            'gaussian_blur': {'range': [3, 7], 'prob': 0.5},
            'color_shift': {'brightness_range': [-0.1, 0.1], 'contrast_range': [-0.1, 0.1], 'prob': 0.5},
            'compression': {'range': [40, 100], 'prob': 0.5},
        },
        'audio': {
            'gaussian_noise': {'range': [0.001, 0.01], 'prob': 0.5},
            'pitch_shift': {'range': [-2, 2], 'prob': 0.5},
            'volume_gain': {'range': [-6, 6], 'prob': 0.5},
        },
        'audiovisual': {
            'delay': {'range': [-0.1, 0.1], 'prob': 0.5},
        },
    }

    def __init__(self, mode, data_config, aug_config=None, dataset_list=None, dataset_name=None):
        """
        Initialize the basic audio-video dataset and its sample index.
        Args:
            mode (str): Dataset mode, options: ['train', 'val', 'test'].
            data_config (dict): Input and sampling settings, including the
                number of returned frames, source-frame stride, frame size,
                audio sampling rate and optional spectrogram parameters.
            aug_config (str, dict or None): Training augmentation settings.
                'default' uses default_aug, None or {} disables augmentation,
                and a nonempty dictionary recursively overrides the defaults.
            dataset_list (list): Dataset names to combine in train mode.
            dataset_name (str): Dataset name to read in val or test mode.
        """
        super().__init__(mode, data_config, aug_config)
        self.video_fps = 25
        self.audio_sr = self.data_config.get('audio_sampling_rate', 16000)
        self.data_info_list = []
        self.data_root = os.path.join(self.root_path, self.data_config.get('data_directory', self.dataset_type))

        if self.mode == 'train':
            assert isinstance(dataset_list, list)
            self.video_aug_transform = init_video_aug(self.aug_config.get('video', {}), data_config.get('with_landmark', False))
            self.audio_aug_transform = init_audio_aug(self.aug_config.get('audio', {}))
            dataset_names = dataset_list
        else:
            assert isinstance(dataset_name, str)
            self.video_aug_transform, self.audio_aug_transform = None, None
            dataset_names = [dataset_name]

        for name in dataset_names:
            json_path = os.path.join(self.path_config['json_dir'], self.dataset_type, f'{name}.json')
            with open(json_path, 'r') as f:
                json_content = json.load(f)
            # Expand source records into the current mode's sampling windows.
            for data_info in json_content[self.mode]:
                start_frame, end_frame = str(Path(data_info['path']).stem).split('_')
                start_frame, end_frame = int(start_frame), int(end_frame)
                num_frames_this_data = end_frame - start_frame
                window_span_frames = self.data_config['num_frames'] * self.data_config['stride']

                if self.mode == 'test':
                    num_samples = num_frames_this_data // window_span_frames
                    for i in range(num_samples):
                        sample_info = copy.deepcopy(data_info)
                        sample_info['start_frame'] = start_frame + window_span_frames * i
                        sample_info['end_frame'] = start_frame + window_span_frames * (i + 1)
                        self.data_info_list.append(sample_info)
                else:
                    num_samples = min(self.data_config.get('max_samples_per_data', 2), num_frames_this_data // window_span_frames)
                    # Split the source interval into capped training/validation portions.
                    separation_frames = np.linspace(start_frame, end_frame, num_samples + 1)
                    separation_frames = np.round(separation_frames).astype(np.int32).tolist()
                    for i in range(len(separation_frames) - 1):
                        sample_info = copy.deepcopy(data_info)
                        sample_info['start_frame'] = separation_frames[i]
                        sample_info['end_frame'] = separation_frames[i + 1]
                        self.data_info_list.append(sample_info)

        if self.data_config.get('data_directory', self.dataset_type) != self.dataset_type:
            validate_audio_video_files(
                self.data_info_list,
                self.data_root,
                with_landmark=self.data_config.get('with_landmark', False),
            )

    def _get_data_paths(self, data_info):
        """
        Resolve prepared audio, video and landmark paths for one source record.
        Args:
            data_info (dict): Indexed source record whose path is relative to
                preprocessed_dir/dataset_type.
        Returns:
            audio_path (str): WAV path beside the source clip's segment folder.
            video_path (str): Prepared video segment path.
            landmark_path (str): Corresponding NPY path, read only when
                data_config.with_landmark is enabled.
        """
        data_root = self.data_root
        audio_path = os.path.join(data_root, str(Path(data_info['path']).parent) + '.wav')
        video_path = os.path.join(data_root, data_info['path'])
        landmark_path = os.path.join(data_root, data_info['path'].replace('mp4', 'npy'))
        return audio_path, video_path, landmark_path

    def _load_video(self, video_path, landmark_path, start_frame, window_span_frames):
        """
        Read fixed-stride video frames and optional matching landmarks.
        Args:
            video_path (str): Prepared video segment path.
            landmark_path (str): Corresponding landmark NPY path.
            start_frame (int): Window start on the normalized source timeline.
            window_span_frames (int): Window duration in source frames, equal
                to num_frames * stride. The returned frame count is num_frames.
        Returns:
            frames (np.ndarray): uint8 frames (T,H,W,C), [0,255], RGB or grayscale C=1; T is the
                configured number of returned frames.
            landmarks (np.ndarray or None): Coordinates in shape (T, 68, 2),
                or None when landmarks are not requested.
        """
        frames = load_video_frames_fixed_stride(
            video_path=video_path,
            color_mode=self.data_config.get('color_mode', 'RGB'),
            sample_start_frame=start_frame,
            sample_end_frame=start_frame + window_span_frames,
            sample_stride=self.data_config['stride'],
            mask_video=False,
        )
        landmarks = None
        if self.data_config.get('with_landmark'):
            landmarks = load_landmarks_fixed_stride(
                landmark_path=landmark_path,
                sample_start_frame=start_frame,
                sample_end_frame=start_frame + window_span_frames,
                sample_stride=self.data_config['stride'],
            )
        return frames, landmarks

    def _load_audio(self, audio_path, start_frame, window_span_frames):
        """
        Read the matching waveform, with the configured training time jitter.
        Args:
            audio_path (str): Prepared mono audio path.
            start_frame (int): Window start on the normalized source timeline.
            window_span_frames (int): Window duration in source frames.
        Returns:
            audio (np.ndarray): float32 mono waveform (L,), normally [-1,1], at audio_sr. Its duration
                is window_span_frames / video_fps seconds. Training delay is
                drawn after video loading, preserving the original random
                call order; val/test use the unshifted interval.
        """
        temporal_shift = 0
        if self.mode == 'train' and self.aug_config.get('audiovisual'):
            delay_cfg = self.aug_config['audiovisual'].get('delay')
            if delay_cfg and random.random() < delay_cfg.get('prob', 0.5):
                shift_range = delay_cfg.get('range', [-0.1, 0.1])
                temporal_shift = random.uniform(shift_range[0], shift_range[1])
        return load_audio_segment(
            audio_path=audio_path,
            remove_dc_offset=self.data_config.get('remove_dc_offset', False),
            start_time=start_frame / self.video_fps,
            end_time=(start_frame + window_span_frames) / self.video_fps,
            audio_sampling_rate=self.audio_sr,
            temporal_shift=temporal_shift,
            mask_audio=False,
        )

    def _apply_augmentation(self, frames, audio, landmarks):
        """
        Apply ordinary training augmentation to the loaded signals.
        Args:
            frames (np.ndarray): Video frames in shape (T, H, W, C).
            audio (np.ndarray): Mono waveform in shape (L,).
            landmarks (np.ndarray or None): Optional coordinates (T, 68, 2).
        Returns:
            frames (np.ndarray or list): Augmented video frames.
            audio (np.ndarray): Augmented waveform.
            landmarks (np.ndarray or None): Coordinates transformed with video.
                Inputs pass through unchanged in val/test modes.
        """
        if self.mode == 'train':
            if self.video_aug_transform is not None:
                frames, landmarks = apply_video_aug(self.video_aug_transform, frames, landmarks)
            if self.audio_aug_transform is not None:
                audio = apply_audio_aug(self.audio_aug_transform, audio, sr=self.audio_sr)
        return frames, audio, landmarks

    def _prepare_model_inputs(self, frames, audio, landmarks):
        """
        Resize and normalize video, and prepare the configured audio features.
        Args:
            frames (np.ndarray or list): Frames in shape (T, H, W, C).
            audio (np.ndarray): Mono waveform in shape (L,).
            landmarks (np.ndarray or None): Optional aligned coordinates.
        Returns:
            frames (np.ndarray): Floating-point RGB/grayscale frames (T,H,W,C), resized to frame_size;
                value range is determined by frame_mean/std after scaling pixels to [0,1].
            audio (np.ndarray): Waveform (L,) or normalized features (Ta, F).
            landmarks (np.ndarray or None): Coordinates scaled from the
                256-pixel preprocessing space to frame_size.
        """
        frame_size = self.data_config['frame_size']
        frames = np.array([cv2.resize(frame, (frame_size, frame_size)) for frame in frames])
        if frames.ndim == 3:
            # cv2.resize drops the singleton channel of grayscale frames.
            frames = frames[..., np.newaxis]
        frames = np.array(frames)
        if self.data_config.get('with_landmark'):
            landmarks = landmarks * self.data_config['frame_size'] / 256
        frames = video_normalization(frames, self.data_config.get('frame_mean', 0), self.data_config.get('frame_std', 1))

        if self.data_config.get('spectrogram_type'):
            audio = convert_audio_spectrogram(
                audio=audio,
                sr=self.audio_sr,
                type=self.data_config['spectrogram_type'],
                backend=self.data_config['spectrogram_backend'],
                params=self.data_config.get('spectrogram_params', {}),
            )
            audio = spectrogram_normalization(audio, self.data_config.get('audio_mean', 0), self.data_config.get('audio_std', 1))
        return frames, audio, landmarks

    def _get_sample_labels(self, data_info, start_frame):
        """
        Determine audio/video labels for the complete sampled time interval.
        Args:
            data_info (dict): Source labels and optional partial-forgery intervals.
            start_frame (int): Window start on the source timeline.
        Returns:
            audio_label (int): Window-level audio label, real=0/fake=1.
            video_label (int): Window-level video label, real=0/fake=1.
                Overlap uses the complete num_frames * stride interval.
        """
        audio_label = data_info['audio_label']
        video_label = data_info['video_label']
        window_span_frames = self.data_config['num_frames'] * self.data_config['stride']
        if data_info.get('partial_fake_type') and data_info.get('fake_frame_range_list'):
            sample_start_frame, sample_end_frame = start_frame, start_frame + window_span_frames
            overlap_with_fake_segment = False
            for fake_frame_range in data_info['fake_frame_range_list']:
                overlap_start = max(sample_start_frame, fake_frame_range[0])
                overlap_end = min(sample_end_frame, fake_frame_range[1])
                if overlap_start < overlap_end:  # The sampled segment has overlap with the fake segment.
                    overlap_with_fake_segment = True
                    break
            if not overlap_with_fake_segment:  # If no overlap with fake segments, recover to corresponding real label.
                if data_info['partial_fake_type'] == 'audio-video':
                    audio_label, video_label = 0, 0
                elif data_info['partial_fake_type'] == 'audio':
                    audio_label = 0
                elif data_info['partial_fake_type'] == 'video':
                    video_label = 0
        return audio_label, video_label

    def _get_frame_labels(self, data_info, start_frame):
        """
        Label the video frames returned by fixed-stride sampling.
        Args:
            data_info (dict): Source video label and optional partial-forgery intervals.
            start_frame (int): Window start on the source timeline.
        Returns:
            frame_label (torch.Tensor): Long labels for num_frames sampled positions.
                Source labels and fake intervals determine each frame's authenticity.
        """
        num_returned_frames = self.data_config['num_frames']
        stride = self.data_config['stride']
        if data_info['video_label'] == 0:  # real
            frame_label = torch.zeros(num_returned_frames, dtype=torch.long)
        elif not data_info.get('partial_fake_type'):  # full fake
            frame_label = torch.ones(num_returned_frames, dtype=torch.long)
        elif 'video' in data_info['partial_fake_type']:  # pf on visual
            frame_label = torch.zeros(num_returned_frames, dtype=torch.long)
            fake_frame_range_list = data_info.get('fake_frame_range_list', [])
            for t in range(num_returned_frames):
                original_frame_idx = start_frame + t * stride
                for fake_range in fake_frame_range_list:
                    if fake_range[0] <= original_frame_idx < fake_range[1]:
                        frame_label[t] = 1
                        break
        elif data_info['partial_fake_type'] == 'audio':  # pf on audio, full fake on visual
            frame_label = torch.ones(num_returned_frames, dtype=torch.long)
        else:
            raise ValueError(f"Can't get corresponding frame label, path={data_info.get('path')!r}")
        return frame_label

    def _build_sample_result(self, data_info, start_frame, frames, audio, landmarks):
        """
        Compute window/frame labels and construct the returned sample.
        Args:
            data_info (dict): Indexed record with source authenticity labels
                and optional partial-forgery and demographic annotations.
            start_frame (int): Selected window start on the source timeline.
            frames (np.ndarray): Prepared frames in shape (T, H, W, C).
            audio (np.ndarray): Prepared waveform (L,) or features (Ta, F).
            landmarks (np.ndarray or None): Optional prepared coordinates.
        Returns:
            sample (dict): Float32 video (C, T, H, W), float32 audio, long
                frame_label (T,), scalar modality/overall labels, path and
                sample_frame_range. Includes demographic_group and float32
                landmarks (T, 68, 2) when available. Labels use real=0/fake=1.
        """
        audio_label, video_label = self._get_sample_labels(data_info, start_frame)
        frame_label = self._get_frame_labels(data_info, start_frame)
        window_span_frames = self.data_config['num_frames'] * self.data_config['stride']
        sample = {
            'video': torch.from_numpy(frames.transpose(3, 0, 1, 2)).float(),
            'audio': torch.from_numpy(audio).float(),
            'frame_label': frame_label,
            'video_label': video_label,
            'audio_label': audio_label,
            'label': 1 if audio_label == 1 or video_label == 1 else 0,
            'path': data_info['path'],
            'sample_frame_range': f'{start_frame}_{start_frame + window_span_frames}',
        }
        if data_info.get('demographic_group') is not None:
            sample['demographic_group'] = data_info['demographic_group']
        if self.data_config.get('with_landmark'):
            sample['landmarks'] = torch.from_numpy(landmarks).float()
        return sample

    def __getitem__(self, index):
        """
        Read one aligned sample using the current mode's sampling strategy.
        Args:
            index (int): Position in data_info_list. Train mode draws a fresh
                start within the indexed portion; val/test use its fixed start.
        Returns:
            sample (dict): Audio-video tensors, labels and sample identifiers
                described by _build_sample_result.
        """
        data_info = self.data_info_list[index]
        audio_path, video_path, landmark_path = self._get_data_paths(data_info)
        window_span_frames = self.data_config['num_frames'] * self.data_config['stride']
        if self.mode == 'train':
            start_frame = random.randint(data_info['start_frame'], data_info['end_frame'] - window_span_frames)
        else:
            start_frame = data_info['start_frame']
        frames, landmarks = self._load_video(video_path, landmark_path, start_frame, window_span_frames)
        audio = self._load_audio(audio_path, start_frame, window_span_frames)
        frames, audio, landmarks = self._apply_augmentation(frames, audio, landmarks)
        frames, audio, landmarks = self._prepare_model_inputs(frames, audio, landmarks)
        return self._build_sample_result(data_info, start_frame, frames, audio, landmarks)

    def __len__(self):
        """
        Return the number of indexed windows or training portions.
        Returns:
            num_samples (int): Length of data_info_list.
        """
        return len(self.data_info_list)


if __name__ == '__main__':
    from datasets.utils.data_helpers.video_helpers import video_denormalization
    from datasets.utils.data_presenters.audio_video_presenter import present_audio_video, save_audio_video

    # Configure paths in configs/path.yaml and choose a prepared local dataset.
    unit_test_params = {
        'mode': 'test',
        'dataset_name': 'Mega-MMDF',
        'data_config': {
            'frame_size': 256,
            'num_frames': 25,
            'stride': 1,
            'frame_mean': [0.5, 0.5, 0.5],
            'frame_std': [0.5, 0.5, 0.5],
        },
        'aug_config': None,
    }
    dataset = AudioVideoDataset(**unit_test_params)
    sample = dataset[0]
    # The presenter takes display-ready NumPy arrays; convert model tensors here.
    video = sample['video'].detach().cpu().numpy().transpose(1, 2, 3, 0)
    video = video_denormalization(video, unit_test_params['data_config']['frame_mean'],
                                 unit_test_params['data_config']['frame_std']).transpose(3, 0, 1, 2)
    audio = sample['audio'].detach().cpu().numpy()
    output_directory = Path('outputs/dataset_examples')
    output_directory.mkdir(parents=True, exist_ok=True)
    duration = (
        unit_test_params['data_config']['num_frames'] * unit_test_params['data_config']['stride'] / dataset.video_fps
    )
    present_audio_video(video, audio, duration, output_directory / 'sample.png')
    save_audio_video(
        video,
        audio,
        output_directory / 'sample.mp4',
        video_fps=dataset.video_fps / unit_test_params['data_config']['stride'],
        audio_sample_rate=dataset.audio_sr,
    )
    for key, value in sample.items():
        print(key, value.shape if isinstance(value, torch.Tensor) else value)
