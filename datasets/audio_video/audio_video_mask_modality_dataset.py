"""Random training masks and fixed test masks on raw audio-video signals."""

import numpy as np
import os
import random
import sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.audio_video.audio_video_dataset import AudioVideoDataset
from datasets.utils.augmentations.audio_augmentations import apply_audio_aug
from datasets.utils.augmentations.video_augmentations import apply_video_aug
from datasets.utils.data_helpers.audio_helpers import load_audio_segment
from datasets.utils.data_helpers.video_helpers import load_video_frames_fixed_stride


class AudioVideoMaskModalityDataset(AudioVideoDataset):
    def __init__(self, mode, data_config, aug_config=None, dataset_list=None, dataset_name=None, robustness_config=None):
        """
        Initialize ordinary sampling with optional modality masking.
        Args:
            mode (str): Dataset mode, options: ['train', 'val', 'test'].
            data_config (dict): Ordinary audio-video input and sampling settings.
            aug_config (str, dict or None): Training augmentation settings.
                audiovisual.mask_modality selects the masking type and probability.
            dataset_list (list): Dataset names combined in train mode.
            dataset_name (str): Dataset name read in val/test mode.
            robustness_config (dict or None): Fixed test masking configuration;
                type may be 'mask_audio' or 'mask_video'. Validation is unmasked.
        """
        self.robustness_config = robustness_config or {}
        super().__init__(mode, data_config, aug_config, dataset_list, dataset_name)

    def _get_modality_masks(self):
        """
        Decide which raw signals are unavailable for the current access.
        Returns:
            mask_audio (bool): Whether to replace the waveform with zeros.
            mask_video (bool): Whether to replace video with black frames.
                Training type 'all' draws the two decisions independently.
                These draws precede random training-window selection.
        """
        mask_audio = False
        mask_video = False
        if self.mode == 'test':
            if self.robustness_config.get('type') == 'mask_video':
                mask_video = True
            elif self.robustness_config.get('type') == 'mask_audio':
                mask_audio = True
        if self.mode == 'train' and self.aug_config.get('audiovisual'):
            if self.aug_config['audiovisual'].get('mask_modality'):
                mask_type = self.aug_config['audiovisual']['mask_modality']['type']
                probability = self.aug_config['audiovisual']['mask_modality'].get('prob', 0.5)
                if mask_type in ['all', 'audio'] and random.random() < probability:
                    mask_audio = True
                if mask_type in ['all', 'video'] and random.random() < probability:
                    mask_video = True
        return mask_audio, mask_video

    def _load_video(self, video_path, landmark_path, start_frame, window_span_frames, mask_video=False):
        """
        Read ordinary video or create raw black frames and zero landmarks.
        Args:
            video_path (str): Prepared video path.
            landmark_path (str): Corresponding landmark path.
            start_frame (int): Window start on the normalized source timeline.
            window_span_frames (int): Window duration in source frames.
            mask_video (bool): Whether video is unavailable for this access.
        Returns:
            frames (np.ndarray): Raw frames (T, H, W, C), with T=num_frames.
            landmarks (np.ndarray or None): Coordinates (T, 68, 2), zero-filled
                for masked video, or None when landmarks are not requested.
        """
        if not mask_video:
            return super()._load_video(video_path, landmark_path, start_frame, window_span_frames)
        frames = load_video_frames_fixed_stride(
            video_path=video_path,
            color_mode=self.data_config.get('color_mode', 'RGB'),
            sample_start_frame=start_frame,
            sample_end_frame=start_frame + window_span_frames,
            sample_stride=self.data_config['stride'],
            mask_video=True,
        )
        landmarks = None
        if self.data_config.get('with_landmark'):
            landmarks = np.zeros((window_span_frames // self.data_config['stride'], 68, 2), dtype=np.int32)
        return frames, landmarks

    def _load_audio(self, audio_path, start_frame, window_span_frames, mask_audio=False):
        """
        Read ordinary audio or create the zero waveform before feature conversion.
        Args:
            audio_path (str): Prepared mono audio path.
            start_frame (int): Window start on the normalized source timeline.
            window_span_frames (int): Window duration in source frames.
            mask_audio (bool): Whether audio is unavailable for this access.
        Returns:
            audio (np.ndarray): Waveform (L,) at audio_sr. A masked training
                access still draws the original training-delay random values,
                preserving subsequent augmentation and sampling randomness.
        """
        if not mask_audio:
            return super()._load_audio(audio_path, start_frame, window_span_frames)
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
            mask_audio=True,
        )

    def _apply_augmentation(self, frames, audio, landmarks, mask_audio=False, mask_video=False):
        """
        Apply ordinary training transforms only to the available modalities.
        Args:
            frames (np.ndarray): Raw video frames (T, H, W, C).
            audio (np.ndarray): Raw waveform (L,).
            landmarks (np.ndarray or None): Optional video coordinates.
            mask_audio (bool): Whether to skip waveform augmentation.
            mask_video (bool): Whether to skip video/landmark augmentation.
        Returns:
            frames (np.ndarray or list): Video frames after eligible transforms.
            audio (np.ndarray): Waveform after eligible transforms.
            landmarks (np.ndarray or None): Matching coordinates.
        """
        if self.mode == 'train':
            if self.video_aug_transform is not None and not mask_video:
                frames, landmarks = apply_video_aug(self.video_aug_transform, frames, landmarks)
            if self.audio_aug_transform is not None and not mask_audio:
                audio = apply_audio_aug(self.audio_aug_transform, audio, sr=self.audio_sr)
        return frames, audio, landmarks

    def _build_sample_result(self, data_info, start_frame, frames, audio, landmarks, mask_audio=False, mask_video=False):
        """
        Build effective labels while preserving pre-mask test-window labels.
        Args:
            data_info (dict): Source record and temporal forgery annotations.
            start_frame (int): Selected source-window start.
            frames (np.ndarray): Prepared frames (T, H, W, C).
            audio (np.ndarray): Prepared waveform (L,) or features (Ta, F).
            landmarks (np.ndarray or None): Optional prepared coordinates.
            mask_audio (bool): Whether audio's effective label becomes zero.
            mask_video (bool): Whether video/frame labels become zero.
        Returns:
            sample (dict): The ordinary tensor/label/identifier fields. Masked
                test samples additionally contain original_audio_label,
                original_video_label and original_label, computed after
                partial-forgery window labeling and before masking.
        """
        audio_label, video_label = self._get_sample_labels(data_info, start_frame)
        original_labels = {
            'original_audio_label': audio_label,
            'original_video_label': video_label,
            'original_label': max(audio_label, video_label),
        }
        if mask_audio:
            audio_label = 0
        if mask_video:
            video_label = 0
            frame_label = torch.zeros(self.data_config['num_frames'], dtype=torch.long)
        else:
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
        if self.mode == 'test' and (mask_audio or mask_video):
            sample.update(original_labels)
        if data_info.get('demographic_group') is not None:
            sample['demographic_group'] = data_info['demographic_group']
        if self.data_config.get('with_landmark'):
            sample['landmarks'] = torch.from_numpy(landmarks).float()
        return sample

    def __getitem__(self, index):
        """
        Read one sample with masks applied before representation/normalization.
        Args:
            index (int): Position in data_info_list.
        Returns:
            sample (dict): Audio-video tensors and effective labels, with
                pre-mask labels included for masked test inputs.
        """
        data_info = self.data_info_list[index]
        audio_path, video_path, landmark_path = self._get_data_paths(data_info)
        mask_audio, mask_video = self._get_modality_masks()
        window_span_frames = self.data_config['num_frames'] * self.data_config['stride']
        if self.mode == 'train':
            start_frame = random.randint(data_info['start_frame'], data_info['end_frame'] - window_span_frames)
        else:
            start_frame = data_info['start_frame']
        frames, landmarks = self._load_video(video_path, landmark_path, start_frame, window_span_frames, mask_video)
        audio = self._load_audio(audio_path, start_frame, window_span_frames, mask_audio)
        frames, audio, landmarks = self._apply_augmentation(frames, audio, landmarks, mask_audio, mask_video)
        frames, audio, landmarks = self._prepare_model_inputs(frames, audio, landmarks)
        return self._build_sample_result(data_info, start_frame, frames, audio, landmarks, mask_audio, mask_video)
