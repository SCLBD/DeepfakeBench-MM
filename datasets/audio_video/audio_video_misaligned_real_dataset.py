"""Real-only, within-clip circular reassignment of complete input windows."""

import os
import sys

from collections import defaultdict
from numbers import Integral

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.audio_video.audio_video_real_dataset import AudioVideoRealDataset


class AudioVideoMisalignedRealDataset(AudioVideoRealDataset):
    def __init__(self, mode, data_config, aug_config=None, dataset_list=None, dataset_name=None, robustness_config=None):
        """
        Initialize real-only test windows and reassign one modality within each clip.
        Args:
            mode (str): Must be 'test'.
            data_config (dict): Ordinary input settings plus optional misalignment
                settings: modality is 'audio' or 'video', and shift is an integer
                circular offset in complete windows, defaulting to audio and 1.
            aug_config (str, dict or None): Accepted by the common constructor;
                training augmentation is not applied in test mode.
            dataset_list (list or None): Unused in test mode.
            dataset_name (str): Dataset whose source-real records are evaluated.
            robustness_config (dict or None): Must not request an additional
                robustness transformation, preserving the original protocol.
        """
        if mode != 'test':
            raise ValueError('AudioVideoMisalignedRealDataset supports test mode only.')
        if (robustness_config or {}).get('type'):
            raise ValueError('Do not combine window misalignment with robustness interventions.')
        cfg = data_config.get('misalignment', {})
        self.shift_modality = cfg.get('modality', 'audio')
        self.shift = cfg.get('shift', 1)
        if self.shift_modality not in ('audio', 'video'):
            raise ValueError('misalignment.modality must be audio or video.')
        if isinstance(self.shift, bool) or not isinstance(self.shift, Integral):
            raise ValueError('misalignment.shift must be an integer window offset.')
        super().__init__(mode, data_config, aug_config, dataset_list, dataset_name)
        self.data_info_list = [
            sample
            for sample in self.data_info_list
            if all(sample[key] == 0 for key in ('label', 'audio_label', 'video_label'))
        ]
        self._assign_window_pairs()

    def _assign_window_pairs(self):
        """
        Annotate each anchor window with within-clip donor indices and ranges.
        Returns:
            None: Updates data_info_list in place and records misalignment_summary.
                Audio or video window i receives window (i + shift) % N from
                the same path. Single-window clips and identity shifts remain
                aligned. Duplicate or overlapping source windows raise ValueError.
        """
        groups = defaultdict(list)
        for sample in self.data_info_list:
            groups[sample['path']].append(sample)
        window_span_frames = self.data_config['num_frames'] * self.data_config['stride']
        for path, samples in groups.items():
            ordered = sorted(samples, key=lambda sample: sample['start_frame'])
            starts = [sample['start_frame'] for sample in ordered]
            if any(b - a < window_span_frames for a, b in zip(starts, starts[1:])):
                raise ValueError(f'Duplicate or overlapping source windows: {path}')
            for i, sample in enumerate(ordered):
                donor_index = (i + self.shift) % len(ordered)
                original = sample['start_frame']
                donor = starts[donor_index]
                sample.update(
                    audio_start_frame=donor if self.shift_modality == 'audio' else original,
                    video_start_frame=donor if self.shift_modality == 'video' else original,
                    misalignment_applied=donor != original,
                    misalignment_modality=self.shift_modality,
                    misalignment_shift=int(self.shift),
                    misalignment_shift_frames=donor - original,
                    original_sample_index=i,
                    donor_sample_index=donor_index,
                    num_samples_in_data=len(ordered),
                )
        self.misalignment_summary = {
            'num_data': len(groups),
            'num_samples': len(self.data_info_list),
            'num_shifted_samples': sum(sample['misalignment_applied'] for sample in self.data_info_list),
        }

    def _append_pairing_metadata(self, result, data_info):
        """
        Append actual modality ranges and pairing identifiers to one result.
        Args:
            result (dict): Ordinary sample result with the anchor's range.
            data_info (dict): Indexed window annotated by _assign_window_pairs.
        Returns:
            None: Updates result in place. sample_frame_range remains the anchor
                identifier; audio/video_sample_frame_range describe actual inputs.
        """
        window_span_frames = self.data_config['num_frames'] * self.data_config['stride']
        for modality in ('audio', 'video'):
            start = data_info[f'{modality}_start_frame']
            result[f'{modality}_sample_frame_range'] = f'{start}_{start + window_span_frames}'
        for key in (
            'misalignment_applied',
            'misalignment_modality',
            'misalignment_shift',
            'misalignment_shift_frames',
            'original_sample_index',
            'donor_sample_index',
            'num_samples_in_data',
        ):
            result[key] = data_info[key]

    def __getitem__(self, index):
        """
        Read the reassigned real audio/video pair while retaining its anchor ID.
        Args:
            index (int): Position in data_info_list.
        Returns:
            sample (dict): Ordinary tensors and source-real labels, plus actual
                modality ranges and pairing metadata. The source labels do not
                describe whether the pair is temporally aligned.
        """
        data_info = self.data_info_list[index]
        audio_path, video_path, landmark_path = self._get_data_paths(data_info)
        window_span_frames = self.data_config['num_frames'] * self.data_config['stride']
        frames, landmarks = self._load_video(video_path, landmark_path, data_info['video_start_frame'], window_span_frames)
        audio = self._load_audio(audio_path, data_info['audio_start_frame'], window_span_frames)
        frames, audio, landmarks = self._prepare_model_inputs(frames, audio, landmarks)
        result = self._build_sample_result(data_info, data_info['start_frame'], frames, audio, landmarks)
        self._append_pairing_metadata(result, data_info)
        return result
