"""Source-real filtering with ordinary audio-video sampling and loading."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.audio_video.audio_video_dataset import AudioVideoDataset


class AudioVideoRealDataset(AudioVideoDataset):
    def __init__(self, mode, data_config, aug_config=None, dataset_list=None, dataset_name=None):
        """
        Initialize the ordinary dataset and retain source records labeled real.
        Args:
            mode (str): Dataset mode, options: ['train', 'val', 'test'].
            data_config (dict): Ordinary audio-video input and sampling settings.
            aug_config (str, dict or None): Ordinary training augmentation.
            dataset_list (list): Dataset names combined in train mode.
            dataset_name (str): Dataset name read in val/test mode.
        """
        super().__init__(mode, data_config, aug_config, dataset_list, dataset_name)
        self.data_info_list = self._filter_real_samples(self.data_info_list)

    @staticmethod
    def _filter_real_samples(samples):
        """
        Retain indexed windows whose source record has overall label zero.
        Args:
            samples (list): Indexed sample dictionaries with source label.
        Returns:
            real_samples (list): Source-real samples in their original order.
                Real windows inside a fake source record are not retained.
                Modality-specific source checks belong to specialized datasets.
        """
        return [sample for sample in samples if sample['label'] == 0]
