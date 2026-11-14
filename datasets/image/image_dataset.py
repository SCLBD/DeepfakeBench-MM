"""Planned image dataset implementation."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.abstract_dataset import AbstractDataset


class ImageDataset(AbstractDataset):
    dataset_type = 'image'

    # Todo: Implement modality-specific sampling, reading and returned fields.

    def __getitem__(self, index):
        """
        Read one image sample once this modality is implemented.
        Args:
            index (int): Sample index.
        Returns:
            sample: Not implemented yet.
        """
        raise NotImplementedError('ImageDataset has not been implemented.')

    def __len__(self):
        """
        Return the number of indexed samples once this modality is implemented.
        Returns:
            num_samples (int): Not implemented yet.
        """
        raise NotImplementedError('ImageDataset has not been implemented.')
