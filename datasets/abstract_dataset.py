"""Common initialization and required interfaces for dataset implementations."""

import abc
import copy
import os
import sys
import yaml

from torch.utils.data import Dataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.dict_utils import deep_merge_dict


class AbstractDataset(Dataset, abc.ABC):
    default_aug = {}  # Declare modality-specific default augmentation configurations within modality-specific Dataset.

    def __init__(self, mode, data_config, aug_config=None):
        """
        Initialize common dataset settings without loading dataset records.
        Args:
            mode (str): Dataset mode, options: ['train', 'val', 'test'].
            data_config (dict): Input and sampling settings interpreted by the concrete dataset implementation.
            aug_config (str, dict or None): Training augmentation configuration.
                - 'default'(str) uses the current class's default_aug;
                - None or an empty dictionary disables augmentation;
                - A nonempty dictionary recursively overrides the defaults.
                See _resolve_augmentation_config() for the complete merge rules.
        """
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.data_config = data_config
        self.aug_config = aug_config
        with open(os.path.join(project_root, 'configs/path.yaml'), 'r') as f:
            self.path_config = yaml.safe_load(f)
        self.root_path = self.path_config['preprocessed_dir']

        if self.mode == 'train':
            self.aug_config = self._resolve_augmentation_config(aug_config)

    def _resolve_augmentation_config(self, aug_config):
        """
        Resolve training augmentation settings using this class's defaults.
        Args:
            aug_config (str, dict or None): One of the following:
                - 'default': Return a deep copy of self.default_aug.
                - None or {}: Return {}, disabling all training augmentation.
                - Nonempty dict: Deep-copy self.default_aug, then recursively merge the supplied settings. Unspecified
                    default fields are retained. A value replaces an existing field unless both values are dictionaries,
                    in which case their contents merge.
                Other types raise RuntimeError.
        Returns:
            resolved (dict): Resolved configuration for transform initialization. Subclasses can override the class
                attribute default_aug without changing this method.
        """
        if aug_config == 'default':
            resolved = copy.deepcopy(self.default_aug)
        elif aug_config is None:
            resolved = {}
        elif isinstance(aug_config, dict):
            resolved = copy.deepcopy(self.default_aug) if aug_config else {}
            deep_merge_dict(resolved, aug_config)
        else:
            raise RuntimeError('Unsupported augmentation settings, Please double check the config file.')
        return resolved

    @abc.abstractmethod
    def __getitem__(self, index):
        """
        Read one sample using the concrete dataset's input and label format.
        Args:
            index (int): Index of the sample to read.
        Returns:
            sample: Sample returned by the concrete dataset implementation.
        """
        pass

    @abc.abstractmethod
    def __len__(self):
        """
        Return the number of indexed samples in this dataset.
        Returns:
            num_samples (int): Number of samples available for indexed access.
        """
        pass
