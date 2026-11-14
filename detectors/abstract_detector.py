import abc
import torch.nn as nn


class AbstractDetector(nn.Module, metaclass=abc.ABCMeta):
    """Interface for detectors optimized with one training step."""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def build_backbone(self, config):
        """Construct the feature network from the detector configuration."""
        pass

    @abc.abstractmethod
    def build_loss(self, config):
        """Construct the configured loss modules and optional weights."""
        pass

    @abc.abstractmethod
    def features(self, data_dict):
        """Extract method-specific features from the input batch dictionary."""
        pass

    @abc.abstractmethod
    def classifier(self, features):
        """Map extracted features to the method-specific classification output."""
        pass

    @abc.abstractmethod
    def forward(self, data_dict, inference=False):
        """Return a method-specific prediction dictionary.

        Classification adapters expose a batch vector of fake scores as ``prob``.
        Input preparation, feature keys and use of ``inference`` vary by method.
        """
        pass

    @abc.abstractmethod
    def get_losses(self, data_dict, pred_dict):
        """Return loss terms, including the scalar ``overall`` optimized by the trainer."""
        pass
