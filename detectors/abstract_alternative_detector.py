import abc
import torch.nn as nn


class AbstractAlternativeDetector(nn.Module, metaclass=abc.ABCMeta):
    """Interface for detectors with separately optimized named stages."""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def build_backbone(self, config):
        """Construct the networks used by the configured optimization stages."""
        pass

    @abc.abstractmethod
    def get_stage_parameters(self, stage_name):
        """Return parameters optimized by the requested named stage."""
        pass

    @abc.abstractmethod
    def build_loss(self, config):
        """Construct the configured loss modules and optional weights."""
        pass

    @abc.abstractmethod
    def features(self, data_dict):
        """Extract features shared by the optimization stages."""
        pass

    @abc.abstractmethod
    def classifier(self, features):
        """Map extracted features to method-specific prediction outputs."""
        pass

    @abc.abstractmethod
    def forward(self, data_dict, stage_name, inference=False):
        """Return the prediction dictionary for the requested named stage."""
        pass

    @abc.abstractmethod
    def get_losses(self, data_dict: dict, pred_dict: dict, stage: str | None = None) -> dict:
        """Return stage-specific terms, including the scalar ``overall`` loss.

        Training supplies stage by keyword. When omitted during validation,
        the detector uses its documented validation objective.
        """
        pass
