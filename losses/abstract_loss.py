import torch.nn as nn


class AbstractLoss(nn.Module):
    """Base interface for detector loss modules."""

    def __init__(self):
        super(AbstractLoss, self).__init__()

    def forward(self, pred, label):
        """Compute a loss from detector outputs and targets.

        Args:
            pred: Model predictions in the subclass-specific format.
            label: Targets in the subclass-specific format.

        Returns:
            The loss value defined by the subclass.
        """
        raise NotImplementedError('Each subclass should implement the forward method.')
