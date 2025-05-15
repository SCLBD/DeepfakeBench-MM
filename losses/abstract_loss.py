import torch.nn as nn


class AbstractLoss(nn.Module):
    """Abstract class for loss functions."""

    def __init__(self):
        super(AbstractLoss, self).__init__()

    def forward(self, pred, label):
        """
        Args:
            pred: prediction of the model
            label: ground truth label

        Return:
            loss: loss value
        """
        raise NotImplementedError('Each subclass should implement the forward method.')
