import torch

from losses.abstract_loss import AbstractLoss
from utils.registry import LOSSFUNC

@LOSSFUNC.register_module(module_name="mds_l1")
class MDSL1(AbstractLoss):
    def __init__(self):
        super().__init__()

    def forward(self, mds_score, targets):
        """
        Computes the cross-entropy loss.

        Args:
            mds_score: A PyTorch tensor of size (batch_size, num_classes) containing MDS scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        loss = (1 - targets) * mds_score ** 2 + targets * torch.max(0.99 - mds_score, torch.zeros_like(mds_score) ** 2)

        return loss.mean()