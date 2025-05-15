import os
import sys
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from losses import LOSSFUNC
from losses.abstract_loss import AbstractLoss


@LOSSFUNC.register_module(module_name="cross_entropy")
class CrossEntropyLoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Computes the cross-entropy loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        # Compute the cross-entropy loss
        loss = self.loss_fn(inputs, targets)

        return loss